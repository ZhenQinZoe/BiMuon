"""Pinned DeiT task, deterministic transforms, independent fresh-batch streams."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

URL = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
LAYERS = {'attn.qkv.weight':(1152,384),'attn.proj.weight':(384,384),
          'mlp.fc1.weight':(1536,384),'mlp.fc2.weight':(384,1536)}
ALLOWLIST = {f'blocks.{b}.{n}':s for b in range(8,12) for n,s in LAYERS.items()}


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def configure():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def make_model(cache, device='cuda'):
    import timm
    cache = Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    weights = cache/'deit_small_patch16_224-cd65a155.pth'
    if not weights.exists():
        tmp = weights.with_suffix('.download')
        torch.hub.download_url_to_file(URL,str(tmp),hash_prefix='cd65a155')
        os.replace(tmp,weights)
    digest = sha(weights)
    if not digest.startswith('cd65a155'):
        raise ValueError('Pretrained checkpoint hash mismatch')
    model = timm.create_model('deit_small_patch16_224.fb_in1k',pretrained=False)
    state = torch.load(weights,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    model.reset_classifier(0)
    params = []
    for name,p in model.named_parameters():
        p.requires_grad_(name in ALLOWLIST)
        if name in ALLOWLIST:
            assert tuple(p.shape)==ALLOWLIST[name],(name,p.shape)
            params.append(p)
    assert len(params)==16 and sum(p.numel() for p in params)==7077888
    return model.eval().to(device), digest


class Features:
    def __init__(self,model,scale=1.):
        self.model,self.scale = model,float(scale)

    def __call__(self,x):
        if x.ndim == 3:
            # Only the *frozen* prefix is cached across outer iterations.
            for block in self.model.blocks[8:]:
                x = block(x)
            return self.model.norm(x)[:,0,:] / self.scale
        return self.model.forward_features(x)[:,0,:] / self.scale


@torch.no_grad()
def frozen_prefix(model,x):
    x=model.patch_embed(x)
    x=model._pos_embed(x)
    x=model.patch_drop(x)
    x=model.norm_pre(x)
    for block in model.blocks[:8]:
        x=block(x)
    return x


class Data:
    def __init__(self,root,model,workers=2):
        from torchvision.datasets import CIFAR100
        from timm.data import create_transform,resolve_model_data_config
        root = Path(root)
        cfg = resolve_model_data_config(model)
        transform = create_transform(**cfg,is_training=False)
        self.dataset = CIFAR100(str(root),train=True,download=True,transform=transform)
        labels = torch.tensor(self.dataset.targets)
        generator = torch.Generator().manual_seed(20261107)
        ll,ul = [],[]
        for label in range(100):
            idx = torch.where(labels==label)[0]
            assert len(idx)==500
            idx = idx[torch.randperm(len(idx),generator=generator)]
            ll.extend(idx[:400].tolist()); ul.extend(idx[400:].tolist())
        self.ll_idx,self.ul_idx = ll,ul
        self.ll,self.ul = Subset(self.dataset,ll),Subset(self.dataset,ul)
        self.workers=workers
        self.metadata=dict(split_seed=20261107,ll_indices=ll,ul_indices=ul,
                           transform=repr(transform),test_loaded=False,
                           archive_sha256=sha(root/'cifar-100-python.tar.gz'))
        self.metadata['split_sha256']=hashlib.sha256(json.dumps([ll,ul]).encode()).hexdigest()

    def loader(self,role,batch=256):
        if role in getattr(self,'tokens',{}):
            return ((self.tokens[role][i:i+batch],self.cached_labels[role][i:i+batch])
                    for i in range(0,len(self.tokens[role]),batch))
        return DataLoader(getattr(self,role),batch_size=batch,shuffle=False,
                          num_workers=self.workers,pin_memory=True)

    @torch.no_grad()
    def cache_frozen_prefix(self,model):
        self.tokens={}; self.cached_labels={}
        for role in ['ll','ul']:
            loader=self.loader(role)
            tokens=torch.empty((len(getattr(self,role)),197,384),device='cuda',dtype=torch.float32)
            labels=torch.empty(len(tokens),device='cuda',dtype=torch.long)
            offset=0
            for x,y in loader:
                x=x.cuda(non_blocking=True)
                tokens[offset:offset+len(x)].copy_(frozen_prefix(model,x))
                labels[offset:offset+len(x)].copy_(y)
                offset+=len(x)
            self.tokens[role]=tokens; self.cached_labels[role]=labels


class FreshSampler:
    def __init__(self,n,batch,seed):
        self.n,self.batch,self.seed=n,batch,seed

    def __iter__(self):
        gen=torch.Generator().manual_seed(self.seed)
        while True:
            yield torch.randint(self.n,(self.batch,),generator=gen).tolist()


class Streams:
    def __init__(self,data,seed,batch,device='cuda'):
        self.seed,self.batch,self.device=seed,batch,device
        self.counts={r:0 for r in ('ll','ul','hvp','mixed')}
        self.loaders={}
        self.iters={}
        self.data=data
        if getattr(data,'tokens',None):
            self.generators={role:torch.Generator().manual_seed(100000+100*seed+i)
                             for i,role in enumerate(self.counts)}
            self.head_cache=None
            return
        for i,role in enumerate(self.counts):
            ds=data.ul if role=='ul' else data.ll
            loader=DataLoader(ds,batch_sampler=FreshSampler(len(ds),batch,100000+100*seed+i),
                              num_workers=data.workers,pin_memory=True,
                              generator=torch.Generator().manual_seed(800000+100*seed+i))
            self.loaders[role]=loader
            self.iters[role]=iter(loader)

    def next(self,role):
        if getattr(self.data,'tokens',None):
            key,idx=self.indices(role)
            return self.data.tokens[key][idx],self.data.cached_labels[key][idx]
        x,labels=next(self.iters[role]); self.counts[role]+=1
        return x.to(self.device,non_blocking=True),labels.to(self.device,non_blocking=True)

    def indices(self,role):
        key='ul' if role=='ul' else 'll'
        idx=torch.randint(len(self.data.tokens[key]),(self.batch,),generator=self.generators[role]).to(self.device)
        self.counts[role]+=1
        return key,idx

    @torch.no_grad()
    def begin_step(self,features):
        self.features=features
        if getattr(self.data,'tokens',None):
            # Exactly current X; never cache trainable outputs across UL updates.
            self.head_cache=torch.cat([features(x) for x,_ in self.data.loader('ll',batch=512)])

    def next_head(self,role):
        if getattr(self.data,'tokens',None):
            key,idx=self.indices(role)
            assert key=='ll' and self.head_cache is not None
            return self.head_cache[idx],self.data.cached_labels[key][idx]
        x,labels=self.next(role)
        return self.features(x),labels

    def state_dict(self):
        # Worker prefetch advances sampler ahead. Replay from seed and these
        # *consumed* counts, not the sampler's prefetched generator state.
        return dict(seed=self.seed,batch=self.batch,consumed_batches=dict(self.counts))

    def close(self):
        self.iters.clear(); self.loaders.clear()


@torch.no_grad()
def extract(features,loader,device='cuda'):
    hs,ys=[],[]
    for x,y in loader:
        hs.append(features(x.to(device,non_blocking=True)).detach())
        ys.append(y.to(device,non_blocking=True))
    return torch.cat(hs),torch.cat(ys)


def frozen_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
