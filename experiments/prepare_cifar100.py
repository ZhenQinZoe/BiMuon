"""Extract and cache frozen ResNet-50 features for CIFAR-100."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from experiments.io import save, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--config',default='configs/cifar100_blockwise.json');ap.add_argument('--data-dir',default='work/cifar100_data');a=ap.parse_args();out=Path(a.cache);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text())
    assert torch.cuda.is_available(),'CUDA GPU required for full feature extraction'
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        assert done['config']==cfg['data']
        for f,h in done['files'].items():assert sha(out/f)==h,f
        print('CACHE VERIFIED',out,flush=True);return
    w=ResNet50_Weights.IMAGENET1K_V2;tf=w.transforms();model=resnet50(weights=w).cuda().eval();model.fc=torch.nn.Identity()
    for p in model.parameters():p.requires_grad_(False)
    torch.backends.cudnn.benchmark=False
    for split,is_train in [('train',True),('test',False)]:
        if (out/f'{split}_complete.json').exists():continue
        ds=torchvision.datasets.CIFAR100(root=a.data_dir,train=is_train,download=True,transform=tf)
        dl=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=4,pin_memory=True)
        f=np.lib.format.open_memmap(out/f'{split}_features.npy',mode='w+',dtype='float32',shape=(len(ds),2048));offset=0;started=time.time()
        with torch.inference_mode():
            for x,y in dl:
                z=model(x.cuda(non_blocking=True)).cpu().numpy();f[offset:offset+len(z)]=z;offset+=len(z)
                if offset%4096<128:print('CACHE',split,offset,len(ds),round(time.time()-started),flush=True)
        f.flush();del f;np.save(out/f'{split}_labels.npy',np.asarray(ds.targets,dtype=np.int64));save(out/f'{split}_complete.json',dict(count=offset,elapsed=time.time()-started))
    y=np.load(out/'train_labels.npy');rng=np.random.RandomState(cfg['data']['split_seed']);tr=[];va=[]
    for c in range(100):
        ids=rng.permutation(np.flatnonzero(y==c));tr.extend(ids[:400].tolist());va.extend(ids[400:].tolist())
    assert len(tr)==40000 and len(va)==10000 and not set(tr)&set(va)
    np.savez(out/'split_indices.npz',train=np.array(tr),validation=np.array(va))
    feat=np.load(out/'train_features.npy',mmap_mode='r');rms=float(np.sqrt(np.mean(np.asarray(feat[tr],dtype=np.float64)**2)))
    weights_file=Path(torch.hub.get_dir())/'checkpoints'/Path(w.url).name
    save(out/'metadata.json',dict(dataset='CIFAR100',backbone='resnet50',weights='IMAGENET1K_V2',weights_sha256=sha(weights_file),preprocess=str(tf),feature_dim=2048,cache_dtype='float32',feature_scale=rms,scaling='one scalar RMS fitted only on LL-train; no whitening/centering',split_seed=cfg['data']['split_seed'],train_count=40000,validation_count=10000,test_count=10000,test_policy='cached once, excluded from validation metrics and stepsize selection',config=cfg['data']))
    fs=['train_features.npy','test_features.npy','train_labels.npy','test_labels.npy','split_indices.npz','metadata.json']
    save(out/'complete.json',dict(config=cfg['data'],files={f:sha(out/f) for f in fs}));print('CACHE COMPLETE',flush=True)
if __name__=='__main__':main()
