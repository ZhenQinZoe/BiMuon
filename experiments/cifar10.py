"""Frozen ResNet-18 features with BiMuon representation learning."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from bimuon.core import direction
from .data import DatasetBundle
from . import representation as e
from .io import save


def prepare(data_dir, cache_path, pool_seed):
    from torchvision.datasets import CIFAR10
    from torchvision.models import resnet18,ResNet18_Weights
    cache_path=Path(cache_path)
    if cache_path.exists():raise FileExistsError(cache_path)
    weights=ResNet18_Weights.IMAGENET1K_V1
    ds=CIFAR10(data_dir,train=True,download=True,transform=weights.transforms())
    labels=np.asarray(ds.targets)
    rng=np.random.RandomState(pool_seed)
    ids=np.concatenate([rng.permutation(np.flatnonzero(labels==c))[:150] for c in range(10)])
    model=resnet18(weights=weights).eval();model.fc=torch.nn.Identity()
    for parameter in model.parameters():parameter.requires_grad_(False)
    features=[]
    with torch.inference_mode():
        for off in range(0,len(ids),32):
            batch=torch.stack([ds[int(i)][0] for i in ids[off:off+32]])
            features.append(model(batch).cpu().numpy())
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,features=np.concatenate(features),labels=labels[ids],
                        official_train_indices=ids,pool_seed=np.int64(pool_seed))


def make_state(cache,cfg,seed):
    gen=torch.Generator().manual_seed(20000+seed)
    labels=torch.as_tensor(cache['labels'],dtype=torch.long)
    tr,va=[],[]
    for cls in range(10):
        ids=torch.where(labels==cls)[0]
        if len(ids)!=150:raise ValueError('Expected 150 examples per class')
        ids=ids[torch.randperm(len(ids),generator=gen)]
        tr.extend(ids[:100].tolist());va.extend(ids[100:].tolist())
    features=torch.as_tensor(cache['features']).double()
    scale=float(features[tr].square().mean().sqrt());features=features/scale
    data=DatasetBundle(features[tr],labels[tr],features[va],labels[va],
                       torch.empty(0,512),torch.empty(0,dtype=torch.long),None,None)
    torch.manual_seed(30000+seed)
    X=.1*torch.randn(cfg['d'],cfg['p'],dtype=e.DT)/np.sqrt(cfg['p'])
    Y=torch.zeros(cfg['c'],cfg['d'],dtype=e.DT)
    return data,X,Y


def run(cache_path,cfg,seed,method,lr,out):
    if method not in e.METHODS:raise ValueError('method must be exact or ns')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with np.load(cache_path,allow_pickle=False) as cache:
        data,X,Y=make_state(cache,cfg,seed)
    M=torch.zeros_like(X);gen=torch.Generator().manual_seed(100000+seed)
    save(out/'config.json',dict(algorithm=cfg,seed=seed,method=method,lr=lr))
    curves=[]
    def evaluate(step):
        yr,res=e.refit(data,X,cfg)
        loss=e.loss(data.x_val,data.y_val,X,yr)
        if not math.isfinite(loss) or not math.isfinite(res):
            raise FloatingPointError('Nonfinite evaluation')
        curves.append(dict(step=step,optimized_val_ce=loss,lower_residual=res,
                           finite_D_val_ce=e.loss(data.x_val,data.y_val,X,Y)))
        e.write_csv(out/'curves.csv',curves)
    evaluate(0)
    for k in range(cfg['T']):
        hg,Y=e.estimate(data,X,Y,cfg,gen)
        M=cfg['momentum']*M+(1-cfg['momentum'])*hg
        d=e.polar(M) if method=='exact' else direction(M,'ns',cfg['q'])[0]
        X=X-lr*d
        if not torch.isfinite(X).all():raise FloatingPointError('Nonfinite UL state')
        if (k+1)%cfg['eval_every']==0 or k+1==cfg['T']:evaluate(k+1)
    torch.save(dict(X=X,Y=Y,M=M),out/'final_state.pt')
    save(out/'result.json',dict(final_ce=curves[-1]['optimized_val_ce'],seed=seed,
                              method=method,steps=cfg['T'],test_accessed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--data-dir',default='work/data')
    prep.add_argument('--cache',default='work/cifar10_features.npz')
    prep.add_argument('--pool-seed',type=int,default=20270921)
    r=sub.add_parser('run');r.add_argument('--cache',required=True)
    r.add_argument('--config',default='configs/cifar10.json')
    r.add_argument('--method',choices=e.METHODS,default='ns')
    r.add_argument('--seed',type=int,default=20);r.add_argument('--steps',type=int)
    r.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.data_dir,a.cache,a.pool_seed);return
    protocol=json.loads(Path(a.config).read_text());cfg=dict(protocol['algorithm'])
    if a.steps is not None:
        if a.steps<=0:p.error('--steps must be positive')
        cfg['T']=a.steps
    run(a.cache,cfg,a.seed,a.method,protocol['learning_rates'][a.method],a.output)

if __name__=='__main__':main()
