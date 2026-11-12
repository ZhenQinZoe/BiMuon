"""Real-data runs with isolated refits, auditable budgets and failure records."""
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from bimuon import core as c
from .task import Data,Features,Streams,configure,extract,frozen_digest,make_model,save,sha

BASE=dict(lam=.01,D=5,Q=1024,alpha=.1,eta=.2,batch=64,T=1000,eval_every=250)


def load(path):
    return json.loads(Path(path).read_text())


def synchronize():
    torch.cuda.synchronize()


def environment():
    import timm,torchvision,numpy
    return dict(python=platform.python_version(),torch=torch.__version__,
                torchvision=torchvision.__version__,timm=timm.__version__,
                numpy=numpy.__version__,cuda=torch.version.cuda,
                gpu_model=torch.cuda.get_device_name())


def setup(root,workers=2):
    root=Path(root); configure()
    model,digest=make_model(root/'cache')
    data=Data(root/'data',model,workers=workers)
    return model,data,digest


def prepare(root,workers=2):
    root=Path(root)
    if (root/'initial.pt').exists():
        raise FileExistsError('Initial state already exists; preserve it, do not overwrite')
    model,data,digest=setup(root,workers)
    feat=Features(model)
    start=time.monotonic()
    h,labels=extract(feat,data.loader('ll'))
    scale=float(h.square().mean().sqrt()); feat.scale=scale
    h=h/scale
    y,diagnostic=c.refit(h,labels)
    if not diagnostic['converged']:
        raise RuntimeError(f'Initial LL refit unresolved: {diagnostic}')
    hv,yv=extract(feat,data.loader('ul'))
    ce=float(F.cross_entropy(hv.double()@y.T,yv))
    state=dict(Y=y.float().cpu(),scale=scale,refit=diagnostic,initial_ce=ce,
               initial_val_accuracy=float((hv.double()@y.T).argmax(1).eq(yv).float().mean()),
               checkpoint_sha256=digest,split_sha256=data.metadata['split_sha256'])
    torch.save(state,root/'initial.pt')
    save(root/'data_manifest.json',data.metadata)
    save(root/'prepare.json',dict(**{k:v for k,v in state.items() if k!='Y'},
                                  seconds=time.monotonic()-start,environment=environment()))
    print(json.dumps(dict(event='PREPARED',initial_ce=ce,scale=scale,refit=diagnostic)),flush=True)


def context(root,workers=2):
    root=Path(root)
    model,data,digest=setup(root,workers)
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    if digest!=initial['checkpoint_sha256'] or data.metadata['split_sha256']!=initial['split_sha256']:
        raise ValueError('Data split or pretrained checkpoint changed')
    features=Features(model,initial['scale'])
    params=[p for p in model.parameters() if p.requires_grad]
    data.cache_frozen_prefix(model)
    return model,data,initial,features,params


def evaluate(features,data,y,cfg,step):
    h,labels=extract(features,data.loader('ll'))
    # The solver copy never replaces training y.
    ref,diag=c.refit(h,labels,cfg['lam'],initial=y)
    if not diag['converged']:
        ref,diag=c.refit(h,labels,cfg['lam'],initial=ref,max_iter=1200)
    finite_res=float(c.head_grad(h,labels,y,cfg['lam']).norm())
    del h
    hv,yv=extract(features,data.loader('ul'))
    optimized=float(F.cross_entropy(hv.double()@ref.T,yv))
    finite=float(F.cross_entropy(hv@y.T,yv))
    row=dict(step=step,optimized_val_ce=optimized,finite_D_val_ce=finite,
             finite_D_ll_residual=finite_res,lower_residual=diag['residual'],
             lower_gap_bound=diag['gap_bound'],refit_converged=diag['converged'],
             val_accuracy=float((hv.double()@ref.T).argmax(1).eq(yv).float().mean()))
    if not diag['converged']:
        raise RuntimeError(f'Checkpoint LL refit unresolved: {row}')
    return row


def run(root,out,method,seed,lr,cfg,workers=2,refits=True,warmup=20):
    if method not in c.METHODS:raise ValueError('method must be exact or ns')
    root,out=Path(root),Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    save(out/'config.json',dict(method=method,seed=seed,lr=lr,algorithm=cfg,
                               refits=refits,warmup=warmup,environment=environment()))
    save(out/'status.json',dict(status='running',start_unix=time.time()))
    streams=None
    try:
        model,data,initial,features,params=context(root,workers)
        frozen_before=frozen_digest(model)
        x0=[p.detach().clone() for p in params]
        y0=initial['Y'].cuda()
        probe_x,_=next(iter(data.loader('ul',batch=4)))
        probe_x=probe_x.cuda()
        with torch.no_grad(): probe_h=features(probe_x).clone()
        # Warm kernels and optimizer allocations using disposable state, then
        # restore *all* trainable/head/momentum/stream states before timing.
        if warmup:
            ws=Streams(data,900000+seed,cfg['batch'])
            wu=c.Update(params,method,lr); wy=y0.clone()
            for _ in range(warmup):
                hg,wy=c.estimate(features,params,wy,ws,cfg); wu.step(hg)
            ws.close(); del ws,wu,wy,hg
            with torch.no_grad():
                for p,p0 in zip(params,x0): p.copy_(p0)
        y=y0.clone(); update=c.Update(params,method,lr)
        streams=Streams(data,seed,cfg['batch'])
        curves=[dict(step=0,optimized_val_ce=initial['initial_ce'],
                     lower_residual=initial['refit']['residual'],refit_converged=True)]
        synchronize(); torch.cuda.reset_peak_memory_stats()
        optimization_seconds=0.; direction_seconds=0.; peak_alloc=0; peak_reserved=0
        total_dropped=0
        updates_file=(out/'updates.csv').open('w')
        writer=csv.DictWriter(updates_file,fieldnames=['step','hg_fro','direction_seconds','rank_dropped'])
        writer.writeheader()
        block_start=time.monotonic()
        for step in range(1,cfg['T']+1):
            hg,y=c.estimate(features,params,y,streams,cfg)
            # Per-update synchronization is shared by every method; included in
            # optimization time and disclosed in the measurement protocol.
            synchronize(); tick=time.monotonic()
            dropped=update.step(hg)
            synchronize(); dt=time.monotonic()-tick
            direction_seconds+=dt; total_dropped+=dropped
            writer.writerow(dict(step=step,hg_fro=float(c.norm(hg)),direction_seconds=dt,rank_dropped=dropped))
            if step%10==0:
                updates_file.flush()
                print(json.dumps(dict(event='STEP',method=method,seed=seed,step=step,T=cfg['T'])),flush=True)
            checkpoint=step==cfg['T'] or (refits and step%cfg['eval_every']==0)
            if checkpoint:
                synchronize(); optimization_seconds+=time.monotonic()-block_start
                peak_alloc=max(peak_alloc,torch.cuda.max_memory_allocated())
                peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
                if refits:
                    row=evaluate(features,data,y,cfg,step); curves.append(row)
                    save(out/'curves.json',curves)
                    print(json.dumps(dict(event='EVAL',method=method,seed=seed,**row)),flush=True)
                synchronize(); torch.cuda.reset_peak_memory_stats(); block_start=time.monotonic()
        updates_file.close()
        with torch.no_grad():
            feature_delta=float((features(probe_x)-probe_h).norm())
            weight_delta=float(c.norm([p-p0 for p,p0 in zip(params,x0)]))
        frozen_after=frozen_digest(model)
        if frozen_before!=frozen_after or weight_delta<=0 or feature_delta<=0:
            raise RuntimeError('Frozen-weight or learned-representation invariant failed')
        torch.save(dict(trainable={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                        Y=y.cpu(),optimizer=update.state_dict(),streams=streams.state_dict(),
                        initial_sha256=sha(root/'initial.pt'),config=cfg),out/'final_state.pt')
        result=dict(status='complete',method=method,seed=seed,lr=lr,steps=cfg['T'],
                    final_ce=curves[-1]['optimized_val_ce'] if refits else None,
                    optimization_seconds=optimization_seconds,seconds_per_step=optimization_seconds/cfg['T'],
                    direction_seconds=direction_seconds,peak_allocated_bytes=peak_alloc,
                    peak_reserved_bytes=peak_reserved,end_to_end_seconds=time.monotonic()-started,
                    frozen_unchanged=True,ul_weight_delta=weight_delta,probe_feature_delta=feature_delta,
                    discarded_singular_count_sum=total_dropped,test_accessed=False,
                    oracle_counts=dict(LL_gradients=cfg['T']*cfg['D'],HVPs=cfg['T']*cfg['Q'],
                                       upper_pairs=cfg['T'],mixed_VJPs=cfg['T']),streams=streams.state_dict())
        save(out/'result.json',result); save(out/'status.json',result)
        print(json.dumps(dict(event='RUN_COMPLETE',**result)),flush=True)
        return result
    except Exception as exc:
        save(out/'status.json',dict(status='failed',error_type=type(exc).__name__))
        raise
    finally:
        if streams is not None: streams.close()
