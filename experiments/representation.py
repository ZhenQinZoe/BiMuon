"""Current paper estimator, analytic derivatives checked against autograd.
Shared tanh representation kernel; only the two BiMuon variants are retained.
"""
import math,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from .data import DatasetBundle

torch.set_num_threads(1)
DT=torch.float64
METHODS=['exact','ns']
BASE=dict(p=16,d=32,c=4,rank=4,n_train=256,n_val=128,n_test=128,noise_std=.02,lam=.01,alpha=.1,eta=.2,D=5,Q=4,q=12,momentum=.9,T=60,batch=64,kappa=1.,eval_every=5)

def dataset(cfg,seed):
    torch.manual_seed(seed)
    w=.8*(torch.randn(cfg['d'],cfg['rank'],dtype=DT)@torch.randn(cfg['p'],cfg['rank'],dtype=DT).T)/math.sqrt(cfg['rank'])
    y=torch.randn(cfg['c'],cfg['d'],dtype=DT)/math.sqrt(cfg['d'])
    # Independent fixed rotation does not consume the teacher/data RNG stream.
    gen=torch.Generator().manual_seed(7301)
    u=torch.linalg.qr(torch.randn(cfg['p'],cfg['p'],generator=gen,dtype=DT))[0]
    s=torch.logspace(0,-math.log10(cfg['kappa']),cfg['p'],dtype=DT); s=s/torch.sqrt((s*s).mean())
    a=(u*s)@u.T if cfg['kappa']!=1 else torch.eye(cfg['p'],dtype=DT)
    samples=[]
    for n in [cfg['n_train'],cfg['n_val'],cfg['n_test']]:
        x=torch.randn(n,cfg['p'],dtype=DT)@a
        logits=3*torch.tanh(x@w.T)@y.T+cfg['noise_std']*torch.randn(n,cfg['c'],dtype=DT)
        samples.extend([x,logits.argmax(1)])
    x0=.1*torch.randn(cfg['d'],cfg['p'],dtype=DT)
    return DatasetBundle(*samples,w,y),x0

def stats(x,label,X,Y):
    h=torch.tanh(x@X.T); p=torch.softmax(h@Y.T,dim=1)
    e=p.clone(); e[torch.arange(len(x),device=x.device),label]-=1
    return h,p,e/len(x)

def deriv(x,label,X,Y,lam=0.):
    h,p,e=stats(x,label,X,Y)
    return ((e@Y)*(1-h*h)).T@x,e.T@h+lam*Y

def hvp(x,X,Y,v,lam):
    h=torch.tanh(x@X.T); p=torch.softmax(h@Y.T,1); a=h@v.T
    return (p*(a-(p*a).sum(1,keepdim=True))).T@h/len(x)+lam*v

def mixed(x,label,X,Y,v):
    h,p,e=stats(x,label,X,Y); a=h@v.T
    return ((e@v+(p*(a-(p*a).sum(1,keepdim=True)))@Y/len(x))*(1-h*h)).T@x

def batch(x,y,size,gen):
    if size==0:return x,y
    idx=torch.randint(len(x),(size,),generator=gen)
    return x[idx],y[idx]

def estimate(data,X,Y,cfg,gen):
    for _ in range(cfg['D']):
        x,y=batch(data.x_train,data.y_train,cfg['batch'],gen)
        Y=Y-cfg['alpha']*deriv(x,y,X,Y,cfg['lam'])[1]
    x,y=batch(data.x_val,data.y_val,cfg['batch'],gen)
    direct,r=deriv(x,y,X,Y); v=cfg['eta']*r
    for _ in range(cfg['Q']):
        x,_=batch(data.x_train,data.y_train,cfg['batch'],gen)
        r=r-cfg['eta']*hvp(x,X,Y,r,cfg['lam']); v=v+cfg['eta']*r
    x,y=batch(data.x_train,data.y_train,cfg['batch'],gen)
    return direct-mixed(x,y,X,Y,v),Y

def polar(m):
    u,s,v=torch.linalg.svd(m,full_matrices=False)
    return (u*(s>1e-14*s.max()).to(DT))@v

def direction(m,method,q):
    if method=='exact':return polar(m)
    if method=='ns':
        o=m/max(1.,float(m.norm()))
        for _ in range(q):o=1.5*o-.5*o@(o.T@o)
        return o
    raise ValueError(method)

def refit(data,X,cfg):
    # Cached features; evaluation does not alter the warm-start training state.
    h=torch.tanh(data.x_train@X.T)
    Y=torch.zeros(cfg['c'],cfg['d'],dtype=DT,device=X.device,requires_grad=True)
    opt=torch.optim.LBFGS([Y],lr=1.,max_iter=300,tolerance_grad=1e-10,tolerance_change=1e-14,history_size=50,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(h@Y.T,data.y_train)+.5*cfg['lam']*(Y*Y).sum();loss.backward();return loss
    opt.step(closure)
    Y=Y.detach(); residual=float(deriv(data.x_train,data.y_train,X,Y,cfg['lam'])[1].norm())
    return Y,residual

def loss(x,y,X,Y):
    return float(torch.nn.functional.cross_entropy(torch.tanh(x@X.T)@Y.T,y))

def run(cfg,seed,method,lr,out=None,diagnostics=False):
    if method not in METHODS:raise ValueError('method must be exact or ns')
    if out is not None and Path(out).exists():raise FileExistsError(out)
    start=time.time(); data,X=dataset(cfg,seed); Y=torch.zeros(cfg['c'],cfg['d'],dtype=DT); M=torch.zeros_like(X)
    gen=torch.Generator().manual_seed(100000+seed)
    curves=[]; updates=[]; states=[]
    initial,_=refit(data,X,cfg)
    curves.append(dict(step=0,val_loss=loss(data.x_val,data.y_val,X,initial),lower_residual=0.))
    stable=True
    for k in range(cfg['T']):
        grad,Y=estimate(data,X,Y,cfg,gen)
        M=cfg['momentum']*M+(1-cfg['momentum'])*grad
        delta=-lr*direction(M,method,cfg['q'])
        if not torch.isfinite(delta).all() or delta.norm()>1e8:
            stable=False;break
        updates.append(dict(step=k+1,delta_fro=float(delta.norm()),delta_op=float(torch.linalg.matrix_norm(delta,2)),momentum_fro=float(M.norm()),momentum_op=float(torch.linalg.matrix_norm(M,2)),delta_rms=float(torch.sqrt((delta*delta).mean())),val_loss=loss(data.x_val,data.y_val,X,Y)))
        if diagnostics and k in [0,cfg['T']//2,cfg['T']-1]:states.append(dict(step=k,X=X.clone(),Y=Y.clone(),HG=grad.clone(),M=M.clone()))
        X=X+delta
        if (k+1)%cfg['eval_every']==0 or k==cfg['T']-1:
            yr,residual=refit(data,X,cfg)
            curves.append(dict(step=k+1,val_loss=loss(data.x_val,data.y_val,X,yr),lower_residual=residual))
    vals=[r['val_loss'] for r in curves]; steps=[r['step'] for r in curves]
    final=vals[-1] if stable else float('inf')
    result=dict(seed=seed,method=method,lr=lr,final_val_loss=final,best_val_loss=min(vals),auc=float(np.trapz(vals,steps)/cfg['T']) if stable else float('inf'),stable=stable,elapsed_sec=time.time()-start,max_lower_residual=max(r['lower_residual'] for r in curves),**{f'target_{t}':next((r['step'] for r in curves if r['val_loss']<=t),cfg['T']+1) for t in [.2,.3,.5]})
    if diagnostics:
        yr,_=refit(data,X,cfg); result['test_loss']=loss(data.x_test,data.y_test,X,yr)
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        (out/'result.json').write_text(json.dumps(result,indent=2)); (out/'config.json').write_text(json.dumps({**cfg,'seed':seed,'method':method,'lr':lr},indent=2))
        write_csv(out/'curves.csv',curves);write_csv(out/'updates.csv',updates)
        if diagnostics:torch.save(dict(states=states,final_X=X,final_Y=Y,data=data),out/'checkpoints.pt')
    return result

def write_csv(path,rows):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
