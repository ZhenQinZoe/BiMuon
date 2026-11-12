"""Thin data/task adapters around the unchanged Section 7.1 estimator.
The exact same engine.estimate recurrence runs both tasks; blockwise uses
specialized CE/proximal derivatives. No backbone gradients are constructed.
"""
import contextlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from experiments import representation as e
from experiments.data import DatasetBundle
from bimuon.core import direction as matrix_direction
from bimuon.blockwise import relative_stepsizes
from experiments.io import save,sha
DT=torch.float64

def load_data(cache,device,test=False):
    p=Path(cache);meta=json.loads((p/'metadata.json').read_text());ids=np.load(p/'split_indices.npz')
    f=torch.as_tensor(np.load(p/'train_features.npy'),device=device,dtype=DT)/meta['feature_scale'];y=torch.as_tensor(np.load(p/'train_labels.npy'),device=device)
    tr=torch.as_tensor(ids['train'],device=device);va=torch.as_tensor(ids['validation'],device=device)
    if test:
        xt=torch.as_tensor(np.load(p/'test_features.npy'),device=device,dtype=DT)/meta['feature_scale'];yt=torch.as_tensor(np.load(p/'test_labels.npy'),device=device)
    else:xt=yt=None
    return DatasetBundle(f[tr],y[tr],f[va],y[va],xt,yt,None,None)

@contextlib.contextmanager
def task_adapter(kind,c,mu):
    """Plug task derivatives into the original solver/Neumann recurrence."""
    if kind=='realdata':yield;return
    old=(e.deriv,e.hvp,e.mixed)
    def deriv(x,label,X,Y,lam=0.):
        p=torch.softmax(x@Y.T,1);r=p.clone();r[torch.arange(len(x),device=x.device),label]-=1
        return torch.zeros_like(X),r.T@x/len(x)+lam*(Y-X*c)
    def hvp(x,X,Y,v,lam):
        p=torch.softmax(x@Y.T,1);a=x@v.T
        return (p*(a-(p*a).sum(1,keepdim=True))).T@x/len(x)+mu*v
    def mixed(x,label,X,Y,v):return -mu*c*v
    e.deriv,e.hvp,e.mixed=deriv,hvp,mixed
    try:yield
    finally:e.deriv,e.hvp,e.mixed=old

def design(data,kind,X,c):
    with torch.no_grad():
        if kind=='realdata':return torch.tanh(data.x_train@X.T),torch.tanh(data.x_val@X.T),torch.zeros(100,X.shape[0],device=X.device,dtype=DT)
        return data.x_train,data.x_val,X*c

def refit(h,labels,center,mu,start,cfg):
    Y=(center.clone() if start is None else start.clone()).detach().requires_grad_(True)
    opt=torch.optim.LBFGS([Y],lr=1,max_iter=cfg['refit_max_iter'],tolerance_grad=1e-10,tolerance_change=1e-14,history_size=30,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad();total=0.;calls+=1
        for lo in range(0,len(h),cfg['refit_chunk']):
            hi=min(lo+cfg['refit_chunk'],len(h));loss=torch.nn.functional.cross_entropy(h[lo:hi]@Y.T,labels[lo:hi],reduction='sum')/len(h);loss.backward();total+=float(loss.detach())
        reg=.5*mu*(Y-center).square().sum();reg.backward();return torch.as_tensor(total,device=h.device,dtype=h.dtype)+reg.detach()
    opt.step(closure);closure();res=float(Y.grad.norm());return Y.detach(),res,calls

def ce(h,y,Y):
    return float(torch.nn.functional.cross_entropy(h@Y.T,y))
def metrics(h,y,Y):
    logits=h@Y.T;return dict(ce=float(torch.nn.functional.cross_entropy(logits,y)),accuracy=float((logits.argmax(1)==y).double().mean()))
def init(data,kind,seed,alg):
    gen=torch.Generator().manual_seed(seed)
    if kind=='realdata':X=.1*torch.randn(alg['d'],alg['p'],generator=gen,dtype=DT)/math.sqrt(alg['p']);Y=torch.zeros(alg['c'],alg['d'],dtype=DT)
    else:X=torch.zeros(100,2048,dtype=DT);Y=torch.zeros_like(X)
    return X.to(data.x_train.device),Y.to(data.x_train.device)
def columns(rho,device):return torch.tensor([rho**(b/3) for b in range(4)],device=device,dtype=DT).repeat_interleave(512)
def direction(M,method,q):return matrix_direction(M,'ns',q)[0] if method=='ns' else e.direction(M,method,q)
def spectrum(M,cfg):
    s=torch.linalg.svdvals(M);th=max(cfg['spectral_abs_threshold'],cfg['spectral_rel_threshold']*float(s[0]));rank=int((s>th).sum())
    return s.detach().cpu().numpy(),dict(sigma_max=float(s[0]),sigma_min=float(s[-1]),support_threshold=th,robust_rank=rank,spread_p90_p10=float(torch.quantile(s,.9)/torch.quantile(s,.1).clamp_min(th)),stable_rank=float(s.square().sum()/s[0].square().clamp_min(1e-30)))

def run(data,cfg,kind,method,seed,beta,rho,T,dest,phase):
    if (kind=='blockwise' and method!='proposed') or (kind=='realdata' and method not in ('exact','ns')):raise ValueError('Unsupported BiMuon method')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);alg=dict(cfg['algorithm']);mu=alg['lam'] if kind=='realdata' else cfg['block_mu'];device=data.x_train.device
    if kind=='blockwise':
        L=.5*float(data.x_train.square().sum(1).max())+mu
        alg['alpha']=min(alg['alpha'],1/L);alg['eta']=min(alg['eta'],1/L)
    save(dest/'config.json',dict(protocol=cfg,actual_algorithm=alg,kind=kind,method=method,seed=seed,beta=beta,rho=rho,T=T,phase=phase))
    c=columns(rho,device) if kind=='blockwise' else None;w=100*torch.tensor([rho**(b/3) for b in range(4)],device=device,dtype=DT).square() if c is not None else None
    X,Y=init(data,kind,seed,alg);M=torch.zeros_like(X);gen=torch.Generator().manual_seed(100000+seed)
    curves=[];rows=[];spectra=[];last_ref=None;last_X=None;last_step=0;refcalls=0;started=time.time();status='complete';error=None
    save(dest/'attempt.json',dict(status='running',started=started))
    def evaluate(k):
        nonlocal last_ref,last_X,last_step,refcalls
        h,hv,center=design(data,kind,X,c);yr,res,n=refit(h,data.y_train,center,mu,last_ref,cfg);refcalls+=n
        row=dict(step=k,optimized_val_ce=ce(hv,data.y_val,yr),finite_D_val_ce=ce(hv,data.y_val,Y),ll_residual=res,ll_solution_distance_bound=res/mu,ll_suboptimality_bound=res*res/(2*mu),tracking_distance_to_approx_ystar=float((Y-yr).norm()),tracking_squared_to_approx_ystar=float((Y-yr).square().sum()),refit_converged=res<=cfg['refit_tolerance'])
        if last_ref is not None:
            row['target_drift_since_last_eval']=float((yr-last_ref).norm());row['target_drift_interval']=k-last_step
            row['target_drift_error_bound']=res/mu+curves[-1]['ll_solution_distance_bound']
            if kind=='blockwise':row['target_drift_upper_bound_from_actual_X']=float(torch.sqrt(sum(w[b]*(X[:,512*b:512*(b+1)]-last_X[:,512*b:512*(b+1)]).square().sum() for b in range(4))))
        curves.append(row);last_ref=yr;last_X=X.clone();last_step=k
        return yr
    try:
        evaluate(0)
        with task_adapter(kind,c,mu):
            for k in range(T):
                with torch.no_grad():hg,Y=e.estimate(data,X,Y,alg,gen);M=alg['momentum']*M+(1-alg['momentum'])*hg
                if not torch.isfinite(M).all() or not torch.isfinite(Y).all():raise FloatingPointError('nonfinite hypergradient/LL state')
                row=dict(step=k+1,momentum_fro=float(M.norm()),hg_fro=float(hg.norm()))
                if kind=='realdata':
                    s,sp=spectrum(M,cfg);spectra.append(s);row.update(sp);O=direction(M,method,alg['q']);delta=-beta*O
                    if k in (0,T//2,T-1):
                        exact=e.polar(M);ns=matrix_direction(M,'ns',alg['q'])[0];row['ns_relative_polar_error']=float((ns-exact).norm()/exact.norm().clamp_min(1e-30))
                else:
                    blocks=[M[:,b*512:(b+1)*512] for b in range(4)];os=[e.polar(m) for m in blocks];a=torch.stack([(m*o).sum() for m,o in zip(blocks,os)])
                    omega=relative_stepsizes(a,w);delta=-beta*torch.cat([omega[b]*os[b] for b in range(4)],1)
                    budget=float((w*omega.square()).sum());zero=not bool((a>0).any());assert abs(budget-(0 if zero else 1))<1e-9
                    row.update(budget=budget,all_zero_progress=zero,theoretical_target_drift_bound=beta*math.sqrt(budget),allocation_progress=float((a*omega).sum()))
                    specs=[]
                    for b in range(4):
                        sb,sp=spectrum(blocks[b],cfg);specs.append(sb)
                        row.update({f'b{b}_{key}':val for key,val in sp.items()});row.update({f'b{b}_a':float(a[b]),f'b{b}_w':float(w[b]),f'b{b}_omega':float(omega[b]),f'b{b}_update_fro':float(delta[:,b*512:(b+1)*512].norm()),f'b{b}_update_op':float(torch.linalg.matrix_norm(delta[:,b*512:(b+1)*512],2))})
                    spectra.append(np.stack(specs))
                row['update_fro']=float(delta.norm());X=X+delta;rows.append(row)
                if not torch.isfinite(X).all() or float(X.norm())>1e8:raise FloatingPointError('unstable UL state')
                if (k+1)%cfg['eval_every']==0 or k+1==T:
                    evaluate(k+1);e.write_csv(dest/'trajectory.csv',curves);e.write_csv(dest/'diagnostics.csv',rows);np.savez_compressed(dest/'spectrum.npz',singular_values=np.stack(spectra));print(kind,phase,rho,method,seed,beta,k+1,round(curves[-1]['optimized_val_ce'],6),'res',format(curves[-1]['ll_residual'],'.2g'),flush=True)
        if not all(r['refit_converged'] for r in curves):status='refit_unresolved'
    except (FloatingPointError,RuntimeError) as ex:
        status='failed';error=str(ex)
    result=dict(kind=kind,phase=phase,method=method,seed=seed,beta=beta,rho=rho,status=status,error=error,completed_steps=len(rows),T=T,initial_ce=curves[0]['optimized_val_ce'] if curves else None,final_ce=curves[-1]['optimized_val_ce'] if status in ['complete','refit_unresolved'] and len(rows)==T else None,seconds=time.time()-started,refit_closure_calls=refcalls,derivative_budget=dict(ll_gradients=len(rows)*alg['D'],HVP=len(rows)*alg['Q'],mixed_VJP=len(rows),upper_gradient_pair=len(rows)),max_refit_residual=max([r['ll_residual'] for r in curves],default=None))
    if phase=='confirmation' and len(rows)==T and status!='failed':
        ht=torch.tanh(data.x_test@X.T) if kind=='realdata' else data.x_test
        result.update({f'test_{k}':v for k,v in metrics(ht,data.y_test,last_ref).items()})
    e.write_csv(dest/'trajectory.csv',curves);e.write_csv(dest/'diagnostics.csv',rows)
    if spectra:np.savez_compressed(dest/'spectrum.npz',singular_values=np.stack(spectra))
    torch.save(dict(X=X.cpu(),Y=Y.cpu(),M=M.cpu(),approx_ystar=last_ref.cpu() if last_ref is not None else None,rng_state=gen.get_state()),dest/'final_state.pt')
    save(dest/'result.json',result);save(dest/'attempt.json',dict(status=status,finished=time.time(),error=error));return result
