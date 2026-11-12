import pytest
import torch
import torch.nn.functional as F
from bimuon import BiMuon, direction, relative_stepsizes
from bimuon.core import head_grad, head_hvp, mixed_cotangent, estimate

torch.set_num_threads(1)


def test_head_derivatives_match_autograd():
    torch.manual_seed(19)
    h=torch.randn(11,4,dtype=torch.float64,requires_grad=True)
    y=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    labels=torch.randint(3,(11,));v=torch.randn_like(y)
    loss=F.cross_entropy(h@y.T,labels)+.005*y.square().sum()
    gy=torch.autograd.grad(loss,y,create_graph=True)[0]
    hv,mixed=torch.autograd.grad((gy*v).sum(),(y,h))
    torch.testing.assert_close(head_grad(h,labels,y,.01),gy)
    torch.testing.assert_close(head_hvp(h,y,v,.01),hv)
    torch.testing.assert_close(mixed_cotangent(h,labels,y,v),mixed)


@pytest.mark.parametrize('shape',[(4,7),(7,4),(4,4)])
def test_polar_and_ns_rectangular_invariants(shape):
    torch.manual_seed(7)
    m=torch.randn(*shape,dtype=torch.float64)
    polar,_=direction(m,'exact');ns,_=direction(m,'ns',q=30)
    gram=polar@polar.T if shape[0]<=shape[1] else polar.T@polar
    torch.testing.assert_close(gram,torch.eye(min(shape),dtype=m.dtype))
    torch.testing.assert_close(ns,polar,rtol=1e-9,atol=1e-9)
    for method in ['exact','ns']:
        z,_=direction(torch.zeros_like(m),method)
        assert torch.count_nonzero(z)==0
        d,_=direction(m,method)
        assert float(torch.linalg.matrix_norm(d,ord=2))<=1+1e-12


def test_supported_polar_rank_deficient():
    m=torch.diag(torch.tensor([3.,1.,0.],dtype=torch.float64))
    d,dropped=direction(m,'exact')
    torch.testing.assert_close(d,torch.diag(torch.tensor([1.,1.,0.],dtype=m.dtype)))
    assert dropped==1


def test_allocation_budget_and_optimality():
    a=torch.tensor([2.,5.,1.],dtype=torch.float64)
    w=torch.tensor([1.,9.,100.],dtype=torch.float64)
    omega=relative_stepsizes(a,w)
    torch.testing.assert_close((w*omega.square()).sum(),torch.tensor(1.,dtype=w.dtype))
    torch.testing.assert_close((a*omega).sum(),torch.sqrt((a.square()/w).sum()))
    assert torch.count_nonzero(relative_stepsizes(torch.zeros_like(a),w))==0
    with pytest.raises(ValueError):relative_stepsizes(a,torch.zeros_like(w))


@pytest.mark.parametrize('method',['exact','ns'])
def test_optimizer_checkpoint_continuation(method):
    torch.manual_seed(21)
    p=torch.nn.Parameter(torch.randn(4,7,dtype=torch.float64))
    opt=BiMuon([p],method=method,lr=.03)
    opt.step([torch.randn_like(p)])
    q=torch.nn.Parameter(p.detach().clone())
    resumed=BiMuon([q],method=method,lr=.03);resumed.load_state_dict(opt.state_dict())
    for _ in range(4):
        g=torch.randn_like(p);opt.step([g]);resumed.step([g])
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    with pytest.raises(ValueError):opt.step([])


def test_implicit_estimator_matches_independent_tanh_kernel():
    from experiments import representation as e
    from experiments.data import DatasetBundle
    torch.manual_seed(11)
    X=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
    Y=torch.randn(3,4,dtype=torch.float64)
    data=DatasetBundle(torch.randn(21,6,dtype=torch.float64),torch.randint(3,(21,)),
        torch.randn(13,6,dtype=torch.float64),torch.randint(3,(13,)),None,None,None,None)
    cfg=dict(D=3,Q=4,alpha=.1,eta=.2,lam=.01,batch=5)
    class Stream:
        def __init__(self):self.gen=torch.Generator().manual_seed(91)
        def next(self,role):
            xx,yy=(data.x_val,data.y_val) if role=='ul' else (data.x_train,data.y_train)
            return e.batch(xx,yy,5,self.gen)
    hg,y=estimate(lambda x:torch.tanh(x@X.T),[X],Y,Stream(),cfg)
    expected,ey=e.estimate(data,X.detach(),Y,cfg,torch.Generator().manual_seed(91))
    torch.testing.assert_close(hg[0],expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(y,ey,rtol=1e-12,atol=1e-12)
