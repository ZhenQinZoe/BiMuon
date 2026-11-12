import numpy as np
import torch
from experiments.cifar10 import make_state
from experiments.transformer.task import ALLOWLIST
from experiments import representation as e


def test_cifar10_split_scaling_and_seed_repeatability():
    rng=np.random.RandomState(8)
    cache=dict(features=rng.randn(1500,512).astype('float32'),labels=np.repeat(np.arange(10),150))
    cfg=dict(d=256,p=512,c=10)
    a,x,y=make_state(cache,cfg,20);b,xx,yy=make_state(cache,cfg,20)
    assert a.x_train.shape==(1000,512) and a.x_val.shape==(500,512)
    assert x.shape==(256,512) and y.shape==(10,256)
    torch.testing.assert_close(a.x_train.square().mean(),torch.tensor(1.,dtype=torch.float64))
    torch.testing.assert_close(a.x_train,b.x_train,rtol=0,atol=0)
    torch.testing.assert_close(x,xx,rtol=0,atol=0)
    torch.testing.assert_close(y,yy,rtol=0,atol=0)


def test_transformer_parameter_scope():
    assert len(ALLOWLIST)==16
    assert sum(a*b for a,b in ALLOWLIST.values())==7077888
    assert {int(k.split('.')[1]) for k in ALLOWLIST}=={8,9,10,11}


def test_synthetic_capped_ns_is_not_silently_normalized():
    m=torch.diag(torch.tensor([.1,.05],dtype=torch.float64))
    result=e.direction(m,'ns',1)
    expected=1.5*m-.5*m@(m.T@m)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)


def test_blockwise_derivatives_keep_coupling():
    from experiments.cifar100_core import task_adapter
    import torch.nn.functional as F
    torch.manual_seed(15)
    x=torch.randn(11,8,dtype=torch.float64)
    labels=torch.randint(3,(11,))
    X=torch.randn(3,8,dtype=torch.float64,requires_grad=True)
    Y=torch.randn_like(X,requires_grad=True)
    v=torch.randn_like(Y)
    c=torch.linspace(1.,3.,8,dtype=torch.float64);mu=.01
    loss=F.cross_entropy(x@Y.T,labels)+.5*mu*(Y-X*c).square().sum()
    gy=torch.autograd.grad(loss,Y,create_graph=True)[0]
    hv,mixed=torch.autograd.grad((gy*v).sum(),(Y,X))
    original=e.deriv
    with task_adapter('blockwise',c,mu):
        torch.testing.assert_close(e.deriv(x,labels,X,Y,mu)[1],gy)
        torch.testing.assert_close(e.hvp(x,X,Y,v,mu),hv)
        torch.testing.assert_close(e.mixed(x,labels,X,Y,v),mixed)
    assert e.deriv is original
