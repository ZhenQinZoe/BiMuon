"""Analytic linear-head derivatives and one shared stochastic hypergradient.

Only first derivatives of the feature network are needed: the mixed derivative
is represented as an analytic feature cotangent followed by a backbone VJP.
"""
import math
import torch
import torch.nn.functional as F

METHODS = ('exact', 'ns')


def probabilities(h, y):
    return (h @ y.T).softmax(dim=1)


def error(h, labels, y):
    p = probabilities(h, y)
    e = p.clone()
    e[torch.arange(len(h), device=h.device), labels] -= 1
    return p, e / len(h)


def head_grad(h, labels, y, lam):
    return error(h, labels, y)[1].T @ h + lam * y


def head_hvp(h, y, v, lam):
    p = probabilities(h, y)
    a = h @ v.T
    return (p * (a - (p * a).sum(1, keepdim=True))).T @ h / len(h) + lam * v


def mixed_cotangent(h, labels, y, v):
    p, e = error(h, labels, y)
    a = h @ v.T
    return e @ v + (p * (a - (p * a).sum(1, keepdim=True))) @ y / len(h)


def norm(tensors):
    return torch.sqrt(sum(t.square().sum() for t in tensors))


@torch.no_grad()
def direction(m, method, q=12, polar_rtol=1e-6):
    """Supported polar or Frobenius-normalized cubic Newton--Schulz direction."""
    if m.ndim != 2 or not m.is_floating_point():
        raise ValueError('Expected a floating-point matrix')
    if not torch.isfinite(m).all():
        raise ValueError('Nonfinite momentum')
    if q < 0 or not 0 <= polar_rtol < 1:
        raise ValueError('Invalid depth or polar support threshold')
    if method == 'exact':
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        mask = s > polar_rtol * s.max()
        return (u * mask) @ vh, int((~mask).sum())
    if method != 'ns':
        raise ValueError('method must be exact or ns')
    transposed = m.shape[0] > m.shape[1]
    x = m.T if transposed else m
    x = x / x.norm().clamp_min(torch.finfo(x.dtype).tiny)
    for _ in range(q):
        x = 1.5*x - 0.5*((x @ x.T) @ x)
    return (x.T if transposed else x), 0


class BiMuon:
    """Matrix UL update driven by a supplied stochastic hypergradient.

    The caller performs LL tracking and hypergradient estimation. Each matrix
    has its own EMA state. Defaults match the Transformer experiment.
    """
    def __init__(self, params, method='ns', lr=1e-3, momentum=.9, q=12, polar_rtol=1e-6):
        self.params = list(params)
        if not self.params or any(p.ndim != 2 for p in self.params):
            raise ValueError('At least one matrix parameter is required')
        if method not in METHODS or lr <= 0 or not math.isfinite(lr):
            raise ValueError('Invalid method or learning rate')
        if not 0 <= momentum < 1 or q < 0 or not 0 <= polar_rtol < 1:
            raise ValueError('Invalid momentum, depth, or polar threshold')
        self.method, self.lr, self.momentum = method, lr, momentum
        self.q, self.polar_rtol = q, polar_rtol
        self.m = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, hg):
        hg = list(hg)
        if len(hg) != len(self.params):
            raise ValueError('One hypergradient per parameter is required')
        if any(g.shape != p.shape or g.dtype != p.dtype or g.device != p.device
               for p,g in zip(self.params,hg)):
            raise ValueError('Hypergradient shape, dtype, and device must match')
        if not all(torch.isfinite(g).all() for g in hg):
            raise FloatingPointError('Nonfinite hypergradient')
        dropped_total = 0
        for p,m,g in zip(self.params,self.m,hg):
            m.mul_(self.momentum).add_(g, alpha=1-self.momentum)
            d,dropped = direction(m,self.method,self.q,self.polar_rtol)
            dropped_total += dropped
            p.add_(d, alpha=-self.lr)
        if not all(torch.isfinite(p).all() for p in self.params):
            raise FloatingPointError('Nonfinite UL parameters')
        return dropped_total

    def state_dict(self):
        return {'momentum':[m.detach().clone() for m in self.m],
                'method':self.method,'lr':self.lr,'coefficient':self.momentum,
                'q':self.q,'polar_rtol':self.polar_rtol}

    def load_state_dict(self, state):
        expected=(self.method,self.lr,self.momentum,self.q,self.polar_rtol)
        actual=(state['method'],state['lr'],state['coefficient'],state['q'],state['polar_rtol'])
        if actual != expected or len(state['momentum']) != len(self.m):
            raise ValueError('Optimizer configuration mismatch')
        for dst,src in zip(self.m,state['momentum']):
            if dst.shape != src.shape:
                raise ValueError('Momentum shape mismatch')
        for dst,src in zip(self.m,state['momentum']):
            dst.copy_(src.to(dst))


def estimate(features, params, y, streams, cfg):
    """Fresh batch for every oracle; no LL unroll or differentiation through v."""
    y = y.detach()
    # Fixed-X head features may be reused inside this step. The cache is fully
    # recomputed after every UL update; direct and mixed still build live graphs.
    if hasattr(streams, 'begin_step'):
        streams.begin_step(features)
    def head_batch(role):
        if hasattr(streams, 'next_head'):
            return streams.next_head(role)
        x, labels = streams.next(role)
        return features(x), labels
    with torch.no_grad():
        for _ in range(cfg['D']):
            h, labels = head_batch('ll')
            y = y - cfg['alpha'] * head_grad(h, labels, y, cfg['lam'])
    x, labels = streams.next('ul')
    h = features(x)
    with torch.no_grad():
        _, e = error(h.detach(), labels, y)
        r = e.T @ h.detach()
        cot = e @ y
    direct = torch.autograd.grad(h, params, grad_outputs=cot)
    with torch.no_grad():
        v = cfg['eta'] * r
        for _ in range(cfg['Q']):
            h, _ = head_batch('hvp')
            r = r - cfg['eta'] * head_hvp(h,y,r,cfg['lam'])
            v = v + cfg['eta'] * r
    x, labels = streams.next('mixed')
    h = features(x)
    with torch.no_grad():
        cot = mixed_cotangent(h.detach(), labels, y, v)
    mixed = torch.autograd.grad(h, params, grad_outputs=cot)
    return [a-b for a,b in zip(direct,mixed)], y.detach()


def refit(h, labels, lam=.01, initial=None, tol=1e-4, max_iter=400):
    """Independent full-data head solve in float64 on fixed FP32 features."""
    h = h.detach().double()
    y = (torch.zeros(100,h.shape[1],device=h.device,dtype=h.dtype)
         if initial is None else initial.detach().double().clone())
    y.requires_grad_(True)
    opt = torch.optim.LBFGS([y],lr=1.,max_iter=max_iter,history_size=30,
                           tolerance_grad=1e-10,tolerance_change=1e-14,
                           line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(h @ y.T,labels) + .5*lam*y.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    y = y.detach()
    residual = float(head_grad(h,labels,y,lam).norm())
    return y, dict(residual=residual, gap_bound=residual**2/(2*lam),
                   converged=math.isfinite(residual) and residual<=tol)
