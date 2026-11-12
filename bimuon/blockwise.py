"""Relative block stepsizes under a unit theoretical LL-drift budget."""
import torch

def relative_stepsizes(a, w):
    if a.ndim != 1 or a.shape != w.shape or a.numel() == 0:
        raise ValueError('Matching nonempty score and sensitivity vectors are required')
    if not torch.isfinite(a).all() or not torch.isfinite(w).all() or (w<=0).any():
        raise ValueError('Nonfinite score/weight or nonpositive sensitivity')
    if (a<0).any():
        raise ValueError('Exact-polar scores must be nonnegative')
    if not (a>0).any():
        return torch.zeros_like(a)
    raw = a/w
    # Algebraically identical to raw/sqrt(sum(w*raw^2)); avoids unnecessary
    # under/overflow and preserves exact unit budget without epsilon inflation.
    raw = raw/raw.max()
    return raw/torch.sqrt((w*raw.square()).sum())
