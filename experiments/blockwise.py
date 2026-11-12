"""Proposed relative stepsizes on the controlled CIFAR-100 task."""
import argparse
import json
from pathlib import Path
import torch
from .cifar100_core import load_data,run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cifar100_blockwise.json')
    p.add_argument('--cache',required=True)
    p.add_argument('--rho',type=float,choices=[3,10],default=10,
                   help='Sensitivity ratio (varrho in the manuscript)')
    p.add_argument('--seed',type=int,default=41001)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--steps',type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA is unavailable')
    cfg=json.loads(Path(a.config).read_text())
    data=load_data(a.cache,a.device)
    steps=a.steps if a.steps is not None else cfg['T']
    if steps<=0:p.error('--steps must be positive')
    result=run(data,cfg,a.seed,cfg['global_stepsize'],a.rho,steps,a.output)
    print(json.dumps(result,indent=2))
    if result['status']!='complete':raise SystemExit('Run failed or LL refit did not converge; outputs kept')

if __name__=='__main__':main()
