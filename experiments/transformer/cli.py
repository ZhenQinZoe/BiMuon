"""Prepare or run the BiMuon DeiT-S/16 experiment on one selected GPU."""
import argparse
import json
from pathlib import Path
import torch
from .runner import prepare,run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--root',required=True)
    p.add_argument('--config',default='configs/transformer.json')
    p.add_argument('--method',choices=['exact','ns'],default='ns')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output')
    p.add_argument('--steps',type=int,help='Development horizon override')
    a=p.parse_args()
    if not torch.cuda.is_available():p.error('The full Transformer experiment requires CUDA')
    if a.command=='prepare':prepare(a.root,a.workers);return
    if a.output is None:p.error('--output is required for run')
    protocol=json.loads(Path(a.config).read_text());cfg=dict(protocol['algorithm'])
    if a.steps is not None:
        if a.steps<=0:p.error('--steps must be positive')
        cfg['T']=a.steps
    run(a.root,a.output,a.method,a.seed,protocol['learning_rates'][a.method],cfg,a.workers)

if __name__=='__main__':main()
