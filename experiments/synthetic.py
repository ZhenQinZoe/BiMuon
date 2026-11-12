"""Run the synthetic representation-learning experiment."""
import argparse
import json
from pathlib import Path
from . import representation as e


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/synthetic.json')
    p.add_argument('--method',choices=e.METHODS,default='ns')
    p.add_argument('--kappa',type=int,default=10)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--steps',type=int,help='Override horizon for a development check')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    protocol=json.loads(Path(a.config).read_text())
    setting=next(s for s in protocol['settings'] if s['algorithm']['kappa']==a.kappa)
    cfg=dict(setting['algorithm'])
    if a.steps is not None:
        if a.steps <= 0:p.error('--steps must be positive')
        cfg['T']=a.steps
    result=e.run(cfg,a.seed,a.method,setting['learning_rates'][a.method],a.output)
    print(json.dumps(result,indent=2))
    if not result['stable']:raise SystemExit('Run diverged; outputs kept')

if __name__=='__main__':main()
