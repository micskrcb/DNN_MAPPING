"""Paired learning attribution at a longer search horizon."""
import argparse,json
from pathlib import Path
from analyze_priority_seeds import analyze

def summarize(root,seeds):
    return analyze(root,seeds,methods=['train_more','frozen','fixed_sa'],
                   challenger='train_more',controls=['frozen','fixed_sa'],
                   allowed_changes=['ppo_asa_disable_learning','report','asa_diagnostics'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[25,26,27]);a=p.parse_args()
    r=summarize(a.directory,a.seeds)
    (a.directory/'learning-horizon-summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    print(json.dumps(r,indent=2))
