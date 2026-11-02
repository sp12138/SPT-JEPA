"""Read all already-complete endpoints from one immutable coverage snapshot."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics


def main(inventory, output):
    inventory, output = Path(inventory), Path(output)
    snapshot = json.loads(inventory.read_text()); rows=[]
    for entry in snapshot['rows']:
        job, terminal = entry['job'], entry['terminal']
        if job['task'] != 'pusht' or not terminal or terminal.get('outcome') != 'nominal_budget_complete':continue
        base = Path(entry['base'])/'evaluations'/job['id']/str(job['stages'][-1])
        variants=['raw','bn'] if job['model_class']=='base' else ['planning']
        arms=[]
        for variant in variants:
            metrics=[]
            for seed in [42,43,44,1042,1043,1044]:
                path=base/variant/f'plan{seed}'/'metrics.json';raw=path.read_bytes();value=json.loads(raw)
                assert value['seed']==seed and len(value['metrics']['episode_successes'])==50
                metrics.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),value=value))
            arms.append(dict(variant=variant,primary=statistics.mean(m['value']['metrics']['success_rate'] for m in metrics[:3]),
                fresh=statistics.mean(m['value']['metrics']['success_rate'] for m in metrics[3:]),metrics=metrics))
        rows.append(dict(id=job['id'],track=job['track'],block=job['block'],arm=job['arm'],seed=job['seed'],steps=job['stages'][-1],model=job['model'],arms=arms))
    output.mkdir(parents=True,exist_ok=False)
    report=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_utc=snapshot['checked_utc'],
        snapshot_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),counts=snapshot['counts'],rows=rows,
        scope='All already-completed models only; preserve allseeds/cases/domain variants. Different stagebudgets not pooled. No training/model/GPU/simulationcalls.')
    (output/'completed-performance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed=len(rows),round2=[dict(id=r['id'],steps=r['steps'],arms=[{k:v for k,v in a.items() if k!='metrics'} for a in r['arms']]) for r in rows if r['track']=='round2'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.inventory,a.output)
