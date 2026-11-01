"""All24 completed cells of body x corruption x task x seed;no selection."""
import argparse
from pathlib import Path

import numpy as np

from common import B, read, sha, write


def interval(x):
    x = np.asarray(x, dtype=float)
    ids = np.random.default_rng(20261101).integers(len(x), size=(10000,len(x)))
    return dict(mean_pp=float(x.mean()), ci95_pp=np.quantile(x[ids].mean(1), [.025,.975]).tolist())


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', required=True);a = p.parse_args()
    base = B / 'runs/unified_three_20261101'
    ordinary = B / 'runs/unified_spatial_checkpoint600_20261101'
    high = B / 'runs/unified_spatial_highnoise600_20261101'
    rows, bindings = [], {}
    for task in ['pusht','tworoom','cube']:
        for seed in [1,2]:
            inputs = None
            for body, shift, root, arm in [('flat',0.,base,'flat'),('spatial',0.,ordinary,'field'),
                                           ('flat',.8,high,'flat'),('spatial',.8,high,'field')]:
                ident = f'{task}-{arm}-seed{seed}';run = root / ident
                terminal = root / 'worker-records' / (ident+'-gate.json');t = read(terminal)
                assert t['passed'] and len(t['results'])==3 and all(x['returncode']==0 for x in t['results'])
                cfg = read(run/'config.json');assert cfg['model'].get('flow_logit_shift',0.)==shift
                assert cfg['model_class']==('spatial_carrier' if body=='flat' else 'spatial_native')
                ap = root/'audits'/f'{ident}-step600/audit.json';audit=read(ap)
                assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore'] and audit['state_before']==audit['state_after']
                assert sha(run/'full-archives/step0000600.ckpt')==audit['full_checkpoint_sha256']
                with np.load(ap.with_name('state-audit.npz')) as f:
                    current = (f['observed'].copy(),f['actions'].copy())
                    if inputs is None:inputs=current
                    else:assert all(np.array_equal(x,y) for x,y in zip(inputs,current))
                mp = root/'evaluations'/ident/'metrics.json';m=read(mp);values=m['metrics']['episode_successes']
                assert len(values)==50 and all(isinstance(x,bool) for x in values)
                pairs=m['pairs'] if 'pairs' in m else dict(episodes=m['episodes'],start_steps=m['start_steps'])
                rows.append(dict(task=task,seed=seed,body=body,noise_logit_shift=shift,success_percent=2*sum(values),
                                 success_vector=values,pairs=pairs,model_config=cfg['model'],
                                 diagnostics=read(ap.with_name('state-audit.json'))['results'],
                                 audit_sha256=sha(ap),metrics_sha256=sha(mp)))
                bindings[str(terminal)]=sha(terminal)
    paired=[]
    definitions=[('spatial_minus_flat_ordinary',[('spatial',0.,1),('flat',0.,-1)]),
                 ('spatial_minus_flat_highnoise',[('spatial',.8,1),('flat',.8,-1)]),
                 ('highnoise_minus_ordinary_flat',[('flat',.8,1),('flat',0.,-1)]),
                 ('highnoise_minus_ordinary_spatial',[('spatial',.8,1),('spatial',0.,-1)]),
                 ('interaction',[('spatial',.8,1),('spatial',0.,-1),('flat',.8,-1),('flat',0.,1)])]
    for task in ['pusht','tworoom','cube']:
        for name,terms in definitions:
            differences=[]
            for seed in [1,2]:
                cells=[(next(x for x in rows if (x['task'],x['seed'],x['body'],x['noise_logit_shift'])==(task,seed,body,shift)),sign) for body,shift,sign in terms]
                assert all(x['pairs']==cells[0][0]['pairs'] for x,sign in cells)
                delta=sum(sign*100*np.asarray(x['success_vector'],float) for x,sign in cells)
                differences.append(delta);paired.append(dict(task=task,contrast=name,seed=seed,**interval(delta)))
            paired.append(dict(task=task,contrast=name,seed='two_fixed_models',**interval(np.mean(differences,axis=0))))
    write(a.output,dict(passed=True,step=600,rows=rows,paired=paired,terminal_bindings=bindings,
          all6_four_cell_observation_and_action_arrays_exact=True,collector_sha256=sha(__file__),
          scope='All24 cells complete;unchanged native task predicates/data/CEM. Full spatial body has388080 fewer parameters than flat but higher compute. '
                'Noise setting fixed+.8 across alltasks;within-body initial tensors/all savedRNG after2 are exact. '
                'Different-time FM training losses are not directly comparable. Conditional case intervals omit seed-population and simulator-repeat uncertainty;not long-budget success/novelty.'))
    print([(x['task'],x['body'],x['noise_logit_shift'],x['seed'],x['success_percent']) for x in rows])
    print([x for x in paired if x['seed']=='two_fixed_models'])


if __name__=='__main__':main()
