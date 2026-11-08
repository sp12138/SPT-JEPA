"""Frozen matched forecasting/filter tests, plus existing real-controller logs."""
import collections
from pathlib import Path
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109 import diagnose
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,now,sha
from experiments.metaworld_horizon_match_20261109.model import build

ROOT=BASE/'runs/metaworld_horizon_match_20261109'


if __name__=='__main__':
    out=ROOT/'factual-12000';out.mkdir(exist_ok=False)
    save(out/'AUDIT.json',dict(utc=now(),source_sha256=sha(__file__),world_updates=0,gpu_contexts=0,physics_steps=0))
    try:
        diagnose.build=build;models=[]
        for job in read(ROOT/'plan.json')['jobs']:
            d=diagnose.evaluate(job,step=12000);save(out/(job['name']+'.json'),d)
            models.append(dict(name=job['name'],aggregate=d['aggregate'],restarts=d['restarts'],
                state_before=d['state_before'],state_after=d['state_after']))
        controllers={}
        for folder in [ROOT/'evaluation',ROOT/'extended-controller/evaluation']:
            for path in folder.glob('*/progress.json'):
                rows=read(path)['rows'];group=collections.defaultdict(list)
                for row in rows:
                    for p in row['planning']:group[row['task']].append(p)
                controllers[str(path.parent)]=dict(episodes=len(rows),by_task={task:dict(decisions=len(plans),
                    mean_prefix_rgb_mse=float(np.mean([p['factual_prefix_rgb_mse'] for p in plans])),
                    median_prefix_rgb_mse=float(np.median([p['factual_prefix_rgb_mse'] for p in plans])),
                    mean_prefix_goal_error_bias=float(np.mean([p['predicted_prefix_goal_mse']-p['actual_prefix_goal_mse'] for p in plans])))
                    for task,plans in group.items()})
        save(out/'SUMMARY.json',dict(passed=True,utc=now(),models=models,controller_logs=controllers,
            world_updates=0,gpu_contexts=0,physics_steps=0,
            scope='16sameheldoutclips/model,warm8densepast. observe2/15 use factual observations andknowncontrols. '
                'ControllerprefixerrorsareonDIFFERENTpolicy-inducedpaths;notpairedsame-inputcausality. No fitting orgoalpolicy labels.'))
        for d in models:print(d['name'],'open60',d['aggregate']['factual_open']['60'],'obs2/15mse60',[d['aggregate'][k]['60']['mse'] for k in ['observe2','observe15']],flush=True)
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
