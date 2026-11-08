"""Frozen factual forecasts from one observed frame versus eight real past frames."""
import os
from pathlib import Path
import traceback
from experiments.metaworld_transition_only_20261109 import diagnose
from experiments.metaworld_transition_only_20261109.common import ROOT,read,save,now,sha
from experiments.metaworld_horizon_match_20261109.model import build


Base=diagnose.FactualWindows


class ColdWindows(Base):
    def __getitem__(self,index):
        row=super().__getitem__(index)
        row['history_valid'].fill_(False);row['history_valid'][-1]=True
        return row


if __name__=='__main__':
    out=ROOT/'cold-diagnostics';out.mkdir(exist_ok=False)
    save(out/'OWNERSHIP.json',dict(pid=os.getpid(),utc=now(),source_sha256=sha(__file__),cpu_only=True,
        change='Onlymaskthe7pastobservations;keepcurrentRGB/factualfuture/knowninputs/parametersfixed. No missingfuture/goalpolicy labels.'))
    try:
        diagnose.FactualWindows=ColdWindows;diagnose.build=build
        jobs=[j for j in read(ROOT/'plan.json')['jobs'] if j['horizon']==15];summary=[]
        for job in jobs:
            value=diagnose.evaluate(job);warm=read(ROOT/'factual-diagnostics'/(job['name']+'.json'))
            assert value['state_before']==warm['state_before'] and value['selected_heldout_windows']==warm['selected_heldout_windows']
            save(out/(job['name']+'.json'),value)
            row=dict(job=job['name'],cold_over_dense_history={mode:{str(h):value['aggregate'][mode][str(h)]['mse']/max(warm['aggregate'][mode][str(h)]['mse'],1e-20)
                for h in [15,30,60]} for mode in ['factual_open','observe2','observe15']})
            summary.append(row);print(row,flush=True)
        save(out/'SUMMARY.json',dict(passed=True,utc=now(),models=8,unique_heldout_episodes=16,groups=summary,
            world_updates=0,head_updates=0,gpu_contexts=0,physics_steps=0,
            interpretation='ConditionsdifferONLYinavailableobservedhistory,notcurrentphysicalstate orfutureinputs. '
                'Cold0history canlackvelocity/context;performancegapdoesnotaloneproveimplementationbug orcontrolimprovement.'))
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
