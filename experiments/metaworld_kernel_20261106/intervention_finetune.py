"""Separate extra-data study with paired eligible forward-only parents."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for host,seed in [('dev13',1),('dev15',2)]:
        probes=json.loads((a.collection/f'diagnostic-pack-launch-{host}.json').read_text())['records']
        warm=json.loads((a.warm/f'launch-{host}.json').read_text())['workers']
        for gpu,mode in[(0,'causal'),(4,'causal-and-goal'),(2,'original')]:
            parent=a.parent/'training'/f'rollout-seed{seed}';assert json.loads((parent/'audit-2500.json').read_text())['passed']
            run=a.root/'training'/f'{mode}-seed{seed}';run.mkdir(parents=True);cfg=json.loads((parent/'config.json').read_text())
            cfg.update(output=str(run),max_steps=5000,retain_steps=[2,500,2500,5000],checkpoint_every=500,
                intervention_rgb_cache='/tmp/prescottsun-intervention-rgb128-20261107-v1',intervention_goal_queries=mode=='causal-and-goal')
            cfg['model'].update(warm_start_checkpoint=str(parent/'model-step0002500/weights.pt'),warm_start_audit=str(parent/'audit-2500.json'))
            if mode!='original':cfg['dataset_kind']='query_rgb_paths_aug'
            save(run/'config.json',cfg)
            save(run/'ORIGIN.json',dict(parent=str(parent),parent_step=2500,parent_checkpoint_sha256=digest(parent/'model-step0002500/weights.pt'),
                lineage='Ownscratch20kforwardRGB+own2500RGBqueryupdates;originalsearchran10000anddiscardedlaterstates,allsearchcostdisclosed.',
                new_optimizer='FreshAdamWand5000-stepcosine,identicalforbothaugmentedarmsandoriginal-data continuation;notanexactoptimizertrajectorycontinuation.',
                shared_trained_parents=2,not_six_independent_pretrainings=True,
                data='Originalonly'if mode=='original'else'Original+separateown-interventionTRAINparents;validation/benchmarkstatesexcluded',
                observation_targets='Same5actualfutureRGBframesperbranchineitheraugmentedarm;noactionlabel,pseudolabel,orcontrol-coordinatehead.',
                mixing='Naturalunionofeligibleoriginalandnewbranchstarts,notatunedsamplerweight.',
                limitations='Extra-data study,notanoriginal-corpusmethodwin/publicSOTA.Anyalgorithmcomparisonmustprovideidenticalextracorpustobaselines.'))
            if mode=='original':owner=next(x for x in warm if x['gpu']==gpu);marker='warm_campaign.py'
            else:owner=next(x for x in probes if x['kind']=='frozen10000'and x['gpu']==gpu);marker='frozen-branch10000-v1.py'
            recipes.append(dict(arm=mode,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=owner['pid'],resource_marker=marker,config_sha256=digest(run/'config.json')))
    old=json.loads((a.parent/'training-plan.json').read_text())
    save(a.root/'training-plan.json',dict(source=str(a.source),data=old['data'],manifest=old['manifest'],recipes=recipes,stages=[2,500,2500,5000],
        objectives=['projection'],deadline_unix=time.time()+24*3600,separate_extra_data_track=True,
        scope='Causalidentificationablation;2selectedtrainedparentscrossedwith3dataconditioningschemes.Noinverse/externalancestry.'
              'Originalcontinuationcontrolsoptimization;causal-vs-causal-and-goalshareexactstarts/RGBtargets/initialweights.'
              'NoequalFLOPs/publicSOTAclaim;extra corpusmustbesharedwithbaselinesbeforeclaimingalgorithmsuperiority.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','source','parent','collection','warm']:p.add_argument('--'+key,type=Path,required=True)
    prepare(p.parse_args())
