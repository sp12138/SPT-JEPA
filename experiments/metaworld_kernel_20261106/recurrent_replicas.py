"""Two genuinely independent eligible parents for original-data H5 replication."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);old=json.loads((a.parent/'training-plan.json').read_text());recipes=[]
    for host,seed in[('dev13',3),('dev15',4)]:
        parent=a.parent/'training'/f'rollout-seed{1 if host=="dev13"else 2}';cfg=json.loads((parent/'config.json').read_text())
        donor=a.parent.parent/'metaworld_native_transport_20261106/training'/f'free-seed{seed}'
        assert json.loads((donor/'ORIGIN.json').read_text())['all_weights_random']and json.loads((donor/'audit-20000.json').read_text())['passed']
        run=a.root/'training'/f'rollout-seed{seed}';run.mkdir(parents=True);cfg.update(seed=seed,output=str(run),retain_steps=[2,500,2500])
        cfg['model']['init_checkpoint']=str(donor/'model-step0020000/weights.pt');save(run/'config.json',cfg)
        owner=next(x for x in json.loads((a.warm/f'retirement-{host}.json').read_text())['rows']if x['gpu']==6)
        save(run/'ORIGIN.json',dict(parent=str(donor),parent_step=20000,scope='OwnindependentrandomseedforwardRGBparent;freshGRU/context/head.Originalcorpusonly,noinverse/externalweights.',
            same_training_as_original_H5=True,SGD_cap=2500,LR_horizon=10000,not_10000_training=True,selection='H5rolloutselectedfromearlierseeds1/2;seeds3/4notindependentnewcasecohort.'))
        recipes.append(dict(arm='rollout',seed=seed,host=host,gpu=6,run=str(run),resource_pid=owner['retained_eval_pid'],resource_marker='native_evaluate.py',config_sha256=digest(run/'config.json')))
    save(a.root/'training-plan.json',dict(source=str(a.source),data=old['data'],manifest=old['manifest'],recipes=recipes,stages=[2,500,2500],objectives=['projection'],
        deadline_unix=time.time()+24*3600,scope='Twoindependenttrainingparentsseeds3/4replicateoriginal-dataH5rollout;matchedLRhorizon10000butfiniteSGDcap2500.'
              'Originalcorpusonly.20physicalcases/taskatfinal,notnewuntouchedevaluationcases.Noinverse/externalancestry.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['root','source','parent','warm']:p.add_argument('--'+key,type=Path,required=True)
    prepare(p.parse_args())
