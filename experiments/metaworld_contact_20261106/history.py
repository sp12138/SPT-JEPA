"""Scratch single-CNN factual-history controls;one action loss,no hidden clock."""
import argparse
import copy
import datetime
import json
from pathlib import Path

from experiments.metaworld_contact_20261106.diagnose import save, digest


def prepare(root,source):
    out=root/'history-v1';out.mkdir()
    old=json.loads((root/'bridge-plan-v3.json').read_text())
    recipes=[];stages=[2,600,3000,10000,20000,40000]
    for host,seed in [('dev13',1),('dev15',2)]:
        owners=json.loads((root/'repeat-v1'/f'launch-{host}.json').read_text())['jobs']
        for gpu,base in [(6,'local-mse'),(7,'local-flow')]:
            parent=next(r for r in old['recipes'] if r['host']==host and r['arm']==base)
            config=copy.deepcopy(json.loads((Path(parent['run'])/'config.json').read_text()))
            name=f'{base}-history-seed{seed}';run=out/'bridge'/name;run.mkdir(parents=True)
            config.update(output=str(run),retain_steps=stages,record_initial_weights=True,seed=seed)
            config.pop('initial_encoder_checkpoint',None);config.pop('feature_cache',None)
            config['model'].update(history_context=True,temporal_aux=False,condition_horizon='none')
            save(run/'config.json',config)
            save(run/'ORIGIN.json',dict(random_encoder_and_controller=True,pretrained_modules=False,
                 previous_image_and_executed_action=True,one_sample_level_action_loss=True,
                 unknown_horizon_not_an_input=True,new_simulator_training_data=False,
                 scope='CurrentRGB alone can alias velocity/contact phase.Test factual history with one sharedCNN;no task-specific architecture or taskID. '
                       'Two fresh seeds/objective;not a clean single-factor comparison to the earlier clock+temporalaux model. '
                       'The successful40kCNN architecture is retained;new2step full-state/backward/independence/physical gates required before long steps.'))
            recipes.append(dict(name=name,arm=f'{base}-history',seed=seed,host=host,gpu=gpu,run=str(run),
                 parent=parent['parent'],stages=stages,technical_step=2,config_sha256=digest(run/'config.json'),
                 wait_owner=next(v for v in owners if v['gpu']==gpu)))
    plan=dict(old,source=str(source),recipes=recipes,stages=stages,
         created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         case_seeds_by_step={'40000':list(range(42,52))+list(range(1042,1052))},
         scope='Fresh scratchCNN-history controls after reproducible manipulation gains;no more pretrainedViT branches. '
               'Same loggedcorpus/split,universal settings,one actionloss and no clockinput.No fresh2042cohort tuning. '
               'Factual previousimage/action may disambiguate dynamics;benefit is a hypothesis until closed-loop evaluation. '
               'Finite40k each,original24h deadline;no automatic next round.')
    save(out/'bridge-plan.json',plan);save(out/'bridge-plan-v3.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.source)
