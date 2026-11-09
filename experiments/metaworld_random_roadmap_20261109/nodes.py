"""Unordered state support from OUR random interventions, not expert paths.

Only TRAIN clips and histories entirely after randomization are eligible.
The planner never receives trajectory order, task/state/reward/success labels,
or logged future actions. All pretrained models are our own scratch ancestry.
This adds TRAIN-state memory and a data budget; not a matched SOTA comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command
from experiments.metaworld_covariance_world_20261109.train import setup as covariance_setup
from experiments.metaworld_horizon_match_20261109.model import build as inverse_build

ROOT=BASE/'runs/metaworld_random_roadmap_20261109'
DATA=Path('/tmp/prescottsun-random-interventions-native128-20261109-v1')


def load_models(seed):
    cfg=read(BASE/f'runs/metaworld_covariance_world_20261109/training/diagonal-seed{seed}/config.json')
    device,world,_,_=covariance_setup(cfg);weight=Path(cfg['output'])/'model-step0002000/weights.pt'
    assert sha(weight)==read(weight.parent/'SEALED.json')['weights_sha256']
    world.load_state_dict(torch.load(weight,map_location=device,weights_only=True),strict=True);world.eval().requires_grad_(False)
    origin=read(BASE/f'runs/metaworld_horizon_match_20261109/training/h60-seed{seed}/config.json')
    parent=Path(origin['output'])/'model-step0012000/weights.pt';assert sha(parent)==read(parent.parent/'SEALED.json')['weights_sha256']
    inverse=inverse_build(origin).to(device).eval().requires_grad_(False)
    inverse.load_state_dict(torch.load(parent,map_location=device,weights_only=True),strict=True)
    return world,inverse,dict(world_weights_sha256=sha(weight),inverse_weights_sha256=sha(parent),
        world_ancestry='Ownscratchfactual12000+meanworld2000+covariance2000. AllFROZEN.',
        inverse_ancestry='Ownfactual12000;inverseONLYSAMEcurrentnativea0fromobservedcompletedtransition. No unknownoptimal/future-actiontargets.')


def prepare(source):
    ROOT.mkdir(exist_ok=False);assert read(BASE/'runs/metaworld_interventions_20261109/CLOSED.json')['passed']
    records=[];shards=[];items=[]
    for shard in range(12):
        m=read(DATA/str(shard)/'manifest.json');shards.append(np.load(DATA/str(shard)/'pixels.npy',mmap_mode='r'))
        for r in m['records']:
            if r['split']=='train':
                record=dict(r,shard=shard);record_index=len(records);records.append(record)
                # Last7 controls and8 images are ALL from OUR random phase.
                for t in range(14,r['length']-1,2):items.append(dict(record=record_index,time=t))
    assert len(records)==400 and len(items)==10800
    rng=np.random.default_rng(1050009);items=[items[int(i)] for i in rng.permutation(len(items))]
    nodes=ROOT/'nodes';nodes.mkdir();rgb=np.lib.format.open_memmap(nodes/'pixels.npy',mode='w+',dtype=np.uint8,shape=(len(items),128,128,3));hashes=[]
    for i,row in enumerate(items):
        r=records[row['record']];frame=shards[r['shard']][r['offset']+row['time']];rgb[i]=frame;hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    rgb.flush();del rgb;np.save(nodes/'image_hashes.npy',np.asarray(hashes,dtype='S64'))
    save(ROOT/'PROVENANCE_ONLY.json',dict(records=records,shuffled_items=items,
        scope='Construction/auditONLY. PlannerreadsNONEofthis;NOtime/taskID/futureinput references in graph search.'))
    jobs=[];bounds=np.linspace(0,len(items),13,dtype=int)
    for seed in [1,2]:
        for part in range(12):jobs.append(dict(name=f'seed{seed}-part{part}',seed=seed,part=part,start=int(bounds[part]),stop=int(bounds[part+1]),output=str(ROOT/'features'/f'seed{seed}'/f'part{part}')))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        nodes_helper_sha256=sha(__file__),jobs=jobs,deadline=time.time()+7200,nodes=len(items),records=400,
        independent_TRAIN_case_contexts=50,original_trained_parents=2,
        raw_sha256=sha(nodes/'pixels.npy'),image_hashes_sha256=sha(nodes/'image_hashes.npy'),provenance_sha256=sha(ROOT/'PROVENANCE_ONLY.json'),
        training_updates=0,scope='EXTRAunorderedTRAIN-state-memoryfromOURrandomtrajectoriesONLY. '
        'No expertpath/future-actionsequence/optimal-actionreferences. Past7knownrandominputsareonlyforworldstatefiltering. '
        'Current-inputinversegeneratesOWNsteeringcontrols;notloggedfuturecontrols. GoalONLYplanning. '
        'Bothmodelswholeancestryownscratch/GNsamplelevel/RGBexecuted-input/rewardfree. '
        'Additionaldata/memory/camerabudgetmustbedisclosed;NOTmatchedpublicSOTA/equalFLOPs.',automatic_training=False))


def data_arrays():
    return [dict(pixels=np.load(DATA/str(s)/'pixels.npy',mmap_mode='r'),inputs=np.load(DATA/str(s)/'known_input.npy',mmap_mode='r')) for s in range(12)]


@torch.no_grad()
def extract(name):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['name']==name);world,inverse,ancestry=load_models(j['seed'])
    before=(tensor_sha(world),tensor_sha(inverse));provenance=read(ROOT/'PROVENANCE_ONLY.json');arrays=data_arrays()
    assert sha(ROOT/'PROVENANCE_ONLY.json')==p['provenance_sha256'];out=Path(j['output']);out.mkdir(parents=True,exist_ok=False);n=j['stop']-j['start']
    hidden=np.lib.format.open_memmap(out/'hidden.npy',mode='w+',dtype=np.float32,shape=(n,128,16,16))
    features=np.lib.format.open_memmap(out/'inverse_features.npy',mode='w+',dtype=np.float32,shape=(n,96,16,16))
    descriptor=np.lib.format.open_memmap(out/'descriptor.npy',mode='w+',dtype=np.float32,shape=(n,1536))
    for start in range(j['start'],j['stop'],64):
        rows=[];stop=min(start+64,j['stop'])
        for row in provenance['shuffled_items'][start:stop]:
            r=provenance['records'][row['record']];t=row['time'];a=arrays[r['shard']];offset=r['offset'];assert t>=14 and r['split']=='train'
            history=torch.from_numpy(np.asarray(a['pixels'][offset+t-7:offset+t+1]).copy()).permute(0,3,1,2)
            controls=torch.from_numpy(np.asarray(a['inputs'][offset+t-7:offset+t]).copy()).float()
            rows.append(dict(history_pixels=history,history_actions=controls,history_valid=torch.ones(8,dtype=torch.bool)))
        batch={k:v.cuda() for k,v in default_collate(rows).items()};direction=torch.zeros(len(rows),device='cuda',dtype=torch.long)
        state,current=world.parent.prepare(batch,direction);f=inverse.vision(current)
        d=torch.nn.functional.normalize(torch.nn.functional.adaptive_avg_pool2d(f,4).flatten(1),dim=1)
        index=slice(start-j['start'],stop-j['start']);hidden[index]=state.cpu().numpy();features[index]=f.cpu().numpy();descriptor[index]=d.cpu().numpy()
    hidden.flush();features.flush();descriptor.flush();del hidden,features,descriptor
    assert before==(tensor_sha(world),tensor_sha(inverse))
    save(out/'DONE.json',dict(passed=True,utc=now(),job=j,files={f:sha(out/f) for f in ['hidden.npy','inverse_features.npy','descriptor.npy']},
        models_unchanged=True,state_before=before,state_after=before,ancestry=ancestry,optimizer_updates=0,physics_steps=0,
        only_TRAIN_and_fully_random_past_context=True,no_logged_future_controls_read=True))


def work(host,gpu):
    p=read(ROOT/'plan.json');source=Path(p['source']);assert sha(__file__)==p['nodes_helper_sha256']
    for j in p['jobs']:
        claim=ROOT/'claims-nodes'/(j['name']+'.json');claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:json.dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        try:command([PYTHON,str(Path(__file__).resolve()),'extract','--name',j['name']],Path(j['output']).with_suffix('.log'),environment(source,gpu),source,p['deadline'],cap=1800)
        except BaseException:
            save(ROOT/'errors-nodes'/(j['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');source=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'nodes-worker-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=source,
                env=environment(source,gpu)|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=child.pid))
    save(ROOT/f'nodes-launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json')
    while time.time()<p['deadline']:
        done=[read(Path(j['output'])/'DONE.json') for j in p['jobs'] if (Path(j['output'])/'DONE.json').exists()]
        errors=list(map(str,ROOT.glob('errors-nodes/*.json')));save(ROOT/'results/nodes-summary.json',dict(utc=now(),completed=len(done),planned=24,errors=errors))
        if len(done)==24:
            assert not errors
            proofs={}
            for seed in [1,2]:
                target=ROOT/'bank'/f'seed{seed}';target.mkdir(parents=True,exist_ok=False)
                for name,shape in [('hidden.npy',(p['nodes'],128,16,16)),('inverse_features.npy',(p['nodes'],96,16,16)),('descriptor.npy',(p['nodes'],1536))]:
                    output=np.lib.format.open_memmap(target/name,mode='w+',dtype=np.float32,shape=shape)
                    for d in done:
                        j=d['job']
                        if j['seed']!=seed:continue
                        file=Path(j['output'])/name;assert sha(file)==d['files'][name];output[j['start']:j['stop']]=np.load(file,mmap_mode='r')
                    output.flush();del output
                proofs[str(seed)]={f.name:sha(f) for f in target.glob('*.npy')}
            save(ROOT/'NODE_BANK_READY.json',dict(passed=True,utc=now(),nodes=p['nodes'],parent_seeds=[1,2],files=proofs,
                all24_part_array_SHAs_verified=True,all_models_unchanged=True,training_updates=0,physics_steps=0,
                no_expert_path_or_future_actions_in_node_features=True));return
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','extract','work','launch','collect']);p.add_argument('--source',type=Path);p.add_argument('--name');p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='extract':extract(a.name)
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='launch':launch(a.host)
    else:collect()
