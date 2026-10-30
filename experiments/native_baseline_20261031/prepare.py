"""Prepare four native-objective LeWM gates; never launches a worker."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/native_lewm600_20261031'
PRE=B/'runs/native_lewm_preflight_20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SELF='experiments/native_baseline_20261031/prepare.py'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(args):
    root,pre=Path(args.root),Path(args.preflight_root)
    if (root/'candidate-plan.json').exists()or (pre/'plan.json').exists():raise FileExistsError('Already prepared')
    src=read(args.source_record);source=Path(src['source'])
    assert digest(__file__)==digest(source/SELF)==read(src['manifest'])['files'][SELF]
    contracts=read(args.contracts)
    assert contracts['passed']and len(contracts['rows'])==2
    for path,key in [('as_jepa_lewm/native_lewm.py','native_adapter_sha256'),
                     ('as_jepa_lewm/factory.py','factory_sha256'),('as_jepa_lewm/train.py','trainer_sha256')]:
        assert digest(source/path)==contracts[key],path
    environment=read(B/'runs/ideas_validate600_20261031/plan.json')['environment']
    jobs=[];prejobs=[];records=[]
    for ti,task in enumerate(['tworoom','cube']):
        data=B/'data/hardtasks-v1'/(task+'-metadata');card=read(data/'dataset_card.json')
        manifest=data/'manifests/validation-strict.json';pairs=read(manifest)
        assert pairs['task']==task and pairs['campaign_split']=='validation'and len(pairs['pairs'])==50
        windows=source/f'experiments/ideas_validate_20261031/audit-windows-{task}.json'
        assert windows.exists()and read(windows)['count']==64
        for seed in [1,2]:
            name=f'{task}-native-lewm-seed{seed}';out=root/name;po=pre/name
            cfg=dict(output=str(out),data=str(data),dataset_kind='benchmark_hdf5',seed=seed,
                batch_size=128,workers=4,cpu_threads=2,max_steps=124910,max_seconds=31536000,
                checkpoint_every=200,retain_steps=[200,400,600],record_initial_weights=True,
                model_class='native_lewm',model=dict(window=8,action_dim=card['model_action_dim']),
                precision='bf16-mixed',float32_matmul_precision='high')
            # Exact data/exposure knobs equal the existing TF/F fresh seed source.
            reference=B/'runs/mechanism_rate600_20261030'/f'{task}-fresh-rate-on-seed{seed}'/'config.json'
            old=read(reference)
            assert all(cfg[k]==old[k]for k in ['data','dataset_kind','seed','batch_size','max_steps'])
            assert cfg['model']['window']==old['model']['window']==8
            assert cfg['model']['action_dim']==old['model']['action_dim']
            config=out/'config.json';write(config,cfg)
            slot=ti*2+seed-1;common=dict(host='dev13',gpu=4+slot,source=src,task=task,
                training_seed=seed,family='native-lewm',arm='history7-matched')
            jobs.append(dict(**common,id=name,phase='train600',output=str(out),config=str(config),
                config_sha256=digest(config),command=[PY,'-B','-u','-m','as_jepa_tools.train_stage',
                    '--config',str(config),'--stop-step','600']))
            previous=name
            for phase in ['fullstate','transition','planning']:
                dest=root/(phase+'-'+name)
                if phase=='fullstate':
                    command=[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/preflight.py'),
                        '--run',str(out),'--output',str(dest),'--expected-step','600']
                elif phase=='transition':
                    command=[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/audit.py'),
                        '--run',str(out),'--output',str(dest),'--cases','64','--manifest',str(windows)]
                else:
                    command=[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),
                        '--checkpoint',str(out/'model-step0000600'),'--data',str(data),'--output',str(dest)]
                job=dict(**common,id=dest.name,phase=phase,output=str(dest),depends_on=[previous],command=command)
                if phase=='planning':job['environment']={'MUJOCO_EGL_DEVICE_ID':str(common['gpu'])}
                jobs.append(job);previous=dest.name
            pcfg=copy.deepcopy(cfg);pcfg.update(output=str(po),checkpoint_every=2,retain_steps=[2])
            # Keep actual production batch128: native SIGReg is batch coupled.
            pc=po/'config.json';write(pc,pcfg)
            prejobs.append(dict(**common,id=name,phase='train2',output=str(po),config=str(pc),config_sha256=digest(pc),
                command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(pc),'--stop-step','2']))
            dest=pre/('gradient-'+name)
            prejobs.append(dict(**common,id=dest.name,phase='preflight',output=str(dest),depends_on=[name],
                command=[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/preflight.py'),
                    '--run',str(po),'--output',str(dest),'--expected-step','2']))
            records.append(dict(id=name,task=task,seed=seed,output=str(out),config_sha256=digest(config),
                reference_config=str(reference),reference_config_sha256=digest(reference),
                dataset_card_sha256=digest(data/'dataset_card.json'),evaluation_manifest_sha256=digest(manifest),
                audit_manifest_sha256=digest(windows),host=common['host'],gpu=common['gpu']))
    write(pre/'plan.json',dict(root=str(pre),deadline=None,environment=environment,jobs=prejobs))
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=environment,jobs=jobs))
    write(root/'protocol.json',dict(created_at=time.time(),source=src,root=str(root),preflight=str(pre),
        runs=records,contracts=str(args.contracts),contracts_sha256=digest(args.contracts),stop_step=600,
        schedule_total=124910,schedule_warmup=1249,preflight_batch=128,prepared_only=True,
        label='Native LeWM objective, explicitly adapted history7/8frame training split/exposure and budget; not an unmodified official recipe or released-checkpoint reproduction.',
        preserved='Official LeWM modules; Cube action smoothed_dim10; native MSE+.09SIGReg/BN; no renderer/carry; native warmup-cosine initialzero then per-update; same train-only action normalization and sampler as our methods.',
        departures='Official history3/4frame becomes7/8; official window-level split/full-data scaler replaced by fixed episode split/train scaler; native Trainer/checkpoint/sampler reused; horizon124910; seeded model initialization; batch128 and existing mixed-precision policy.',
        review='Four production-batch actual2step checkpoints plus finite state/gradients and saved RNG/cursor must pass.600 candidate plan remains inactive until explicit root review/activation.600 is inside warmup; no automatic longer run.',
        audits='Existing generic fullstate/preflight and clean latent forecast/intervention audits. AS-JEPA renderer/rate health tools intentionally not called for a model without these components.',
        fairness='Same datasets, observation/action row exposure, strict50cases and CEM budget. Native models~18.04M versus TF/F~51.1M: not equal capacity or compute. Native baseline retains batch loss; proposed methods remain sample-level.'))
    print(json.dumps(dict(preflight=str(pre/'plan.json'),candidate=str(root/'candidate-plan.json'),preflight_jobs=len(prejobs),candidate_jobs=len(jobs))),flush=True)


def activate(args):
    root,pre=Path(args.root),Path(args.preflight_root)
    protocol=read(root/'protocol.json');assert protocol['preflight']==str(pre)
    for job in read(pre/'plan.json')['jobs']:
        assert read(pre/'exits'/(job['id']+'.json'))['returncode']==0,job['id']
        if job['phase']=='preflight':assert read(Path(job['output'])/'preflight.json')['passed']
    write(root/'plan.json',read(root/'candidate-plan.json'))
    write(root/'preflight-review.json',dict(activated_at=time.time(),passed=True,
        scope='All8 native preflight endpoints complete0; four actual production-batch2step fullstate/gradient audits passed. This creates the600plan only; it launches nothing.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','activate'])
    p.add_argument('--source-record');p.add_argument('--contracts');p.add_argument('--root',default=str(ROOT));p.add_argument('--preflight-root',default=str(PRE))
    a=p.parse_args()
    if a.phase=='prepare':
        if not a.source_record or not a.contracts:p.error('prepare requires --source-record and --contracts')
        prepare(a)
    else:activate(a)
