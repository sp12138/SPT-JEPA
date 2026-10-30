"""Freeze one24-slot600-update causal test: operator x encoder-gradient route."""
import hashlib
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def main():
    b=Path('/efs/prescottsun/SPT-JEPA');code=b/'code'
    root=b/'runs/ideas10_relative600_20261031';pre=b/'runs/ideas10_relative_preflight_20261031'
    assert not root.exists() and not pre.exists()
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=code,check=True,stdout=subprocess.DEVNULL)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    root.mkdir();pre.mkdir();source=root/'source-v1';source.mkdir()
    paths=['as_jepa_lewm','as_jepa_tools','tests/as_jepa','experiments/ideas10_20261030']
    archive=subprocess.check_output(['git','archive',commit,*paths],cwd=code)
    with tarfile.open(fileobj=io.BytesIO(archive))as tar:tar.extractall(source,filter='data')
    write(root/'source-v1.json',dict(source=str(source),source_commit=commit,files={str(p.relative_to(source)):digest(p)for p in source.rglob('*')if p.is_file()}))
    src=json.loads((b/'runs/ideas10_response600_20261030/source-record.json').read_text());old=src['source']
    src=dict(src,source=str(source),source_commit=commit,manifest=str(root/'source-v1.json'),pythonpath=src['pythonpath'].replace(old,str(source)))
    env=json.loads((b/'runs/ideas10_response600_20261030/plan.json').read_text())['environment']
    py='/efs/prescottsun/project_jepa/envs/lewm/bin/python';control=b/'runs/mechanism_rate600_20261030'
    jobs=[];prejobs=[];records=[]
    for ti,task in enumerate(['tworoom','cube']):
        for seed in [1,2]:
            ref=control/f'{task}-fresh-rate-on-seed{seed}'
            for mi,method in enumerate(['direct_relative','annihilator_relative','random_relative']):
                for ri,(route,fraction) in enumerate([('forecast',0.),('mixed',.5)]):
                    name=f'{task}-{method}-{route}-seed{seed}';out=root/name;out.mkdir()
                    cfg=json.loads((ref/'config.json').read_text())
                    cfg.update(output=str(out),checkpoint_every=200,retain_steps=[200,400,600],record_initial_weights=True)
                    cfg['model'].update(response_query_mode=method,response_query_encoded_fraction=fraction)
                    config=out/'config.json';write(config,cfg)
                    slot=ti*12+(seed-1)*6+mi*2+ri;host=['dev2','dev13','dev15'][slot//8];gpu=slot%8
                    common=dict(host=host,gpu=gpu,source=src,task=task,method=method,route=route,training_seed=seed)
                    jobs.append(dict(**common,id=name,output=str(out),config=str(config),config_sha256=digest(config),phase='train600',command=[py,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(config),'--stop-step','600']))
                    previous=name
                    for phase,module in [('health','as_jepa_tools.benchmark_health'),('info','as_jepa_tools.benchmark_information_audit'),('query',None)]:
                        dest=root/(phase+'-'+name);ident=phase+'-'+name
                        command=[py,'-B','-u','-m',module]if module else[py,'-B','-u',str(source/'experiments/ideas10_20261030/query_audit.py')]
                        jobs.append(dict(**common,id=ident,output=str(dest),depends_on=[previous],phase=phase,command=command+['--run',str(out),'--output',str(dest)]))
                        previous=ident
                    dest=root/('planning-'+name)
                    jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase='planning',environment={'MUJOCO_EGL_DEVICE_ID':str(gpu)},
                        command=[py,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(Path(cfg['data'])/'manifests/validation-strict.json'),'--checkpoint',str(out/'model-step0000600'),'--data',cfg['data'],'--output',str(dest)]))
                    records.append(dict(id=name,original_control=str(ref),task=task,seed=seed,method=method,route=route,encoded_fraction=fraction,config_sha256=digest(config)))
                    if seed==1:
                        po=pre/name;po.mkdir();pcfg=json.loads(json.dumps(cfg));pcfg.update(output=str(po),batch_size=2,workers=0,checkpoint_every=2,retain_steps=[2]);pc=po/'config.json';write(pc,pcfg)
                        c=dict(host='dev2'if task=='tworoom'else'dev15',gpu=mi*2+ri,source=src)
                        prejobs.append(dict(**c,id=name,output=str(po),config=str(pc),config_sha256=digest(pc),command=[py,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(pc),'--stop-step','2']))
                        go=pre/('gradient-'+name)
                        prejobs.append(dict(**c,id=go.name,output=str(go),depends_on=[name],command=[py,'-B','-u','-m','as_jepa_tools.gradient_preflight','--run',str(po),'--output',str(go)]))
    write(root/'plan.json',dict(root=str(root),deadline=None,environment=env,jobs=jobs))
    write(pre/'plan.json',dict(root=str(pre),deadline=None,environment=env,jobs=prejobs))
    write(root/'source-record.json',src)
    write(root/'protocol.json',dict(created_at=time.time(),source_commit=commit,runs=records,stop_step=600,preflight=str(pre),
        prior_review='First40endpoints exit0, no task winner. Query nearly matches zero prediction; state/action sensitivity negligible. Relative weighting does not fix exact carry-gradient cancellation.',
        mechanism='One3x2 factorial: identity/action-null/random-null operator x clean-forecast-only/half-observed-half-forecast query objective. Same550initialtensors, identical12480head, same data/seed/nativeTF objective/planner.',
        loss='Shared initial grounding plus weighted raw projected error / (detached projected target energy+(1/127.5)^2). Same target/operator/normalizer for observed and forecast branches. Observed future encodings never enter causal rollout.',
        compute='All24arms compute7cleanpredictor calls. Mixed adds only a sharedhead read on already-computed observedstates. All24H200slots used for one600stepgate, not a long hyperparameter grid.',
        limitations=['Relative weighting changes conditional Bayes targets when energy depends on outcome; unpredictable tiny residuals may dominate.',
            'OneRGBLSB squared is a fixed conservative floor, not a calibrated post-pooling noise variance; record floor-dominated samples.',
            'Posterior grounding removes a structural carry-path Jacobian cancellation only when per-frame encoder Jacobians differ; it does not prove escape from all constant representations.',
            'Cube action-nullspace usually hasrank1. Fixed random projector matchesrank and removes1/time but is not equivalent in every signal statistic.',
            '600steps are still in1249warmup. This is not a maturity, academicnovelty or baseline-superiority claim.'],
        next_review='Stop all24at600, inspect health, query use, probes and all50-case strict tasks; no automatic continuation. Do not count normalization or route variants as separate ideas. All older holds immutable.'))
    print(json.dumps(dict(root=str(root),preflight=str(pre),source_commit=commit,jobs=len(jobs),preflight_jobs=len(prejobs))))


if __name__=='__main__':main()
