"""Prepare sixteen native fresh-case evaluations of completed2000 checkpoints."""
import copy
import json
from pathlib import Path

from experiments.carrier_progress_20261031.prepare_learning import read, write, digest

B = Path('/efs/prescottsun/SPT-JEPA')
ROOT = B/'runs/carrier_fresh2000_20261031'


def main():
    assert not ROOT.exists(), 'Preserve existing plans and results'
    parent = B/'runs/carrier_learn2000_20261031'
    old = read(parent/'plan.json')
    result = read(B/'reports/carrier-learn2000-20261031/results.json')
    assert result['completed_endpoint_count'] == 96
    source = read(parent/'source-record.json')
    jobs = []
    slots = [('dev13', n) for n in range(4,8)] + [('dev15', n) for n in range(2,8)]
    manifests = {t:B/'runs'/folder/'fresh-manifest.json' for t,folder in [
        ('tworoom','carrier_freshcases_20261031'),('cube','carrier_cube_freshcases_20261031')]}
    records = [(r, r['id'], parent/r['id']) for r in result['rows']]
    records += [(r, r['id'], Path(r['artifacts']['config']['path']).parent)
                for r in result['baselines'] if r['method']=='temporal-forecast']
    assert len(records)==16
    for slot,(row,name,run) in enumerate(records):
        host,gpu=slots[slot%len(slots)]
        task=row['task'];manifest=read(manifests[task]);cfg=read(run/'config.json')
        assert read(run/'stage-complete.json')['stop_step']==2000
        assert manifest['campaign_split']=='validation' and len(manifest['pairs'])==50
        checkpoint=run/'model-step0002000';dest=ROOT/name
        jobs.append(dict(id=name,task=task,training_seed=row['seed'],family=row.get('family','temporal-forecast'),
            arm=row.get('arm'),host=host,gpu=gpu,source=copy.deepcopy(source),phase='fresh2000',
            output=str(dest),depends_on=[],environment={'MUJOCO_EGL_DEVICE_ID':str(gpu)},
            manifest_sha256=digest(manifests[task]),checkpoint_config_sha256=digest(checkpoint/'config.json'),
            checkpoint_weights_sha256=digest(checkpoint/'weights.pt'),
            command=['/efs/prescottsun/project_jepa/envs/lewm/bin/python','-B','-u','-m',
                'as_jepa_tools.benchmark_evaluate','--manifest',str(manifests[task]),
                '--checkpoint',str(checkpoint),'--data',cfg['data'],'--output',str(dest)]))
    ROOT.mkdir(parents=True)
    write(ROOT/'plan.json',dict(root=str(ROOT),deadline=None,environment=old['environment'],jobs=jobs))
    write(ROOT/'source-record.json',source)
    write(ROOT/'protocol.json',dict(jobs=16,step=2000,model_updates=0,source_commit=source['source_commit'],
        input_result_sha256=digest(B/'reports/carrier-learn2000-20261031/results.json'),
        preparation_sha256=digest(__file__),slots=slots,
        scope='All12 completed Carrier models plus four temporal-forecast2000 controls; exact existing fresh50 '
              'validation pairs and policy seeds. Native planner and metrics unchanged. Reuse released LeWM '
              'fresh68/24 references with unmatched training/capacity; no published-score equivalence. '
              'No training and no validation-dependent case selection.'))
    print(json.dumps(dict(root=str(ROOT),jobs=len(jobs))))


if __name__=='__main__':
    main()
