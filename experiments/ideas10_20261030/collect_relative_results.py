"""Collect completed gates; retain per-seed paired outcomes without winner selection."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plan=read(root/'plan.json');protocol=read(root/'protocol.json')
    exits=[read(root/'exits'/(j['id']+'.json'))for j in plan['jobs']]
    assert len(exits)==120 and all(x['returncode']==0 for x in exits)
    rows=[];manifests={}
    for rec in protocol['runs']:
        name=rec['id'];health=read(root/('health-'+name)/'health.json')
        info=read(root/('info-'+name)/'information-audit.json')
        query=read(root/('query-'+name)/'query-audit.json')
        task=read(root/('planning-'+name)/'metrics.json')
        assert health['audit']['step']==600
        assert all(v for k,v in health['audit'].items()if isinstance(v,bool))
        successes=task['metrics']['episode_successes'];assert len(successes)==50
        assert abs(task['metrics']['success_rate']-100*sum(successes)/len(successes))<1e-8
        if rec['task']in manifests:assert task['manifest_sha256']==manifests[rec['task']]
        else:manifests[rec['task']]=task['manifest_sha256']
        rows.append(dict(**rec,step=600,success_rate=task['metrics']['success_rate'],episode_successes=successes,
            manifest_sha256=task['manifest_sha256'],health=health['audit'],feature_std=health['feature_std'],
            feature_signal_to_posterior_noise=health['feature_signal_to_posterior_noise'],parameters=health['parameters'],
            query=query['summary'],probes=info['probes']['scores'],probe_label_keys=info['probes']['label_keys']))
    lookup={(r['task'],r['seed'],r['method'],r['route']):r for r in rows}
    paired=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            comparisons=[]
            for method in ['direct_relative','annihilator_relative','random_relative']:
                comparisons.append((lookup[task,seed,method,'mixed'],lookup[task,seed,method,'forecast'],'route'))
            for route in ['forecast','mixed']:
                for reference in ['direct_relative','random_relative']:
                    comparisons.append((lookup[task,seed,'annihilator_relative',route],lookup[task,seed,reference,route],'operator'))
            for a,b,kind in comparisons:
                wins=sum(x and not y for x,y in zip(a['episode_successes'],b['episode_successes']))
                losses=sum(y and not x for x,y in zip(a['episode_successes'],b['episode_successes']))
                paired.append(dict(task=task,seed=seed,kind=kind,candidate=a['id'],reference=b['id'],gained=wins,lost=losses,
                    difference_pp=a['success_rate']-b['success_rate']))
    initial=read(root/'initial-tensor-identity.json')
    assert len(initial['runs'])==24 and all(r['all_original_tensors_equal']for r in initial['runs'])
    result=dict(root=str(root),protocol=protocol,exits=exits,initial_tensor_identity=initial,
        preflight=read(Path(protocol['preflight'])/'review.json'),rows=rows,paired=paired,manifest_sha256=manifests,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All24fresh600gates and all120endpoints. Paired fixed50-case outcomes, bothseeds retained. No long-run, broad-stability or novelty claim. Query objectives differ across operators; their errors are not a common task metric.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(id=r['id'],success_rate=r['success_rate'],state_std=r['feature_std'])for r in rows],indent=2))


if __name__=='__main__':main()
