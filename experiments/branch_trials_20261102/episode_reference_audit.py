"""Change only the reference distribution on cached factual query outputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from common import dataset,read,write,sha
from as_jepa_tools.goal_query_data import QueryReferenceWindows
from as_jepa_tools.evaluate import load_model
from experiments.trajectory_search_20261101.common import state_hash
from experiments.ideas_validate_20261031.audit import move_batch
from vforge.ops import OpsConfig,initialize_ops


def main(root,ident,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    p=read(root/'protocol.json');row=next(r for r in p['runs'] if r['id']==ident);assert row['family']=='query' and row['arm']=='function'
    run=Path(row['run']);cfg=read(run/'config.json');old=root/'audits'/f'{ident}-step2000';audit=read(old/'audit.json');assert audit['passed']
    cache=np.load(old/'diagnostics.npz');cohort=read(root/'cohorts'/f'{row["task"]}.json');indices=[x['index'] for x in cohort]
    base=dataset(dict(cfg,goal_query_reference=False),'validation');ds=QueryReferenceWindows(base,cfg['seed'],mode='episode')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=load_model(run/'model-step0002000',device='cuda').eval();model.requires_grad_(False);before=state_hash(model)
    negative=[];references=[];cursor=0
    with torch.no_grad():
        for raw in DataLoader(Subset(ds,indices),batch_size=4,num_workers=0):
            batch=move_batch(raw,'cuda');assert torch.equal(batch['episode'],batch['reference_episode'])
            n=len(batch['episode']);q=torch.from_numpy(cache['queries'][cursor:cursor+n]).cuda()
            observed=torch.from_numpy(cache['observed'][cursor:cursor+n,3:]).cuda()
            positive=model.query_logits(q,observed)
            torch.testing.assert_close(positive,torch.from_numpy(cache['positive'][cursor:cursor+n]).cuda(),rtol=3e-5,atol=3e-5)
            emb=model.encode(dict(pixels=model.normalize(batch['reference_pixels'])))['emb']
            negative.append(model.query_logits(q,emb).cpu().numpy())
            references.extend(zip(batch['episode'].cpu().tolist(),batch['start'].cpu().tolist(),batch['reference_start'].cpu().tolist()))
            cursor+=n
    after=state_hash(model);assert after==before and cursor==64
    actual=cache['positive'];global_negative=cache['reference'];episode_negative=np.concatenate(negative)
    def metrics(neg):
        return dict(negative_logit=float(neg.mean()),positive_minus_negative=float((actual-neg).mean()),
                    source_accuracy=float(((actual>0).mean()+(neg<0).mean())/2),
                    factual_beats_reference=float((actual>neg).mean()),
                    balanced_bce=float(np.mean(np.logaddexp(0,-actual)+np.logaddexp(0,neg))))
    report=dict(passed=True,id=ident,global_reference=metrics(global_negative),episode_reference=metrics(episode_negative),
                weights_sha256=sha(run/'model-step0002000/weights.pt'),factual_cache_sha256=sha(old/'diagnostics.npz'),
                state_before=before,state_after=after,references=references,reference_frames_encoded=64*5,
                new_factual_encoder_calls=0,model_updates=0,simulation_calls=0,
                scope='Same cached factual queries;reference-source difficulty only,not a task improvement or proof of action mediation.')
    np.savez_compressed(out/'reference_scores.npz',actual=actual,global_reference=global_negative,episode_reference=episode_negative)
    write(out/'result.json',report);print(dict(passed=True,id=ident,global_reference=report['global_reference'],episode_reference=report['episode_reference']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.root,a.id,a.output)
