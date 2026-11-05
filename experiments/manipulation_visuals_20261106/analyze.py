"""Read-only approach diagnostics and encoding of saved initial/final/goal RGB."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def trajectories(root):
    rows=[]
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir() or not (folder/'proof.json').exists():continue
        proof=json.loads((folder/'proof.json').read_text())
        if proof['task']=='reach-v3':continue
        trace=Path(proof['source_trace'])
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==proof['source_trace_sha256']
        actions=np.load(trace)['actions']
        states=json.loads((folder/'diagnostics.json').read_text())
        obj=np.asarray([x['object_xyz'] for x in states]);tcp=np.asarray([x['tcp_xyz'] for x in states])
        target=np.asarray([x['target_xyz'] for x in states])
        distance=np.linalg.norm(tcp-obj,axis=1);closest=int(distance.argmin())
        rows.append(dict(task=proof['task'],arm=proof['arm'],case_seed=proof['case_seed'],parent_seed=proof['parent_seed'],
                         min_gripper_center_object_distance=float(distance[closest]),closest_step=closest+1,
                         final_gripper_center_target_distance=float(np.linalg.norm(tcp[-1]-target[-1])),
                         max_object_displacement=float(np.linalg.norm(obj-np.asarray(proof['object_initial']),axis=1).max()),
                         gripper_command_range=[float(actions[:,3].min()),float(actions[:,3].max())],
                         gripper_command_at_closest=float(actions[closest,3]),
                         definition='Gripper center is native mean of left/rightEndEffector sites; distances sampled after every executed action.'))
    return rows


def encode(root,checkpoint):
    import torch
    import stable_worldmodel as swm
    from PIL import Image
    from experiments.metaworld_proposal_20261104.evaluate import resize_tensor,model_sha
    torch.set_num_threads(4)
    print('Loading frozen CPU encoder',flush=True)
    model=swm.wm.utils.load_pretrained(str(checkpoint)).cpu().eval().requires_grad_(False)
    before=model_sha(model);rows=[]
    with torch.no_grad():
        for task in ['bin-picking-v3','pick-place-v3']:
            folder=Path(root)/f'tf-control-{task}'
            images=[np.asarray(Image.open(folder/name).convert('RGB')) for name in ['step000.png','step500.png','goal.png']]
            z=model.encode(dict(pixels=model.normalize(resize_tensor(images)[:,None])))['emb'][:,0].float()
            initial=float((z[0]-z[2]).square().mean());final=float((z[1]-z[2]).square().mean())
            rows.append(dict(task=task,initial_goal_mse=initial,failed_final_goal_mse=final,
                             final_over_initial=final/initial,model_state_sha256=before))
    assert before==model_sha(model)
    return dict(checkpoint=str(checkpoint),model_unchanged=True,device='cpu',encoding_only=True,optimizer_updates=0,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--media-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');a=p.parse_args()
    value=dict(passed=True,trajectory_diagnostics=trajectories(a.media_root),scope='Same saved cases, no new independent benchmark trials or learning.')
    if a.checkpoint:value['encoder_goal']=encode(a.media_root,a.checkpoint)
    Path(a.output).write_text(json.dumps(value,indent=2)+'\n')
