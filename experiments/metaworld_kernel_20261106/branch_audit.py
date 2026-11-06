"""Frozen forward/goal identifiability audit on own-command physical branches.

Recorded controls are dynamics INPUTS only.No learner,action decoder,or action
classification objective is optimized here.Privileged diagnostics are not read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.metaworld_eval_20261103.evaluate import model_sha


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def seal(root,output):
    contexts={}
    for tile in range(4):
        manifest=root/f'tile{tile}/manifest.json';value=json.loads(manifest.read_text());assert value['passed']and value['policy_unchanged']
        for r in value['records']:
            if r['split']!='validation':continue
            contexts.setdefault((r['task_id'],r['context']),[]).append(r)
    selected=[]
    for task in range(50):
        candidates=sorted((key,rows)for key,rows in contexts.items()if key[0]==task)
        assert len(candidates)==4
        _,rows=candidates[task%4];assert len(rows)==29 and len({r['parent']for r in rows})==1
        parent=Path(rows[0]['parent']);assert parent.exists()
        for r in rows:assert sha(Path(r['branch']))==r['branch_sha256']
        selected.append(dict(task=task,context=rows[0]['context'],parent=str(parent),parent_sha256=sha(parent),rows=rows))
    save(output,dict(passed=True,contexts=selected,source_root=str(root),selection='Oneof4uniformtimespertask,cyclictaskmod4;validationparentsonly.No contact/success selection.',
        new_updates=0,not_training_data_for_original_models=True))


def audit(a):
    from as_jepa_lewm.factory import build_model
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    sealed=json.loads(a.manifest.read_text());assert sealed['passed']and len(sealed['contexts'])==50
    cfg=json.loads((a.run/'config.json').read_text());assert cfg['model_class']=='query_recurrent'
    model=build_model(cfg).cuda().eval().requires_grad_(False);weights=a.run/f'model-step{a.step:07d}/weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=model_sha(model);reports=[]
    def image(array):return model.encoder(torch.from_numpy(np.asarray(array).copy()).permute(0,3,1,2).cuda())
    with torch.inference_mode():
        for context in sealed['contexts']:
            parent=Path(context['parent']);assert sha(parent)==context['parent_sha256'];data=np.load(parent);rows=context['rows'];t=rows[0]['prefix_stop']-1
            current=image(data['pixels'][t:t+1]);previous=image(data['pixels'][t-5:t-4]);past=torch.as_tensor(data['controls'][t-1:t],device='cuda')
            branches=[np.load(r['branch'])for r in rows]
            for r in rows:assert sha(Path(r['branch']))==r['branch_sha256']
            commands=torch.from_numpy(np.stack([np.pad(b['controls'],((0,10-len(b['controls'])),(0,0)))for b in branches])).cuda()
            observed=torch.stack([image(np.concatenate([b['pixels'],np.repeat(b['pixels'][-1:],10-len(b['pixels']),0)]))for b in branches])
            size=len(rows);role=current.new_zeros(size);hidden,latent=model.prepare(current.expand(size,-1,-1,-1),previous.expand(size,-1,-1,-1),
                torch.zeros_like(current).expand(size,-1,-1,-1),past.expand(size,-1),role+1,role)
            physical=current.expand(size,-1,-1,-1);forecasts=[]
            for k in range(10):hidden,physical=model.step(hidden,physical,latent,commands[:,k],role);forecasts.append(physical)
            prediction=torch.stack(forecasts,1);factual=[]
            for j,r in enumerate(rows):
                length=len(branches[j]['controls']);truth=observed[j,:length];estimated=prediction[j,:length]
                factual.append(dict(word=r['word'],length=length,forecast_mse=float((estimated-truth).square().mean()),
                    copy_mse=float((current-truth).square().mean())))
            odd=[];real_fields=[];predicted_fields=[]
            for axis in range(4):
                plus=next(i for i,r in enumerate(rows)if r['word']==f'axis{axis}-sign1');minus=next(i for i,r in enumerate(rows)if r['word']==f'axis{axis}-sign-1')
                real=(observed[plus,:5]-observed[minus,:5]).flatten();predicted=(prediction[plus,:5]-prediction[minus,:5]).flatten()
                real_fields.append(real);predicted_fields.append(predicted)
                odd.append(dict(axis=axis,true_norm=float(real.norm()),predicted_norm=float(predicted.norm()),
                    cosine=float(torch.dot(real,predicted)/(real.norm()*predicted.norm()).clamp_min(1e-12))))
            long=[j for j,b in enumerate(branches)if len(b['controls'])==10];goals=observed[long,-1];truth=observed[long,:5];size=len(long);role=current.new_ones(size)
            goal_errors={}
            for name,g in [('matched',goals),('other_branch',goals.roll(1,0)),('current',current.expand(size,-1,-1,-1))]:
                hidden,latent=model.prepare(current.expand(size,-1,-1,-1),previous.expand(size,-1,-1,-1),g,current.new_zeros(size,4),role,role)
                generated=current.expand(size,-1,-1,-1);errors=[]
                for k in range(5):hidden,generated=model.step(hidden,generated,latent,current.new_zeros(size,4),role);errors.append((generated-truth[:,k]).square().mean((1,2,3)))
                goal_errors[name]=float(torch.stack(errors,1).mean())
            real_matrix=torch.stack(real_fields,1);predicted_matrix=torch.stack(predicted_fields,1)
            gramian=real_matrix.T@real_matrix/len(real_matrix);predicted_gramian=predicted_matrix.T@predicted_matrix/len(predicted_matrix)
            reports.append(dict(task=context['task'],context=context['context'],factual=factual,odd_response=odd,goal_prediction_mse=goal_errors,
                observed_control_gramian=gramian.cpu().tolist(),predicted_control_gramian=predicted_gramian.cpu().tolist(),
                observed_control_gramian_eigenvalues=torch.linalg.eigvalsh(gramian).cpu().tolist()))
            print(json.dumps(dict(context=context['context'],goal=goal_errors,gripper=odd[3])),flush=True)
    assert before==model_sha(model)
    save(a.output,dict(passed=True,model_unchanged=True,model_sha256=before,weights_sha256=sha(weights),manifest_sha256=sha(a.manifest),
        rows=reports,actual_physical_contexts=50,new_updates=0,no_privileged_model_inputs=True,
        scope='Frozen original-data learners tested on separate new VALIDATION own-interventions.No action reconstruction/classifier/learning.'
              'Goals are achieved futureRGB only.No command inputs to goal queries.Factual controls only condition corresponding forward transitions.'
              'RawRGB errors/cosines are shared observation units;these are prediction diagnostics,not task-success scores.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['seal','audit'],required=True)
    for key in ['root','manifest','run','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--step',type=int,default=2500);a=p.parse_args()
    if a.mode=='seal':seal(a.root,a.output)
    else:audit(a)
