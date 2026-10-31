"""Candidate: score known actuator projection on the saved single-case bank.

No environment construction/reset/rollout. --cpu-contracts loads zero models;
--score needs root's explicit frozen-source GPU authorization.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,inspect,json,os
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

B=Path('/efs/prescottsun/SPT-JEPA')
ORIGINAL=B/'runs/basis_counterfactual_pilot_20261031'
FROZEN=B/'runs/basis_counterfactual_source_20261031/source-v1'
PILOT_SHA='dad92cea2711b9e18fa826f1f84bb7f97c3bd31b288d7e9ca946cd457558a182'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original():
    path=FROZEN/'experiments/basis_counterfactual_20261031/pilot.py';assert sha(path)==PILOT_SHA
    spec=importlib.util.spec_from_file_location('frozen_counterfactual_pilot',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=json.loads((ORIGINAL/'results.json').read_text())
    assert result['provenance']['script_sha256']==PILOT_SHA and result['bank']['device']=='cuda'
    for path,digest in result['provenance']['native_files'].items():assert sha(path)==digest
    bank=np.load(ORIGINAL/'bank-endpoints.npz')['bank'];assert bank.shape==(64,5,10) and bank.dtype==np.float32
    assert module.array_hash(bank[None])==result['bank']['first64_hash']
    scaler=StandardScaler();scaler.mean_=np.asarray(result['provenance']['action_mean']);scaler.scale_=np.asarray(result['provenance']['action_scale'])
    scaler.var_=scaler.scale_**2;scaler.n_features_in_=2
    return module,result,bank,scaler


def project(bank,scaler):
    raw=scaler.inverse_transform(bank.reshape(-1,2)).reshape(64,25,2)
    effective=np.clip(raw,-1.,1.)
    normalized=scaler.transform(effective.reshape(-1,2)).reshape(bank.shape)
    return normalized,raw,effective


def projection_contracts(bank,scaler):
    projected,raw,effective=project(bank,scaler)
    # Installed sklearn first casts statistics to input dtype, then performs
    # separate in-place operations. Make both casts/roundings explicit without
    # calling inverse_transform/transform in this independent oracle.
    scale32=scaler.scale_.astype(np.float32).astype(np.float64)
    mean32=scaler.mean_.astype(np.float32).astype(np.float64)
    oracle=(bank.reshape(-1,2).astype(np.float64)*scale32).astype(np.float32)
    oracle=(oracle.astype(np.float64)+mean32).astype(np.float32)
    np.testing.assert_array_equal(oracle.reshape(raw.shape),raw)
    oracle=np.maximum(np.minimum(oracle,np.float32(1)),np.float32(-1))
    oracle=(oracle.astype(np.float64)-mean32).astype(np.float32)
    oracle=(oracle.astype(np.float64)/scale32).astype(np.float32).reshape(bank.shape)
    np.testing.assert_array_equal(projected,oracle)
    repeated,_,effective_twice=project(projected,scaler)
    idempotence=float(np.max(np.abs(repeated-projected)))
    execution=float(np.max(np.abs(effective_twice-effective)))
    # Explicit float32 normalization roundoff; no task/validation tuned floor.
    tolerance=float(8*np.finfo(np.float32).eps*max(1,float(np.max(np.abs(projected)))))
    assert idempotence<=tolerance and execution<=tolerance
    assert np.isfinite(projected).all() and np.array_equal(projected[0],bank[0])
    import sklearn
    return projected,dict(exact_sklearn_vs_scalar_rounding_oracle=True,sklearn_version=sklearn.__version__,
        scaler_method_source_sha256=hashlib.sha256((inspect.getsource(StandardScaler.inverse_transform)+inspect.getsource(StandardScaler.transform)).encode()).hexdigest(),
        max_projection_idempotence_error=idempotence,max_executed_action_difference=execution,
        float32_roundoff_tolerance=tolerance,zero_mean_candidate_unchanged=True,
        clipped_coordinate_fraction=float((np.abs(raw)>1).mean()),
        requested_effective_action_rmse=float(np.sqrt(np.mean((bank-projected)**2))))


def original_initial_info(p,original,scaler):
    """Recover the existing initial/goal image pair directly; never create envs."""
    import h5py,hdf5plugin
    from clear_lewm.runner import _image_transform
    from stable_worldmodel.policy import BasePolicy
    card_path=B/'data/hardtasks-v1/tworoom-metadata/dataset_card.json';card=json.loads(card_path.read_text())
    np.testing.assert_array_equal(card['action_mean'],scaler.mean_);np.testing.assert_array_equal(card['action_std'],scaler.scale_)
    pair=original['case'];assert (pair['episode_id'],pair['start_step'],pair['goal_step'])==(94,36,61)
    with h5py.File(card['h5_path'],'r') as f:
        images=[np.asarray(f['pixels'][pair[k]])[None,None] for k in ['start_row','goal_row']]
    cpu=json.loads((FROZEN/'experiments/basis_counterfactual_20261031/cpu-contracts.json').read_text())
    assert p.array_hash(images[0])==cpu['initial']['pixels_hash'] and p.array_hash(images[1])==cpu['initial']['goal_hash']
    process=BasePolicy();process.process={'action':scaler};process.transform={'pixels':_image_transform(224),'goal':_image_transform(224)}
    info=process._prepare_info(dict(pixels=images[0],goal=images[1],action=np.full((1,1,2),np.nan,dtype=np.float32)))
    assert info['pixels'].shape==info['goal'].shape==(1,1,3,224,224)
    return info,dict(exact_original_raw_initial_goal_hashes=True,first_context_frames=1,goal_frames=1,
        future_observation_to_predictor=False,environment_constructed=False,dataset_card_sha256=sha(card_path))


@torch.inference_mode()
def score(p,original,bank,projected,info,device):
    import stable_worldmodel as swm
    from vforge.ops import OpsConfig,initialize_ops
    from stable_worldmodel.planning import GoalMSE,ShootingCostEvaluator
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device(device),parallel_state=None)
    outputs={}
    for name in ['identity','basis']:
        old=original['models'][name];path=Path(old['checkpoint']);assert sha(path/'weights.pt')==old['weights_sha256']
        weights=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
        model=swm.wm.utils.load_pretrained(str(path)).to(device).eval();model.requires_grad_(False)
        assert set(weights)==set(model.state_dict()) and all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in weights.items())
        cost=ShootingCostEvaluator(model,GoalMSE(reduction='sum'))
        # One original8-candidate chunk validates reconstructed input semantics.
        replay=cost.get_cost(p.expand_info(info,8,device),torch.from_numpy(bank[:8])[None].to(device)).cpu().flatten().numpy()
        saved=np.asarray(old['predicted_cost'][:8]);assert np.allclose(replay,saved,rtol=1e-6,atol=2e-4)
        predicted=[]
        for start in range(0,64,8):
            actions=torch.from_numpy(projected[start:start+8])[None].to(device)
            predicted.append(cost.get_cost(p.expand_info(info,8,device),actions).cpu().flatten())
        predicted=torch.cat(predicted).numpy();true=np.asarray(old['true_cost'])
        assert np.isclose(predicted[0],old['predicted_cost'][0],rtol=1e-6,atol=2e-4)
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in weights.items())
        assert sha(path/'weights.pt')==old['weights_sha256']
        outputs[name]=dict(checkpoint=str(path),weights_sha256=old['weights_sha256'],weights_and_buffers_unchanged=True,
            original8_cost_reproduction_max_abs=float(np.max(np.abs(replay-saved))),
            original_ranking=old['ranking'],projected_ranking=p.ranking(predicted,true),
            original_predicted_cost=old['predicted_cost'],projected_predicted_cost=predicted.tolist(),true_cost=old['true_cost'])
        del cost,model,weights
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-record',default=str(B/'runs/local_basis2000_20261031/source-record.json'))
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda')
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--cpu-contracts',action='store_true');modes.add_argument('--score',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    p,original,bank,scaler=load_original();source=p.activate(args.source_record)
    projected,contracts=projection_contracts(bank,scaler);info,input_contract=original_initial_info(p,original,scaler)
    output=dict(passed=True,source=source,projection_contracts=contracts,input_contract=input_contract,
        original_results=dict(path=str(ORIGINAL/'results.json'),sha256=sha(ORIGINAL/'results.json')),
        original_bank_hash=original['bank']['first64_hash'],projected_bank_hash=p.array_hash(projected[None]),
        projection='P(z)=StandardScaler.transform(clip(StandardScaler.inverse_transform(z),-1,1))',
        script_sha256=sha(__file__),cpu_contracts=args.cpu_contracts,new_environment_rollouts=0,
        scope='Single original case/64-bank/true endpoints unchanged. A common planner-input treatment for both models, '
              'not a new algorithm or proved bug fix. True costs reused under unchanged checkpoint hashes. '
              'GPU branch first reproduces8 original scores, then scores64 projected candidates per model; no expansion or training.')
    if args.cpu_contracts:
        assert not torch.cuda.is_initialized();output['models_loaded']=0;output['model_forward_candidates']=0
    else:
        assert args.device.startswith('cuda');output['models']=score(p,original,bank,projected,info,args.device)
        output['model_forward_candidates']=2*(8+64)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),passed=True,cpu_contracts=args.cpu_contracts)),flush=True)


if __name__=='__main__':main()
