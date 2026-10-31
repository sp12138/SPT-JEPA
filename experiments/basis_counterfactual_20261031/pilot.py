"""Single-case native-CEM counterfactual pilot. No training or automatic dispatch.

--cpu-contracts runs only numerical/native-CPU sampler and one real-case replay
contracts, with no model load. The separate --pilot path requires explicit two
checkpoint paths and is left for root's frozen-source GPU preflight.
"""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, os, sys
from pathlib import Path
import numpy as np
import torch

B=Path('/efs/prescottsun/SPT-JEPA')
DEFAULT_MANIFEST=B/'data/hardtasks-v1/tworoom-metadata/manifests/validation-strict.json'
DATA=B/'data/hardtasks-v1/tworoom-metadata'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
    return h.hexdigest()


def array_hash(x):
    if torch.is_tensor(x):x=x.detach().cpu().numpy()
    a=np.asarray(x);return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def activate(path):
    source=json.loads(Path(path).read_text())
    assert source['source_commit']=='da26748c866137ad1ea06cf7250cabd64431e9ac'
    for part in reversed(source['pythonpath'].split(':')):sys.path.insert(0,part)
    manifest=json.loads(Path(source['manifest']).read_text())
    for name in ['as_jepa_lewm/spatial_carrier.py','as_jepa_lewm/model.py','as_jepa_lewm/robust.py','as_jepa_tools/benchmark_evaluate.py']:
        assert digest(Path(source['source'])/name)==manifest['files'][name]
    return source


def ranking(predicted,true):
    from scipy.stats import rankdata
    p,t=np.asarray(predicted,float),np.asarray(true,float)
    assert p.shape==t.shape and p.ndim==1 and len(p)>1 and np.isfinite(p).all() and np.isfinite(t).all()
    i=int(np.argmin(p));rp,rt=rankdata(p),rankdata(t);span=float(np.ptp(t))
    correlation=None if np.std(rp)==0 or np.std(rt)==0 else float(np.corrcoef(rp,rt)[0,1])
    return dict(spearman=correlation,selected_index=i,oracle_index=int(np.argmin(t)),
        selected_true_cost=float(t[i]),oracle_cost=float(t.min()),regret=float(t[i]-t.min()),
        span_normalized_regret=None if span==0 else float((t[i]-t.min())/span),
        selected_true_percentile=float((rt[i]-1)/(len(t)-1)),true_cost_span=span)


def setup(manifest_path,data):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf,open_dict
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm
    from clear_lewm.datasets import metadata_fingerprint
    from clear_lewm.protocols import protocol_from_dict
    from clear_lewm.runner import _image_transform
    from stable_worldmodel.world.world import _extract_init_goal
    manifest=json.loads(Path(manifest_path).read_text());assert manifest['task']=='tworoom' and manifest['campaign_split']=='validation'
    spec=protocol_from_dict(manifest['protocol']);card=json.loads((data/'dataset_card.json').read_text())
    assert metadata_fingerprint(card['h5_path'])==manifest['dataset']['fingerprint']['value']
    assert manifest['pairs'][0]['episode_id'] in np.load(data/'validation_episodes.npy')
    with initialize_config_dir(config_dir=str(B/'sources/stable-worldmodel/scripts/plan/config'),version_base=None):cfg=compose(config_name='tworoom')
    with open_dict(cfg):
        cfg.world.num_envs=1;cfg.world.max_episode_steps=2*spec.eval_budget
        cfg.plan_config.history_len=3;cfg.solver.batch_size=1
    assert (cfg.plan_config.horizon,cfg.plan_config.receding_horizon,cfg.plan_config.action_block)==(5,5,5)
    assert cfg.solver.num_samples==300 and cfg.solver.n_steps==30 and cfg.solver.seed==42
    ds=swm.data.load_dataset(card['h5_path'],keys_to_cache=list(cfg.dataset.keys_to_cache))
    pair=manifest['pairs'][0]
    starts,goals,_=_extract_init_goal(ds,[pair['episode_id']],[pair['start_step']],spec.goal_offset)
    assert pair['goal_step']==pair['start_step']+spec.goal_offset
    process={}
    for col in cfg.dataset.keys_to_cache:
        vals=ds.get_col_data(col);vals=vals[np.isfinite(vals).all(axis=1)]
        scaler=StandardScaler().fit(vals)
        if col=='action':
            scaler.mean_=np.asarray(card['action_mean']);scaler.scale_=np.asarray(card['action_std']);scaler.var_=scaler.scale_**2
        process[col]=scaler
        if col!='action':process['goal_'+col]=scaler
    transform={'pixels':_image_transform(224),'goal':_image_transform(224)}
    return dict(manifest=manifest,spec=spec,cfg=cfg,ds=ds,process=process,transform=transform,
        callables=OmegaConf.to_container(cfg.eval.callables,resolve=True),
        expected_position=np.asarray(starts[0]['pos_agent'],dtype=np.float32),
        expected_goal=np.asarray(goals[0]['goal_pos_agent'],dtype=np.float32))


def make_world(ctx):
    import stable_worldmodel as swm
    from clear_lewm.runner import _install_task_success
    w=swm.World(**ctx['cfg'].world,image_shape=(224,224))
    _install_task_success(w,'tworoom',ctx['spec']);return w


def evaluate(world,ctx,budget):
    p=ctx['manifest']['pairs'][0]
    return world.evaluate(dataset=ctx['ds'],episodes_idx=[p['episode_id']],start_steps=[p['start_step']],
        goal_offset=ctx['spec'].goal_offset,eval_budget=budget,callables=ctx['callables'],video=None)


class Captured(Exception):pass


class CaptureCost(torch.nn.Module):
    def get_cost(self,info,actions):
        # Stop *inside the unmodified first native get_cost*, before any env step.
        self.bank=actions[:, :64].detach().cpu().clone()
        self.info={k:v[:,0].detach().cpu().clone() for k,v in info.items() if torch.is_tensor(v)}
        self.full_bank_hash=array_hash(actions)
        raise Captured()


def capture_bank(ctx,device):
    from hydra.utils import instantiate
    import stable_worldmodel as swm
    cost=CaptureCost();solver=instantiate(ctx['cfg'].solver,cost=cost,device=device)
    policy=swm.policy.WorldModelPolicy(solver=solver,config=swm.PlanConfig(**ctx['cfg'].plan_config),
        process=ctx['process'],transform=ctx['transform'])
    world=make_world(ctx);world.set_policy(policy)
    try:
        try:evaluate(world,ctx,25)
        except Captured:pass
        else:raise AssertionError('Native first cost was not intercepted')
        assert cost.bank.shape==(1,64,5,10) and torch.equal(cost.bank[:,0],torch.zeros_like(cost.bank[:,0]))
        assert cost.info['pixels'].shape==cost.info['goal'].shape==(1,1,3,224,224)
        if 'action_history' in cost.info:assert cost.info['action_history'].shape==(1,0,10)
        return cost.bank[0],cost.info,dict(full300_hash=cost.full_bank_hash,first64_hash=array_hash(cost.bank),
            device=str(device),seed=42,iteration=0,full_samples=300,retained_indices=list(range(64)))
    finally:world.close()


class ReplaySolver:
    def __init__(self,sequence):self.sequence=sequence.detach().cpu().clone();self.calls=0
    def configure(self,*,action_space,n_envs,config):
        assert n_envs==1 and tuple(action_space.shape)==(1,2)
        self._n_envs=n_envs;self._action_dim=10;self._horizon=5
    @property
    def n_envs(self):return self._n_envs
    @property
    def action_dim(self):return self._action_dim
    @property
    def horizon(self):return self._horizon
    @property
    def dtype(self):return torch.float32
    def __call__(self,info_dict,init_action=None):return self.solve(info_dict,init_action)
    def solve(self,info_dict,init_action=None):
        assert init_action is None and self.calls==0
        assert info_dict['pixels'].shape==info_dict['goal'].shape==(1,1,3,224,224)
        self.info={k:v.detach().cpu().clone() for k,v in info_dict.items() if torch.is_tensor(v)}
        self.calls+=1;return {'actions':self.sequence[None]}


def replay(ctx,sequence,*,clip_before_env=False):
    """Original World.evaluate + WorldModelPolicy queue; no reset/step rewrite."""
    import stable_worldmodel as swm
    from clear_lewm.tworoom_runtime import topology_audit_records
    solver=ReplaySolver(sequence)
    policy=swm.policy.WorldModelPolicy(solver=solver,config=swm.PlanConfig(**ctx['cfg'].plan_config),
        process=ctx['process'],transform=ctx['transform'])
    world=make_world(ctx);world.set_policy(policy)
    trace=[];initial={};step=world.envs.step;get_action=policy.get_action
    expected=ctx['process']['action'].inverse_transform(sequence.numpy().reshape(25,2))
    def observed_action(info,**kwargs):
        if not initial:
            env=world.envs.envs[0].unwrapped;p=ctx['manifest']['pairs'][0]
            np.testing.assert_array_equal(np.asarray(env.agent_position),ctx['expected_position'])
            np.testing.assert_array_equal(np.asarray(env.target_position),ctx['expected_goal'])
            initial.update(position=np.asarray(env.agent_position).copy(),goal_position=np.asarray(env.target_position).copy(),
                pixels_hash=array_hash(info['pixels']),goal_hash=array_hash(info['goal']),
                raw_pixels_shape=list(info['pixels'].shape),raw_goal_shape=list(info['goal'].shape),
                geometry=copy.deepcopy(topology_audit_records()[0]['geometry']),episode=p['episode_id'],start=p['start_step'])
        out=get_action(info,**kwargs)
        # This proves native unblocking and inverse normalization; env clips later.
        np.testing.assert_array_equal(out[0],expected[len(trace)])
        return np.clip(out,-1,1) if clip_before_env else out
    policy.get_action=observed_action
    def observed_step(actions,mask=None):
        result=step(actions,mask=mask)
        trace.append(dict(action=actions[0].copy(),effective_action=np.clip(actions[0],-1,1),
            position=np.asarray(world.envs.envs[0].unwrapped.agent_position).copy(),
            pixels_hash=array_hash(result[-1]['pixels']),terminated=bool(result[2][0]),truncated=bool(result[3][0])))
        return result
    world.envs.step=observed_step
    try:
        metrics=evaluate(world,ctx,25);assert solver.calls==1 and 1<=len(trace)<=25
        if len(trace)<25:assert trace[-1]['terminated'] or trace[-1]['truncated']
        endpoint=np.asarray(world.infos['pixels']).copy()
        result=dict(initial=initial,trace=trace,success=bool(metrics['episode_successes'][0]),steps=len(trace),
            endpoint=endpoint,topology=copy.deepcopy(topology_audit_records()[0]),info=solver.info,
            clipped_coordinate_fraction=float(np.mean(np.abs(expected)>1)))
        return result
    finally:world.close()


def same_replay(a,b,compare_requested=True):
    for key in a['initial']:
        x,y=a['initial'][key],b['initial'][key]
        if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        else:assert x==y,key
    assert a['steps']==b['steps'] and a['success']==b['success']
    np.testing.assert_array_equal(a['endpoint'],b['endpoint'])
    for x,y in zip(a['trace'],b['trace']):
        for key in ['position','effective_action']+(['action'] if compare_requested else []):np.testing.assert_array_equal(x[key],y[key])
        for key in ['pixels_hash','terminated','truncated']:assert x[key]==y[key]


def cpu_contracts(ctx):
    bank,info,provenance=capture_bank(ctx,'cpu')
    bank2,info2,_=capture_bank(ctx,'cpu');assert torch.equal(bank,bank2)
    for key in ['pixels','goal','action_history']:
        if key in info:assert torch.equal(info[key],info2[key])
    generator=torch.Generator(device='cpu').manual_seed(42)
    expected=torch.randn(1,300,5,10,generator=generator);expected[:,0]=0
    assert torch.equal(bank,expected[0,:64])
    # Deliberate clipping and within-block variation: a contract, not a ranked bank.
    sequence=torch.arange(50,dtype=torch.float32).reshape(5,10).remainder(7)-3
    a,b=replay(ctx,sequence),replay(ctx,sequence);same_replay(a,b)
    c=replay(ctx,sequence,clip_before_env=True);same_replay(a,c,compare_requested=False)
    assert a['clipped_coordinate_fraction']>0
    assert ranking([2,1,0],[0,1,2])['regret']==2 and ranking([2,1,0],[0,1,2])['spearman']==-1
    assert ranking([0,1,2],[0,1,2])['regret']==0
    x=np.array([[1.,2.],[4.,-1.],[0.,3.]]);g=np.array([2.,-1.]);q=np.array([[0.,1.],[-1.,0.]])
    np.testing.assert_array_equal(((x-g)**2).sum(1),((x@q-g@q)**2).sum(1))
    from hydra.utils import instantiate
    objective=instantiate(ctx['cfg'].objective)
    predicted=torch.tensor(x,dtype=torch.float32)[None,:,None]
    goal=torch.tensor(g,dtype=torch.float32)[None,None]
    torch.testing.assert_close(objective({'predicted_emb':predicted,'goal_emb':goal}),
        (predicted[:,:,0]-goal).square().sum(-1),rtol=0,atol=0)
    return dict(passed=True,native_cpu_first64_oracle=True,cpu_not_cuda_bank=True,
        native_reset_repeat_trace_exact=True,native_inverse_normalization_and_unblocking_exact=True,
        env_clip_equivalence_exact=True,exact_dataset_start_goal_positions=True,native_goal_cost_oracle=True,
        steps=a['steps'],clipped_coordinate_fraction=a['clipped_coordinate_fraction'],
        initial={k:v for k,v in a['initial'].items() if k not in ['position','goal_position']},
        shapes={k:list(v.shape) for k,v in info.items() if k in ['pixels','goal','action','action_history']},
        bank=provenance,ranking_and_orthogonal_oracles=True,models_loaded=0)


def expand_info(info,count,device):
    return {k:v.to(device).unsqueeze(1).expand(v.shape[0],count,*v.shape[1:]).clone() for k,v in info.items()}


@torch.inference_mode()
def score(ctx,info,bank,endpoints,identity,basis,device):
    import stable_worldmodel as swm
    from hydra.utils import instantiate
    from vforge.ops import OpsConfig,initialize_ops
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device(device),parallel_state=None)
    inputs=[Path(identity),Path(basis)];weights=[torch.load(p/'weights.pt',map_location='cpu',weights_only=True) for p in inputs]
    common=[k for k in weights[0] if k.startswith(('encoder.','pool.')) or k in ['spatial_projection','spatial_mean','spatial_scale']]
    assert len(common)==210 and all(torch.equal(weights[0][k],weights[1][k]) for k in common)
    q=weights[1]['spatial_orthogonal_basis'].double();assert torch.equal(q.T@q,torch.eye(1024,dtype=torch.double))
    outputs={};scaler=swm.policy.BasePolicy();scaler.process={};scaler.transform=ctx['transform']
    images=scaler._prepare_info({'pixels':np.concatenate(endpoints,axis=0)})['pixels']
    for name,path,w in zip(['identity','basis'],inputs,weights):
        sha=digest(path/'weights.pt');model=swm.wm.utils.load_pretrained(str(path)).to(device).eval();model.requires_grad_(False)
        assert set(w)==set(model.state_dict()) and all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in w.items())
        assert model.spatial_basis==('identity' if name=='identity' else 'orthogonal')
        objective=instantiate(ctx['cfg'].objective);cost=swm.planning.ShootingCostEvaluator(model,objective)
        pred=[];observed=[]
        goal=model.encode({'pixels':info['goal'].to(device)})['emb']
        for start in range(0,64,8):
            candidate=bank[start:start+8][None].to(device)
            pred.append(cost.get_cost(expand_info(info,len(candidate[0]),device),candidate).cpu().flatten())
            observed.append(model.encode({'pixels':images[start:start+8].to(device)})['emb'].cpu())
        observed=torch.cat(observed);p=torch.cat(pred)
        true=objective({'predicted_emb':observed[None], 'goal_emb':goal.cpu()}).flatten()
        assert true.shape==p.shape==(64,)
        assert digest(path/'weights.pt')==sha and all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in w.items())
        outputs[name]=dict(checkpoint=str(path),weights_sha256=sha,predicted_cost=p.tolist(),true_cost=true.tolist(),
            ranking=ranking(p.numpy(),true.numpy()),weights_and_buffers_unchanged=True)
        del model,cost
    a,b=np.asarray(outputs['identity']['true_cost']),np.asarray(outputs['basis']['true_cost'])
    tolerance=5e-6*max(1.,float(np.max(np.abs(a))))
    assert np.max(np.abs(a-b))<=tolerance,'Observed Q-invariant cost contract failed; stop interpretation'
    return dict(models=outputs,observed_cost_max_abs_difference=float(np.max(np.abs(a-b))),
        observed_cost_tolerance=tolerance,common_frozen210_exact=True,Q_orthogonal_exact_fp64=True)


def pilot(ctx,args):
    assert args.identity and args.basis and args.device.startswith('cuda')
    bank,info,provenance=capture_bank(ctx,args.device)
    branches=[replay(ctx,candidate) for candidate in bank]
    same_replay(branches[0],replay(ctx,bank[0]));same_replay(branches[1],replay(ctx,bank[1]))
    # Every branch must see exactly the same native initial observation/goal.
    for branch in branches:
        for key in ['pixels','goal','action_history']:
            if key in info:assert torch.equal(info[key],branch['info'][key])
        for key in ['position','goal_position']:np.testing.assert_array_equal(branch['initial'][key],branches[0]['initial'][key])
        assert branch['initial']['geometry']==branches[0]['initial']['geometry']
    result=score(ctx,info,bank,[b['endpoint'] for b in branches],args.identity,args.basis,args.device)
    result.update(bank=provenance,case=ctx['manifest']['pairs'][0],branches=[dict(index=i,steps=b['steps'],
        early_terminal=b['steps']<25,success_within25=b['success'],endpoint_position=b['trace'][-1]['position'].tolist(),
        clipped_coordinate_fraction=b['clipped_coordinate_fraction'],topology=b['topology']) for i,b in enumerate(branches)],
        scope='Single first-primary-case technical pilot,64 fixed candidates. Native terminal-frozen25-step endpoints; '
              'not full50-step P0, not multiround CEM performance, not training labels. No method winner from one case.')
    np.savez_compressed(Path(args.output)/'bank-endpoints.npz',bank=bank.numpy(),endpoints=np.concatenate([b['endpoint'] for b in branches]))
    return result


def provenance(source,args,ctx):
    import stable_worldmodel.policy as policy
    from stable_worldmodel.world.world import World
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from stable_worldmodel.planning.evaluator import ShootingCostEvaluator
    from clear_lewm.runner import _install_task_success
    from omegaconf import OmegaConf
    paths={inspect.getsourcefile(x) for x in [World,TwoRoomEnv,CEMSolver,ShootingCostEvaluator,policy.WorldModelPolicy,_install_task_success]}
    for name in ['world/env_pool.py','wrapper/default.py','wm/lewm/lewm.py','planning/objective.py','planning/solver/utils.py']:
        paths.add(str(B/'sources/stable-worldmodel/stable_worldmodel'/name))
    for name in ['tworoom_runtime.py','topology.py']:
        paths.add(str(B/'sources/clear-lewm-20261030/clear_lewm'/name))
    return dict(source=source,script_sha256=digest(__file__),manifest_sha256=digest(args.manifest),
        native_files={p:digest(p) for p in sorted(paths)},solver=OmegaConf.to_container(ctx['cfg'].solver,resolve=True),
        plan=OmegaConf.to_container(ctx['cfg'].plan_config,resolve=True),protocol=ctx['manifest']['protocol'],
        action_mean=ctx['process']['action'].mean_.tolist(),action_scale=ctx['process']['action'].scale_.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-record',required=True)
    p.add_argument('--manifest',default=str(DEFAULT_MANIFEST));p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--output',required=True);p.add_argument('--identity');p.add_argument('--basis');p.add_argument('--device',default='cuda')
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--cpu-contracts',action='store_true');modes.add_argument('--pilot',action='store_true')
    args=p.parse_args();source=activate(args.source_record)
    if args.cpu_contracts:assert not torch.cuda.is_initialized(),'CPU contract must not initialize CUDA'
    os.environ.setdefault('SDL_VIDEODRIVER','dummy');torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    ctx=setup(args.manifest,args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=cpu_contracts(ctx) if args.cpu_contracts else pilot(ctx,args)
    result['provenance']=provenance(source,args,ctx)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),cpu_contracts=args.cpu_contracts,completed=True)),flush=True)


if __name__=='__main__':main()
