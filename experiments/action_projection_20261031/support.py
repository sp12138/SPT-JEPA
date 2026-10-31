"""CPU-only Cube training support and installed action semantics; no env/model."""
import ast, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import h5py, hdf5plugin
import numpy as np
import gymnasium as gym

B = Path('/efs/prescottsun/SPT-JEPA')
D = B/'data/hardtasks-v1/cube-metadata'
OUT = B/'reports/action-projection-20261031/cube-support.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    source = Path('/efs/prescottsun/project_jepa/envs/lewm/lib/python3.12/site-packages/ogbench/manipspace/envs/manipspace_env.py')
    tree = ast.parse(source.read_text()); cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ManipSpaceEnv')
    methods = {x.name:x for x in cls.body if isinstance(x, ast.FunctionDef)}
    namespace = dict(np=np, gym=gym)
    for name in ['action_space', 'unnormalize_action']:
        node = methods[name]; node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    init = methods['__init__']; assignment = next(x for x in init.body if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) and x.targets[0].id == 'action_range')
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), 'exec'), namespace)
    state = SimpleNamespace(action_low=-namespace['action_range'], action_high=namespace['action_range'])
    space = namespace['action_space'](state)
    np.testing.assert_array_equal(space.low, -np.ones(5)); np.testing.assert_array_equal(space.high, np.ones(5))
    command = np.array([2., 0., 0., 0., 0.], np.float32)
    raw_delta = namespace['unnormalize_action'](state, command)
    bounded_delta = namespace['unnormalize_action'](state, np.clip(command, space.low, space.high))
    assert np.isclose(raw_delta[0], .1) and np.isclose(bounded_delta[0], .05)
    # This verifies an interior target counterexample, not a simulated trajectory.
    assert .25 < .4 + bounded_delta[0] < .4 + raw_delta[0] < .6
    cube = B/'sources/stable-worldmodel/stable_worldmodel/envs/ogbench/cube_env.py'
    cube_tree = ast.parse(cube.read_text()); cube_cls = next(x for x in cube_tree.body if isinstance(x, ast.ClassDef) and x.name == 'CubeEnv')
    assert not {'step','set_control','unnormalize_action','action_space'} & {x.name for x in cube_cls.body if isinstance(x, ast.FunctionDef)}
    card = json.loads((D/'dataset_card.json').read_text()); train = np.load(D/'train_episodes.npy')
    ids, offsets, lengths = [np.load(D/f) for f in ['episode_ids.npy','offsets.npy','lengths.npy']]
    selected = np.flatnonzero(np.isin(ids,train)); assert len(selected) == len(train) == 9000
    counters = dict(rows=0,finite_rows=0,finite_coordinates=0,outside_coordinates=0,outside_rows=0,terminal_all_nan=0,window8_stride5_outgoing_rows=0)
    lo=np.full(5,np.inf); hi=-lo; action_hash=hashlib.sha256()
    with h5py.File(card['h5_path'],'r') as data:
        np.testing.assert_array_equal(data['ep_offset'][:], offsets); np.testing.assert_array_equal(data['ep_len'][:], lengths)
        for ep in selected:
            start,n = int(offsets[ep]),int(lengths[ep]); a=np.asarray(data['action'][start:start+n])
            assert np.all(data['ep_idx'][start:start+n] == ids[ep]); finite=np.isfinite(a)
            outside=finite & ((a<space.low)|(a>space.high)); valid=a[finite.all(1)]
            counters['rows']+=n; counters['finite_rows']+=len(valid); counters['finite_coordinates']+=int(finite.sum())
            counters['outside_coordinates']+=int(outside.sum()); counters['outside_rows']+=int(outside.any(1).sum())
            counters['terminal_all_nan']+=int(np.isnan(a[-1]).all())
            if n>35: counters['window8_stride5_outgoing_rows']+=n-1
            lo=np.minimum(lo,np.nanmin(a,axis=0)); hi=np.maximum(hi,np.nanmax(a,axis=0)); action_hash.update(a.tobytes())
    result=dict(task='cube',training_episodes=len(train),counts=counters,minimum=lo.tolist(),maximum=hi.tolist(),
        outside_finite_coordinate_fraction=counters['outside_coordinates']/counters['finite_coordinates'],
        action_space=dict(low=space.low.tolist(),high=space.high.tolist()),
        counterexample=dict(command=command.tolist(),unnormalized_delta=raw_delta.tolist(),bounded_delta=bounded_delta.tolist(),initial_effector_x=.4,original_target_x=float(.4+raw_delta[0]),bounded_target_x=float(.4+bounded_delta[0]),both_inside_workspace=True),
        cube_inherits_action_methods=True,semantics='Cube does not coordinate-clip incoming action in step/set_control. It affine-scales the command then state-dependently clips workspace/gripper targets. Box clipping therefore changes the Cube controller; scoring-only clipping would mismatch execution.',
        source_files={str(p):sha(p) for p in [source,cube,B/'sources/stable-worldmodel/stable_worldmodel/wrapper/default.py',B/'sources/stable-worldmodel/stable_worldmodel/envs/ogbench/expert_policy.py',B/'sources/clear-lewm-20261030/clear_lewm/runner.py']},
        acquisition=card['acquisition'],dataset_card_sha256=sha(D/'dataset_card.json'),train_ids_sha256=sha(D/'train_episodes.npy'),ordered_training_action_bytes_sha256=action_hash.hexdigest(),
        provenance_caveat='Current wrapper records incoming commands and current ExpertPolicy clips before return. Acquired H5 has a dataset revision but no historical generator commit; its actual bounds are independently measured here. No training labels changed.',
        cpu_only=True,models_loaded=0,environments_created=0,new_rollouts=0)
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
