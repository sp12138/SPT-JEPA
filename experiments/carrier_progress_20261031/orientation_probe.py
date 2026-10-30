"""CPU-only orientation readouts from already-saved train/validation embeddings.

Diagnostic physical labels fit only a fixed readout. No encoder, policy, model,
planner, metric calibration, evaluation case or frozen source is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: register the existing dataset's compression filters
import numpy as np
from scipy.spatial.transform import Rotation
from clear_lewm.tasks import (
    CUBE_SYMMETRY_MATRICES, _quaternion_matrix_wxyz, cube_symmetry_angle_deg,
)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb")as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def invariant_from_rotation(rotation):
    # Columns are the three unoriented cube axes. All 81 entries are retained;
    # duplicated symmetric entries give the ordinary tensor Frobenius norm.
    return np.einsum("...ia,...ja,...ka,...la->...ijkl",rotation,rotation,rotation,rotation,
                     optimize=True).reshape(*rotation.shape[:-2],81)


def invariant(quaternion):return invariant_from_rotation(_quaternion_matrix_wxyz(quaternion))


def quaternion_from_matrix(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    return q[..., [3,0,1,2]]


def project_rotation(values):
    matrix=np.asarray(values).reshape(-1,3,3)
    u,_,vh=np.linalg.svd(matrix)
    correction=np.ones((len(matrix),3));correction[:,-1]=np.linalg.det(u@vh)
    return quaternion_from_matrix((u*correction[:,None,:])@vh)


def ridge_fit(x,y):
    # Exactly the existing benchmark_information_audit ridge protocol.
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    center,scale=x.mean(0),np.maximum(x.std(0),1e-12)
    standardized=(x-center)/scale;target_mean=y.mean(0)
    coefficient=np.linalg.solve(standardized.T@standardized/len(x)+.001*np.eye(x.shape[1]),
                                standardized.T@(y-target_mean)/len(x))
    return coefficient/scale[:,None], target_mean-center@(coefficient/scale[:,None]), target_mean


def feature_metrics(prediction,target,train_mean):
    error=np.square(prediction-target).mean(0)
    reference=np.square(target-train_mean).mean(0)
    valid=reference>1e-12
    r2=1-error/np.maximum(reference,1e-20)
    return dict(mse=error.tolist(),constant_training_mean_mse=reference.tolist(),
        r2_against_train_mean=[float(v)if ok else None for v,ok in zip(r2,valid)],
        mean_r2=float(r2[valid].mean())if valid.any()else None,
        variance_weighted_r2=float(1-error.sum()/reference.sum())if reference.sum()>1e-12 else None,
        informative_dimensions=int(valid.sum()))


def decode_codebook(features,bank,quaternions):
    squared=np.square(features).sum(-1)[:,None]+np.square(bank).sum(-1)[None]-2*features@bank.T
    indices=squared.argmin(-1)
    return quaternions[indices],indices


def angle_metrics(angles,windows):
    return dict(mean_deg=float(angles.mean()),median_deg=float(np.median(angles)),
        p90_deg=float(np.quantile(angles,.9)),fraction_within15deg=float((angles<=15).mean()),
        per_frame_deg=angles.tolist(),per_window_mean_deg=angles.reshape(windows,-1).mean(1).tolist())


def load_quaternions(h5,manifest):
    rows=np.asarray([x["rows"]for x in manifest["windows"]],dtype=np.int64)
    unique,inverse=np.unique(rows.reshape(-1),return_inverse=True)
    expected=np.repeat([x["episode"]for x in manifest["windows"]],rows.shape[1])
    np.testing.assert_array_equal(h5["ep_idx"][unique][inverse],expected)
    q=h5["privileged_block_0_quat"][unique][inverse].reshape(*rows.shape,4)
    if not np.isfinite(q).all()or np.any(np.linalg.norm(q,axis=-1)<1e-6):raise ValueError("Invalid quaternion labels")
    return q


def contracts():
    rotation=Rotation.from_euler("xyz",[[11,23,37],[43,2,-19]],degrees=True).as_matrix()
    q=quaternion_from_matrix(rotation)
    sign_error=float(np.max(np.abs(invariant(q)-invariant(-q))))
    symmetry_error=max(float(np.max(np.abs(invariant_from_rotation(rotation@s)-invariant(q))))
                       for s in CUBE_SYMMETRY_MATRICES)
    pinned_error=max(float(np.max(cube_symmetry_angle_deg(q,quaternion_from_matrix(rotation@s))))
                     for s in CUBE_SYMMETRY_MATRICES)
    assert sign_error==0 and symmetry_error<1e-12 and pinned_error<2e-6
    identity=np.array([[1.,0,0,0]])
    quarter=quaternion_from_matrix(Rotation.from_euler("z",90,degrees=True).as_matrix()[None])
    assert cube_symmetry_angle_deg(identity,quarter)[0]<1e-6
    return dict(quaternion_sign_max_error=sign_error,cube24_feature_max_error=symmetry_error,
                pinned_cube24_angle_max_deg=pinned_error,passed=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration-root",default="/efs/prescottsun/SPT-JEPA/runs/carrier_metric_calibration_20261031")
    p.add_argument("--gate-root",default="/efs/prescottsun/SPT-JEPA/runs/ideas_validate600_20261031")
    p.add_argument("--output",required=True);args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if(output/"results.json").exists():raise FileExistsError("Preserve existing analysis")
    calibration_root,gate=Path(args.calibration_root),Path(args.gate_root)
    completed,weights={},{}
    import clear_lewm.tasks as task_source
    contract=contracts();data_cache={};rows=[]
    modes=["observed_future","clean_rollout","teacher_forced_prefix","initial_carry","shifted_observed_carry"]
    for seed in [1,2]:
        for arm in ["identity","bounded"]:
            name=f"cube-carrier-{arm}-seed{seed}";cal=calibration_root/("calibration-"+name)
            meta=read(cal/"calibration.json");audit_dir=gate/("transition-"+name)
            audit=read(audit_dir/"state-audit.json");train_manifest=read(meta["train_manifest"])
            val_manifest=read(audit_dir/"windows.json")
            assert meta["weights_sha256"]==audit["weights_sha256"]
            assert digest(meta["train_manifest"])==meta["train_manifest_sha256"]
            assert digest(audit_dir/"windows.json")==audit["manifest_sha256"]
            assert digest(cal/"training-embeddings.npz")==meta["training_embeddings_sha256"]
            assert train_manifest["split"]=="train"and val_manifest["split"]=="validation"
            assert len(train_manifest["windows"])==128 and len(val_manifest["windows"])==64
            train_eps={x["episode"]for x in train_manifest["windows"]}
            val_eps={x["episode"]for x in val_manifest["windows"]}
            assert not train_eps.intersection(val_eps)
            key=(meta["train_manifest_sha256"],audit["manifest_sha256"])
            if key not in data_cache:
                data=Path(meta["data"]);card=read(data/"dataset_card.json")
                assert digest(data/"dataset_card.json")==meta["dataset_card_sha256"]
                for manifest in [train_manifest,val_manifest]:
                    for file,sha in manifest["metadata_sha256"].items():assert digest(data/file)==sha
                with h5py.File(card["h5_path"],"r",swmr=True)as h5:
                    train_q=load_quaternions(h5,train_manifest).reshape(-1,4)
                    val_q=load_quaternions(h5,val_manifest)[:,1:].reshape(-1,4)
                train_features=invariant(train_q);val_features=invariant(val_q)
                nearest_true,_=decode_codebook(val_features,train_features,train_q)
                angular_floor=[]
                for start in range(0,len(val_q),32):
                    angular_floor.extend(cube_symmetry_angle_deg(val_q[start:start+32,None],train_q[None]).min(1).tolist())
                constant_q,_=decode_codebook(train_features.mean(0,keepdims=True),train_features,train_q)
                const_angles=cube_symmetry_angle_deg(np.repeat(constant_q,len(val_q),axis=0),val_q)
                refs=dict(training_codebook_best_possible_angle=angle_metrics(np.asarray(angular_floor),64),
                    true_invariant_feature_codebook_decode=angle_metrics(cube_symmetry_angle_deg(nearest_true,val_q),64),
                    constant_training_invariant_mean_decode=angle_metrics(const_angles,64))
                data_cache[key]=(train_q,val_q,train_features,val_features,refs)
            train_q,val_q,train_features,val_features,refs=data_cache[key]
            with np.load(cal/"training-embeddings.npz")as stored:train_z=stored["clips"].reshape(-1,64).astype(float)
            with np.load(audit_dir/"state-audit.npz")as stored:states={mode:stored[mode].reshape(-1,64).astype(float)for mode in modes}
            assert train_z.shape==(1024,64)and all(z.shape==(448,64)for z in states.values())
            target_kinds=dict(cube24_tensor81=(train_features,val_features),
                rotation_matrix9=(_quaternion_matrix_wxyz(train_q).reshape(-1,9),_quaternion_matrix_wxyz(val_q).reshape(-1,9)))
            scores={}
            for kind,(train_y,val_y)in target_kinds.items():
                matrix,intercept,mean=ridge_fit(train_z,train_y)
                weights[name+"__"+kind+"__matrix"]=matrix;weights[name+"__"+kind+"__intercept"]=intercept
                kind_scores={}
                for mode,z in states.items():
                    prediction=z@matrix+intercept
                    if kind=="cube24_tensor81":qhat,index=decode_codebook(prediction,train_features,train_q)
                    else:qhat=project_rotation(prediction);index=None
                    angles=cube_symmetry_angle_deg(qhat,val_q)
                    kind_scores[mode]=dict(features=feature_metrics(prediction,val_y,mean),
                        angles=angle_metrics(angles,64))
                    completed[name+"__"+kind+"__"+mode]=prediction
                constant=mean[None].repeat(len(val_y),axis=0)
                if kind=="cube24_tensor81":constant_q,_=decode_codebook(constant,train_features,train_q)
                else:constant_q=project_rotation(constant)
                kind_scores["constant_train_mean"]=dict(features=feature_metrics(constant,val_y,mean),
                    angles=angle_metrics(cube_symmetry_angle_deg(constant_q,val_q),64))
                scores[kind]=kind_scores
            rows.append(dict(id=name,seed=seed,arm=arm,step=600,scores=scores,codebook_references=refs,
                provenance=dict(calibration=str(cal),calibration_sha256=digest(cal/"calibration.json"),
                    state_audit_sha256=digest(audit_dir/"state-audit.json"),state_arrays_sha256=digest(audit_dir/"state-audit.npz"),
                    training_manifest_sha256=key[0],validation_manifest_sha256=key[1],weights_sha256=meta["weights_sha256"])))
    np.savez_compressed(output/"readout-weights.npz",**weights)
    np.savez_compressed(output/"orientation-predictions.npz",**completed)
    result=dict(rows=rows,contracts=contract,ridge=.001,standardization="Training-only per-coordinate mean/std, std floor1e-12",
        calibration_cases=128,validation_cases=64,frames_per_clip=8,scored_future_frames=448,
        source_sha256=digest(__file__),pinned_task_source=str(task_source.__file__),pinned_task_source_sha256=digest(task_source.__file__),
        quaternion_order="wxyz",task_orientation="min geodesic rotation error over24 proper cube symmetries, threshold15degrees",
        hold_gap="Strict success additionally requires position<=.03m and pose for3 consecutive physical steps. Saved latent observations/predictions are spaced5 steps, and the goal-dependent hold counter is not a stored state label. Upstream HDF5 success is not substituted; this analysis does not measure hold or actual planning success.",
        interpretation="Linear accessibility only, not total information. Tensor81 is sign- and cube24-invariant. Its angle decoder uses only training quaternions and has explicit quantization controls. Rotation9 is a stricter labeling-sensitive control, projected to SO(3). No labels enter model/metric/policy training.",
        weights_sha256=digest(output/"readout-weights.npz"),predictions_sha256=digest(output/"orientation-predictions.npz"))
    (output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps([dict(id=r["id"],scores={mode:{"r2":s["features"]["variance_weighted_r2"],
        "mean_deg":s["angles"]["mean_deg"],"within15":s["angles"]["fraction_within15deg"]}
        for mode,s in r["scores"]["cube24_tensor81"].items()})for r in rows],indent=2))


if __name__=="__main__":main()
