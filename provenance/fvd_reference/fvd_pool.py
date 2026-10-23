from pathlib import Path
import json,numpy as np
from common import S
import sys
sys.path.insert(0,S["runtime_snapshot"])
from vforge.evaluation.runner import sample_artifact_name,_stable_sample_salt
from fvd_coordination import sampling_batch_size

def read_features(out, rows, manifest, expected_hash):
    if manifest.get("num_samples") != len(rows):
        raise ValueError("Incomplete or wrong-size Native eval manifest")
    if manifest.get("fvd_protocol_sha256") != expected_hash:
        raise ValueError("Manifest protocol mismatch")
    expected_batch = sampling_batch_size(out.parent.name)
    if manifest.get("sampling_batch_size", 1) != expected_batch:
        raise ValueError("Manifest sampling batch mismatch")
    features = []
    for row in rows:
        sid = row["sample_id"]
        path = out / sample_artifact_name(sid, suffix=".npy")
        sidecar = json.loads(Path(str(path) + ".json").read_text())
        meta, provenance = sidecar["metadata"], sidecar["provenance"]
        if sidecar["sample_id"] != sid or provenance["sample_id"] != sid:
            raise ValueError("Case identity mismatch")
        if meta["fvd_protocol_sha256"] != expected_hash:
            raise ValueError("Case protocol mismatch")
        if provenance["checkpoint"]["identity_sha256"] != manifest["checkpoint"]["identity_sha256"]:
            raise ValueError("Case checkpoint mismatch")
        if provenance["resolved_config_hash"] != manifest["resolved_config_hash"]:
            raise ValueError("Case config mismatch")
        expected_seed = 20260920 + _stable_sample_salt(sid)
        if not sidecar["seed"] == provenance["seed"] == meta["seed"] == expected_seed:
            raise ValueError("Case seed mismatch with fixed evaluation seed")
        if (meta["cfg_scale"], meta["steps"], meta["history_states"], meta["total_states"], meta["metric_frames"]) != (1, 50, 1, 9, [16, 144]):
            raise ValueError("Sampling settings mismatch")
        if meta.get("sampling_batch_size", 1) != expected_batch:
            raise ValueError("Case sampling batch mismatch")
        feature = np.load(path, allow_pickle=False)
        if feature.shape != (400,) or not np.isfinite(feature).all():
            raise ValueError("Invalid I3D feature")
        features.append(feature)
    return np.stack(features)
