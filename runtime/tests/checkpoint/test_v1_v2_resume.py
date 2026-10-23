import json
import os

import torch

from vforge.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
    load_model_weights,
    save_dcp,
)


def test_v2_manifest_roundtrip_preserves_components():
    manifest = CheckpointManifest.create(state="full", completed_steps=7)
    manifest = CheckpointManifest(
        **{
            **manifest.to_dict(),
            "components": {
                "weights": "weights",
                "optimizer": "optimizer",
                "trainer": "trainer",
            },
            "components_present": {
                "weights": True,
                "optimizer": True,
                "trainer": False,
            },
            "model_fingerprint": {"architecture": "tiny"},
        }
    )

    restored = CheckpointManifest.from_dict(manifest.to_dict())

    assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION == 2
    assert restored.components_present["optimizer"]
    assert restored.model_fingerprint == {"architecture": "tiny"}


def test_reader_accepts_v1_manifest_without_component_fields():
    legacy = {
        "schema_version": 1,
        "state": "full",
        "format": "dcp",
        "completed_steps": 3,
        "world_size": 1,
        "torch_version": "2.test",
        "python_version": "3.test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "execution_policy": {},
        "topology": {},
    }

    manifest = CheckpointManifest.from_dict(legacy)

    assert manifest.schema_version == 1
    assert manifest.components == {}
    assert manifest.components_present == {}


def test_v1_state_directory_supports_model_only_load(tmp_path):
    model = torch.nn.Linear(3, 2)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "00001.dcp"
    save_dcp(
        model,
        None,
        str(checkpoint),
        manifest=CheckpointManifest.create(
            state="weights",
            completed_steps=1,
        ),
    )
    os.replace(checkpoint / "weights", checkpoint / "state")
    manifest_path = checkpoint / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["schema_version"] = 1
    payload.pop("components", None)
    payload.pop("components_present", None)
    payload["fingerprints"] = {}
    manifest_path.write_text(json.dumps(payload))

    for parameter in model.parameters():
        parameter.data.zero_()
    load_model_weights(model, str(checkpoint), format="dcp")
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
