import json
import tempfile
import unittest
from pathlib import Path
import torch
from vforge.ops import initialize_ops,OpsConfig
from vforge.models.as_jepa.model import ASJepaSPT
from as_jepa_tools.evaluate import load_model


class CheckpointFiniteContracts(unittest.TestCase):
    def test_finite_checkpoint_loads_and_one_nonfinite_weight_is_rejected(self):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
        config=dict(_target_="vforge.models.as_jepa.model.ASJepaSPT",width=96,encoder_depth=2,prior_depth=2,renderer_depth=2)
        model=ASJepaSPT(**{k:v for k,v in config.items() if k!="_target_"})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"config.json").write_text(json.dumps(config))
            weights=model.state_dict();torch.save(weights,root/"weights.pt")
            load_model(root,device="cpu")
            weights["pool.head.bias"][0]=float("nan");torch.save(weights,root/"weights.pt")
            with self.assertRaisesRegex(FloatingPointError,"pool.head.bias"):
                load_model(root,device="cpu")


if __name__=="__main__":
    unittest.main()
