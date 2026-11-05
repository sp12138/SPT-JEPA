"""Held-out factual action recovery,conditioning use,and horizon exposure gap."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.factory import build_model
from as_jepa_tools.trajectory_pairs import TrajectoryPairs
from experiments.metaworld_contact_20261106.diagnose import save
from experiments.metaworld_eval_20261103.evaluate import model_sha


def run(a):
    cfg = json.loads((a.run/'config.json').read_text())
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    model = build_model(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt', map_location='cuda', weights_only=True))
    before = model_sha(model)
    data = TrajectoryPairs(cfg, split='test')
    indices = torch.randperm(data.natural_length, generator=torch.Generator().manual_seed(20261106))[:512].tolist()
    loader = DataLoader(Subset(data, indices), batch_size=32, num_workers=0)
    records = []
    with torch.inference_mode():
        for batch in loader:
            b = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            memory = model.memory(b); true_h = b['horizon'].float().log1p(); predicted_h = model.duration(memory)
            if model.objective == 'metric':
                records.extend(dict(episode=int(ep), horizon=int(h), predicted_log_h=float(p), squared_log_h_error=float((p-t)**2))
                               for ep,h,p,t in zip(b['episode'],b['horizon'],predicted_h,true_h))
                continue
            target = b['action'].float(); mask = b['action_mask'].float()
            noise = torch.randn_like(target)
            # Counterfactual goal is the current observation,not another label.
            changed = dict(b)
            key = 'pixels' if model.visual_source == 'local' else 'features'
            changed[key] = b[key].clone(); changed[key][:, 1] = b[key][:, 0]
            wrong_memory = model.memory(changed)
            for timestep in ([0.] if model.objective == 'mse' else [.1, .5, 1.]):
                t = target.new_full((len(target),), timestep)
                x = torch.zeros_like(target) if model.objective == 'mse' else (1-t[:,None,None])*target+t[:,None,None]*noise
                truth = target if model.objective == 'mse' else noise-target
                fields = dict(true_h=model.action_field(x, t, true_h, memory),
                              predicted_h=model.action_field(x, t, predicted_h, memory),
                              current_as_goal=model.action_field(x, t, true_h, wrong_memory))
                error = {}
                for name, output in fields.items():
                    e = (output.float()-truth).square()
                    # Native action layout is repeated XYZG,not12XYZ+4G.
                    xyz = e.reshape(len(e), model.chunk, 5, 4)[..., :3].mean((2, 3))
                    grip = e.reshape(len(e), model.chunk, 5, 4)[..., 3].mean(2)
                    denominator = mask.sum(-1).clamp_min(1)
                    error[name] = dict(all=(e.mean(-1)*mask).sum(-1)/denominator,
                                       xyz=(xyz*mask).sum(-1)/denominator, gripper=(grip*mask).sum(-1)/denominator)
                for i in range(len(target)):
                    records.append(dict(episode=int(b['episode'][i]), horizon=int(b['horizon'][i]), timestep=timestep,
                                        true_log_h=float(true_h[i]), predicted_log_h=float(predicted_h[i]),
                                        errors={name:{kind:float(value[i]) for kind,value in group.items()} for name,group in error.items()}))
    assert model_sha(model) == before
    save(a.output, dict(passed=True, records=records, step=a.step, model_sha256=before, model_unchanged=True,
         scope='512heldout factual pairs.No optimal-action assertion,SGD,or real control score.Oracle temporaloffset is diagnostic only;online controller estimates it.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',required=True);run(p.parse_args())
