"""Actual native state restore, independent-sample and held-out gradient gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.metaworld_contact_20261106.diagnose import save, digest
from experiments.metaworld_eval_20261103.audit import equal


def run(a):
    from as_jepa_lewm.train import TrainModule
    from as_jepa_tools.trajectory_pairs import TrajectoryPairs
    from vforge.ops import OpsConfig, initialize_ops
    from torch.utils.data import DataLoader, Subset
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    # Training uses BF16; this independent FP32 audit must also disable
    # cuDNN TF32,whose convolution rounding depends on batch shape.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(a.run); cfg = json.loads((root/'config.json').read_text())
    checkpoint = torch.load(root/'last.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == a.step
    assert checkpoint['campaign_samples'] == a.step*cfg['batch_size']
    module = TrainModule(cfg).cpu(); module.load_state_dict(checkpoint['state_dict'], strict=True)
    configured = module.configure_optimizers(); optimizer = configured['optimizer']; scheduler = configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer_states'][0]); scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    restore = dict(model=equal(module.state_dict(), checkpoint['state_dict']),
                   optimizer=equal(optimizer.state_dict(), checkpoint['optimizer_states'][0]),
                   scheduler=equal(scheduler.state_dict(), checkpoint['lr_schedulers'][0]))
    assert all(restore.values())
    portable = torch.load(root/f'model-step{a.step:07d}/weights.pt', map_location='cpu', weights_only=True)
    assert equal(module.model.state_dict(), portable)
    assert 'campaign_rng' in checkpoint and len(checkpoint['campaign_rng']['cuda']) == 1
    if cfg['dataset_kind']in['native_inverse_pairs','local_inverse_mix']:
        if cfg['dataset_kind']=='local_inverse_mix':
            from as_jepa_tools.local_inverse_mix import LocalInverseMix as PairData
        else:
            from as_jepa_tools.native_inverse import NativeInversePairs as PairData
        data=PairData(cfg,'test');train=PairData(cfg,'train')
        assert not {r['episode']for r in train.records}&{r['episode']for r in data.records}
        assert cfg['model']['chunk']==1
    else:
        data = TrajectoryPairs(cfg, 'test'); train = TrajectoryPairs(cfg, 'train')
        assert not {r['episode']//5 for r in train.records} & {r['episode']//5 for r in data.records}
    batch = next(iter(DataLoader(Subset(data, [0, data.natural_length//2]), batch_size=2, num_workers=0)))
    model = module.model.cuda().eval()
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    moved = {k:v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(20261106+cfg['seed'])
    loss, metrics = model.loss(moved, a.step)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for prefix, expectation in model.audit_expected_gradients().items():
        grads = [p.grad.detach().float() for n, p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
        norm = sum(float(g.square().sum()) for g in grads)**.5
        if expectation == 'frozen':
            assert not grads and all(not p.requires_grad for n,p in model.named_parameters() if n.startswith(prefix+'.')), prefix
        else:
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and norm > 0, prefix
        gradients[prefix] = norm
    with torch.no_grad():
        memory = model.memory(moved); duration = model.duration(memory)
        independent_memory = []; independent_duration = []
        for i in range(2):
            singleton = {k:v[i:i+1] if torch.is_tensor(v) else v for k, v in moved.items()}
            single = model.memory(singleton); independent_memory.append(single); independent_duration.append(model.duration(single))
        memory_error = float((memory-torch.cat(independent_memory)).abs().max())
        duration_error = float((duration-torch.cat(independent_duration)).abs().max())
        save(root/f'independence-debug-{a.step}.json', dict(memory_max_abs=memory_error, duration_max_abs=duration_error,
             fp32_cudnn_tf32=False, fp32_matmul_tf32=False))
        assert memory_error < 1e-4 and duration_error < 1e-4, (memory_error, duration_error)
        field_error=None
        if model.objective!='metric':
            target=moved['action'].float();t=target.new_tensor([.3,.9]);noise=torch.randn_like(target)
            value=target*(1-t[:,None,None])+noise*t[:,None,None];h=target.new_zeros(2)
            field=model.action_field(value,t,h,memory)
            singles=torch.cat([model.action_field(value[j:j+1],t[j:j+1],h[j:j+1],independent_memory[j])for j in range(2)])
            field_error=float((field-singles).abs().max());assert field_error<1e-4,field_error
    save(a.output, dict(passed=True, step=a.step, actual_cpu_restore=restore,
         portable_exact=True, checkpoint_sha256=digest(root/'last.ckpt'),
         loss=float(loss.detach()), metrics={k:float(v.detach()) for k, v in metrics.items()}, gradients=gradients,
         sample_independence=dict(max_memory_difference=memory_error, max_duration_difference=duration_error,max_action_field_difference=field_error),
         heldout_original_episodes=(batch['episode']//5).tolist(), heldout_rows=batch['rows'].tolist(),
         scope='Actual model/optimizer/scheduler restore and held-out FP32 backward;not task performance.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--step', type=int, required=True)
    p.add_argument('--output', required=True); run(p.parse_args())
