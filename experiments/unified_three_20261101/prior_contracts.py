"""Mean-versus-distribution witness plus full trained-model compatibility."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.utils.data import default_collate

from common import read, sha, dataset, write
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


def gaussian_witness():
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))
    sigma = .1; rows = []
    for epsilon in [.01, .001, .0001]:
        width = epsilon / 10
        q_mass = cdf((epsilon + width) / sigma) - cdf((epsilon - width) / sigma)
        p_mass = cdf(width / sigma) - cdf(-width / sigma)
        kl = epsilon ** 2 / (2 * sigma ** 2)
        bound = math.sqrt(kl / 2)
        assert abs(q_mass - p_mass) <= bound and abs(1 - q_mass) > bound
        rows.append(dict(mean_difference=epsilon, prior_sigma=sigma, KL_q_p=kl, Pinsker_TV_bound=bound,
                         posterior_event_probability=q_mass, sampled_prior_event_probability=p_mass,
                         prior_mean_event_value=1.))
    return dict(rows=rows, scope='Bounded indicator around prior mean. Gaussian KL controls sampled-law expectations, '
                'not evaluation at a deterministic mean. Standard probability fact,not a new theorem or task result. '
                'Fresh sigma0 innovations give Cov(z_next|history)=sigma0²I even if the predictor contracts initial noise. '
                'This motivates a sampling control;it does not establish encoder information,planning utility or novelty.')


def main():
    p = argparse.ArgumentParser();p.add_argument('--parent', required=True);p.add_argument('--output', required=True)
    a = p.parse_args();parent = Path(a.parent);torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)
    source = read(parent / 'source-record.json')
    legacy_path = Path(source['source']) / 'as_jepa_lewm/robust.py'
    current_model = Path(__file__).resolve().parents[2] / 'as_jepa_lewm/model.py'
    assert sha(current_model) == sha(Path(source['source']) / 'as_jepa_lewm/model.py')
    spec = importlib.util.spec_from_file_location('legacy_mean_robust', legacy_path)
    legacy = importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    rows = []
    for task in ['pusht', 'tworoom', 'cube']:
        run = parent / f'{task}-tf-seed1';cfg = read(run / 'config.json')
        assert cfg['model_class'] == 'robust' and cfg['model']['forecast_fraction'] == .5
        ds = dataset(cfg, 'train');batch = default_collate([ds[0]]);batch.pop('state', None)
        weights = torch.load(run / 'model-step0000600/weights.pt', map_location='cpu', weights_only=True, mmap=True)
        bootstrap = torch.Generator(device='cpu').manual_seed(1042).get_state()
        reference_loss = reference_metrics = reference_gradients = reference_rng = None
        parameter_count = 0;sampling_loss = None
        for mode in ['legacy', 'default', 'sampled']:
            torch.manual_seed(cfg['seed'])
            model = (legacy.RobustLeWM(**cfg['model']) if mode == 'legacy'
                     else RobustLeWM(**cfg['model'], sample_forecast=mode == 'sampled')).train()
            model.load_state_dict(weights, strict=True)
            assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
            parameter_count = sum(x.numel() for x in model.parameters())
            torch.set_rng_state(bootstrap)
            loss, metrics = model.loss(batch, step=600, ramp_steps=cfg.get('ramp_steps', 0))
            assert torch.isfinite(loss) and all(torch.isfinite(x).all() for x in metrics.values())
            loss.backward();rng = torch.get_rng_state()
            gradients = {k: None if v.grad is None else v.grad.clone() for k, v in model.named_parameters()}
            assert all(v is None or torch.isfinite(v).all() for v in gradients.values())
            if mode == 'legacy':
                reference_loss = loss.detach().clone();reference_metrics = {k:v.detach().clone() for k,v in metrics.items()}
                reference_gradients, reference_rng = gradients, rng
            elif mode == 'default':
                assert torch.equal(loss, reference_loss) and metrics.keys() == reference_metrics.keys()
                assert all(torch.equal(v, reference_metrics[k]) for k, v in metrics.items())
                assert gradients.keys() == reference_gradients.keys()
                assert all((v is None and reference_gradients[k] is None) or
                           (v is not None and reference_gradients[k] is not None and torch.equal(v, reference_gradients[k]))
                           for k, v in gradients.items())
                assert torch.equal(rng, reference_rng)
            else:
                assert torch.equal(rng, reference_rng)
                assert float(metrics['forecast_prior_samples']) == 1
                sampling_loss = float(loss.detach())
            assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
            del model, loss, metrics, gradients;gc.collect()
        rows.append(dict(task=task, seed=1, full_actual_model_kwargs=True,
                         trained_weights_sha256=sha(run / 'model-step0000600/weights.pt'), parameters=parameter_count,
                         default_loss_metrics_all_gradients_and_RNG_exact=True,
                         sampled_loss_finite=sampling_loss, sampled_outer_RNG_after_backward_exact=True,
                         real_train_window=dict(episode=int(batch['episode'][0]), start=int(batch['start'][0])),
                         scope='One real train window/actual trained600 weights;no optimizer or task-performance claim.'))
        del weights, reference_gradients;gc.collect()
    assert not torch.cuda.is_initialized()
    write(a.output, dict(passed=True, CUDA_initialized=False, gaussian_witness=gaussian_witness(),
          full_actual_model_regressions=rows, legacy_source=str(legacy_path), legacy_source_sha256=sha(legacy_path),
          new_source_sha256=sha(Path(__file__).resolve().parents[2] / 'as_jepa_lewm/robust.py'),
          note='Default False is exactly legacy-compatible. Sampled mode uses existing prior sigma0,no new parameter/loss/action target. '
               'Different sampled-mode loss is expected;not a total-ELBO or novelty claim. Native six-run technical/task gates still required.'))
    print(json.dumps(dict(passed=True, CUDA_initialized=False, regressions=rows)))


if __name__ == '__main__':main()
