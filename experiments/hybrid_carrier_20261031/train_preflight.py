"""Read-only observations around the native TWO-step stage and clipping hook."""
import json
import sys
from pathlib import Path

import torch


def branch_statistics(embedding):
    result = {}
    for name, values in [('fixed',embedding[..., :64]), ('query',embedding[..., 64:])]:
        x = values.detach().double().flatten(0,-2)
        mean = x.mean(0)
        result[name] = dict(mean=mean.cpu().tolist(), mean_rms=float(mean.square().mean().sqrt()),
            centered_rms=float((x-mean).square().mean().sqrt()), total_rms=float(x.square().mean().sqrt()),
            finite=bool(torch.isfinite(x).all()))
    return result


def gradient_norms(model):
    result = {}
    all_squared = []
    for prefix in ('encoder','pool','predictor','action_encoder','renderer'):
        values = [p.grad.detach() for n,p in model.named_parameters()
                  if n.startswith(prefix+'.') and p.grad is not None]
        squared = [g.double().square().sum() for g in values]
        result[prefix] = dict(gradient_tensors=len(values),
            norm=float(torch.stack(squared).sum().sqrt()) if squared else 0.,
            finite=all(bool(torch.isfinite(g).all()) for g in values))
        all_squared.extend(squared)
    result['total'] = dict(norm=float(torch.stack(all_squared).sum().sqrt()) if all_squared else 0.,
                          finite=all(x['finite'] for x in result.values()))
    return result


def observe_clip(original, module, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None, sink=None):
    before = gradient_norms(module.model)
    result = original(module, optimizer, gradient_clip_val, gradient_clip_algorithm)
    after = gradient_norms(module.model)
    if sink is not None:
        sink(dict(kind='native_gradient_clipping', step=int(module.global_step),
            configured_threshold=gradient_clip_val, configured_algorithm=gradient_clip_algorithm,
            before=before, after=after, original_calls=1))
    return result


def main():
    argv = sys.argv[1:]
    assert '--resume' not in argv and int(argv[argv.index('--stop-step')+1]) == 2
    cfg = json.loads(Path(argv[argv.index('--config')+1]).read_text())
    assert cfg['model_class'] == 'hybrid_carrier' and cfg.get('freeze_encoder_steps',0) == 0
    assert cfg.get('calibrate_state_std') is None
    output = Path(cfg['output'])/'preflight-observations.jsonl'
    assert not output.exists()
    import as_jepa_lewm.train as entry
    from as_jepa_tools import train_stage
    old_build = entry.build_model
    own_clip = 'configure_gradient_clipping' in entry.TrainModule.__dict__
    old_clip = entry.TrainModule.configure_gradient_clipping
    counter = {'encode':0,'clip':0}

    def sink(value):
        with output.open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')

    def build(config):
        model = old_build(config)
        old_encode = model.encode
        def encode(info):
            result = old_encode(info)
            counter['encode'] += 1
            assert counter['encode'] <= 2, 'No expanded trace/training budget'
            sink(dict(kind='encoded_branches', ordinal=counter['encode'],
                input_frames=int(result['emb'].shape[0]*result['emb'].shape[1]),
                statistics=branch_statistics(result['emb'])))
            return result
        model.encode = encode
        return model

    def clipping(module, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        counter['clip'] += 1
        assert counter['clip'] <= 2
        return observe_clip(old_clip,module,optimizer,gradient_clip_val,gradient_clip_algorithm,sink)

    try:
        entry.build_model = build
        entry.TrainModule.configure_gradient_clipping = clipping
        train_stage.main()
        assert counter == {'encode':2,'clip':2}, counter
    finally:
        entry.build_model = old_build
        if own_clip: entry.TrainModule.configure_gradient_clipping = old_clip
        else: del entry.TrainModule.configure_gradient_clipping


if __name__ == '__main__': main()
