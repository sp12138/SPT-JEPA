"""CPU native restore audit adapted from pinned093 spatial audit for patch models.

The prior auditor is unchanged. Its native checkpoint/optimizer/sampler helpers
are reused, with patch class/source assertions and the additional frozen basis.
No model forward, optimizer step, custom Trainer, or GPU call.
"""
import importlib
import importlib.util
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
RECORD=B/'runs/local_isotropic2000_20261031/helper-source-record.json'
FILES=['experiments/local_carrier_20261031/prepare2000.py',
       'experiments/carrier_progress_20261031/prepare_learning.py',
       'experiments/carrier_progress_20261031/prepare_next.py']
_PINNED=None
_OPS_READY=False


def pinned():
    global _PINNED
    if _PINNED is None:
        import json,hashlib
        record=json.loads(RECORD.read_text());assert record['source_commit']=='0934d3c445e6c754a6ab547d801890c34740b330'
        manifest=json.loads(Path(record['manifest']).read_text())
        for rel in FILES:
            assert hashlib.sha256((Path(record['source'])/rel).read_bytes()).hexdigest()==manifest['files'][rel]
        path=Path(record['source'])/FILES[0]
        spec=importlib.util.spec_from_file_location('patch_pinned_native_resume_primitives',path)
        _PINNED=importlib.util.module_from_spec(spec);spec.loader.exec_module(_PINNED)
    return _PINNED


def audit_state(run,step,parent,initial,training,portable_override=None):
    old=pinned();old.activate_source(training)
    import torch
    from experiments.ideas_validate_20261031.preflight import checkpoint_report
    from as_jepa_lewm.train import TrainModule
    from as_jepa_lewm.factory import gradient_expectations
    from as_jepa_tools.benchmark_data import make_training_dataset
    from vforge.ops import OpsConfig,initialize_ops
    global _OPS_READY
    torch.set_num_threads(2)
    if not _OPS_READY:
        initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)
        _OPS_READY=True
    for name in ('as_jepa_lewm.train','as_jepa_lewm.factory','as_jepa_lewm.patch_carrier',
                 'as_jepa_lewm.spatial_carrier','as_jepa_lewm.robust'):
        assert Path(importlib.import_module(name).__file__).is_relative_to(Path(training['source'])),name
    assert training['source_commit']=='1018ebbe9beb3e5f9b7c91018f03ff8a4f17884d'
    run,parent=Path(run),Path(parent);cfg=old.read(run/'config.json')
    assert old.semantic(cfg)==old.semantic(old.read(parent/'config.json'))
    assert cfg['model_class']=='patch_carrier' and cfg['model']['patch_basis']in['identity','orthogonal']
    state=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    portable_path=Path(portable_override) if portable_override else run/f'model-step{step:07d}/weights.pt'
    portable=torch.load(portable_path,map_location='cpu',weights_only=True)
    report=checkpoint_report(state,portable,cfg,step);checks=report['checks']
    checks['complete_model_keys']=set(state['state_dict'])=={'model.'+k for k in portable}
    checks['saved_training_semantics']=old.checkpoint_semantic(state['hyper_parameters'])==old.semantic(cfg)
    checks['scheduler_step']=all(s['last_epoch']==step for s in state['lr_schedulers'])
    checks['optimizer_scheduler_lr_equal']=([g['lr']for g in state['optimizer_states'][0]['param_groups']]==state['lr_schedulers'][0]['_last_lr'])
    module=TrainModule(cfg);module.load_state_dict(state['state_dict'],strict=True);checks['native_model_strict_load']=True
    native=module.configure_optimizers();optimizer=native['optimizer'];scheduler=native['lr_scheduler']['scheduler']
    named={id(p):n for n,p in module.model.named_parameters()}
    current=[[named[id(p)]for p in g['params']]for g in optimizer.param_groups]
    trainable={n for n,p in module.model.named_parameters()if p.requires_grad}
    ordered=[n.removeprefix('model.')for n in state['state_dict']if n.removeprefix('model.')in trainable]
    checks['optimizer_parameter_order']=current==[[n for n in ordered if not n.startswith('renderer.')],[n for n in ordered if n.startswith('renderer.')]]
    optimizer.load_state_dict(state['optimizer_states'][0]);scheduler.load_state_dict(state['lr_schedulers'][0])
    checks['native_optimizer_roundtrip']=old.nested_equal(state['optimizer_states'][0],optimizer.state_dict())
    checks['native_scheduler_roundtrip']=old.nested_equal(state['lr_schedulers'][0],scheduler.state_dict())
    original=torch.load(initial,map_location='cpu',weights_only=True)
    frozen=[n for n,p in module.model.named_parameters()if not p.requires_grad]
    buffers=['spatial_projection','spatial_mean','spatial_scale']
    if cfg['model']['patch_basis']=='orthogonal':buffers+=['predictor.coordinate_basis']
    registered=dict(module.model.named_buffers())
    assert all(n in registered and n in portable and n in original for n in buffers)
    frozen+=buffers;assert len(frozen)==len(set(frozen))==210+(cfg['model']['patch_basis']=='orthogonal')
    different=[n for n in frozen if n not in original or n not in portable or not torch.equal(original[n],portable[n])]
    checks['all_frozen_parameters_and_measurement_buffers_unchanged']=not different
    sequence=old.resume_sequence(make_training_dataset(cfg),state['campaign_samples'],cfg['seed'],cfg['batch_size'])
    checks['sampler_cursor_sequence']=sequence['passed']
    report.update(run=str(run),parent=str(parent),initial_weights=str(initial),
        checkpoint_sha256=old.digest(run/'last.ckpt'),portable_path=str(portable_path),portable_sha256=old.digest(portable_path),
        config_sha256=old.digest(run/'config.json'),source_commit=training['source_commit'],
        optimizer_parameter_tensors=[len(g)for g in current],gradient_expectations=gradient_expectations(module.model,cfg),
        frozen_tensors=dict(equal=not different,compared_tensors=len(frozen),names=frozen,different=different),
        sampler=sequence,rng_component_sha256={k:old.rng_digest(v)for k,v in state['campaign_rng'].items()},
        passed=all(checks.values()),scope='CPU-only actual native model/AdamW/scheduler strict restore and frozen tensor/full-state audit. '
            'Saved RNG bytes are preserved, not restored here. Stateless sampler indices are compared with start0 skipped to cursor. '
            'No model forward/training/GPU; no bitwise resumed-versus-uninterrupted GPU trajectory claim.')
    assert report['passed'],report
    assert not torch.cuda.is_initialized()
    return report
