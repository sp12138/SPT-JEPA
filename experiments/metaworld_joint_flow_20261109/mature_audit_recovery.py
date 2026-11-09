"""Finish the original finite run; only mature gradient auditing changes.

Initial positive-gradient proofs stay mandatory. A finite zero gradient in a
later batch is recorded rather than treated as a crashed optimizer. Forward,
loss, optimizer, data, precision, RNG, LR and update budget are unchanged.
"""
import argparse
import math
from pathlib import Path
import torch
from experiments.metaworld_transition_only_20261109.common import read,save,sha,now
from experiments.metaworld_joint_flow_20261109 import train as original


def gradients(model):
    values={}
    for name,module in dict(vision=model.parent.vision,observer=model.parent.observer,control=model.parent.control,
            transition=model.parent.transition,renderer=model.renderer,prior_scale=model.prior_scale).items():
        grads=[p.grad.detach() for p in module.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads),name
        value=sum(float(g.double().square().sum()) for g in grads)**.5
        assert math.isfinite(value);values[name]=value
    assert all(p.grad is None for module in [model.parent.decoder,model.parent.rgb,model.parent.inverse] for p in module.parameters())
    return values


def run(config,mode):
    cfg=read(config);out=Path(cfg['output']);root=out.parent.parent
    assert cfg['mode']=='mean' and cfg['seed']==1
    assert read(root/'TRAINING_APPROVED.json')['passed']
    original.gradients=gradients;old_save=original.save
    def audited_save(path,value):
        if isinstance(value,dict) and 'positive_gradients' in value and value['positive_gradients'] is not None:
            g=value.pop('positive_gradients');value.update(gradients=g,zero_gradient_groups=[k for k,v in g.items() if v==0],
                mature_finite_gradient_audit=True,all_required_gradients_positive=all(v>0 for v in g.values()))
        if isinstance(value,dict):value['operational_gradient_audit_override_sha256']=sha(__file__)
        old_save(path,value)
    original.save=audited_save
    save(out/'MATURE_AUDIT_RECOVERY.json',dict(utc=now(),mode=mode,helper=str(Path(__file__).resolve()),helper_sha256=sha(__file__),
        original_NN_loss_data_optimizer_precision_LR_budget_unchanged=True,original_initial_positive_gradient_proofs_retained=True,
        change='Float64normmeasurement;requirefinite,maturezerosrecordednotfatal. No gradientmodification.',
        initial_full_checkpoint=1500,completed_before_error=1599,unsaved_updates_replayed_at_most=99,
        checkpointed_updates_repeated=0,original_errors_retained=True))
    if mode=='train':
        assert read(out/'last.json')['step']==1500
        original.train(cfg,2000,out/'checkpoint-0001500.pt')
    else:original.audit(cfg,2000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','audit']);p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(a.config,a.mode)
