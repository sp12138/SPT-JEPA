"""Separate early ordered pairing from a mere differential reparameterization."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_lewm.factory import build_model
from as_jepa_tools.native_inverse import NativeInversePairs
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);(a.root/'results').mkdir();recipes=[]
    for host,seed in[('dev13',1),('dev15',2)]:
        owners=json.loads((a.wait_parent/f'launch-{host}.json').read_text())['workers']
        for gpu,arm in[(1,'ordered-equivalent'),(5,'distinct-roles')]:
            cfg=json.loads((a.parent/'training'/f'episodes-seed{seed}'/'config.json').read_text());run=a.root/'training'/f'{arm}-seed{seed}';run.mkdir(parents=True);cfg['output']=str(run)
            cfg.update(inverse_goals='phase_mixture');cfg['model'].update(goal_representation='ordered'if gpu==1 else'absolute',role_initialization='zero'if gpu==1 else'distinct',fusion_initialization='directed_equivalent'if gpu==1 else'default')
            save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(all_weights_random=True,one_current_interval_flow_loss=True,original_corpus_only=True,
                hypothesis='OrderedconcatenationhasEXACTsamecapacityasdirecteddifference.Equivalentinitialweightsmakethewholeinitialnetworkfunctionmatch;onlyparameter/optimizationcoordinatesdiffer.'if gpu==1 else'Originalseparatetokenswithdeterministicorthogonalunitroleinitialization;breaksource/goalswapsymmetrywithoutpairfusion.',
                novelty='No noveltyclaimforconcatenation/difference.Thisisamechanismcontrol,notanewalgorithmname.',
                same_CNN_decoder_seed_current_labels_and_sampler=True,limitations='Same40kupdates/batchbutpairfusionhalvesdecoder-memorytokens;not equalFLOPs.'))
            owner=next(x for x in owners if x['gpu']==gpu)
            recipes.append(dict(arm=arm,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=owner['pid'],resource_marker='fifty_current.py',config_sha256=digest(run/'config.json')))
    old=json.loads((a.parent/'training-plan.json').read_text());save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=old['stages'],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        scope='Representationmechanismcontrols;twoindependentseeds,current-onlyoneFMloss,originaldata.Equivalentconcatvsdifferencefunctionclassandinitialfunctionsareexplicitlyaudited.'))


def contract(a):
    torch.set_num_threads(2);plan=json.loads((a.root/'training-plan.json').read_text());reports=[]
    for r in plan['recipes']:
        cfg=json.loads((Path(r['run'])/'config.json').read_text());data=NativeInversePairs(cfg,'test');batch=next(iter(DataLoader(Subset(data,[0,len(data)//2]),batch_size=2)))
        torch.manual_seed(cfg['seed']);model=build_model(cfg).eval();reference=json.loads(json.dumps(cfg));reference['model'].update(goal_representation='directed'if r['arm']=='ordered-equivalent'else'absolute',role_initialization='zero',fusion_initialization='default')
        torch.manual_seed(cfg['seed']);baseline=build_model(reference).eval();state=model.state_dict()
        for k,v in baseline.state_dict().items():
            if k not in['kind','goal_fusion.0.weight']:assert torch.equal(v,state[k]),k
        with torch.no_grad():
            memory=model.memory(batch);ref=baseline.memory(batch);t=torch.ones(2);x=torch.zeros_like(batch['action']);field=model.action_field(x,t,t*0,memory)
            if r['arm']=='ordered-equivalent':
                error=float((memory-ref).abs().max());out_error=float((field-baseline.action_field(x,t,t*0,ref)).abs().max());assert error<1e-4 and out_error<1e-4,(error,out_error)
                assert sum(p.numel()for p in model.parameters())==sum(p.numel()for p in baseline.parameters())
            else:error=out_error=None
            swap=float((field-model.action_field(x,t,t*0,model.memory(dict(pixels=batch['pixels'].flip(1))))).square().mean().sqrt());assert swap>1e-5,swap
            def generator():return torch.Generator().manual_seed(91)
            assert torch.equal(model.propose(batch,generator=generator()),model.propose(dict(pixels=batch['pixels']),generator=generator()))
        loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
        gradients={p:sum(float(v.grad.square().sum())for n,v in model.named_parameters()if n.startswith(p+'.')and v.grad is not None)**.5 for p in model.audit_expected_gradients()};assert all(v>0 for v in gradients.values())
        reports.append(dict(arm=r['arm'],seed=r['seed'],memory_equivalence_max=error,initial_action_field_equivalence_max=out_error,swap_rms=swap,gradients=gradients))
    save(a.root/'initial-contract.json',dict(passed=True,reports=reports,all_weights_random=True,model_updates=0,
        interpretation='Differenceandorderedconcatarenotessentiallydifferentmodelclasses.Initialfunctionparityisolatesoptimizationcoordinates;distinctrolesisolatesinitialroleambiguity.'))
    print('PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','contract'],required=True)
    for k in['root','source','parent','wait-parent']:p.add_argument('--'+k,type=Path)
    a=p.parse_args();prepare(a)if a.mode=='prepare'else contract(a)
