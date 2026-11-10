"""Paired locality and analytic-noise tests, fixed1000-update budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import torch

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare.campaign import environment,command
from experiments.robocasa_compare.train import build,make_optimizer,associations,state_hash,save_full_state
from experiments.robocasa_refine import intervention_phase as phase
from experiments.robocasa_refine.intervene import Interventions

ARMS={"dev13":"spatial-global","dev15":"spatial-local","dev6":"analytic"}
phase.ARMS=ARMS


def initialize(root):
    assert not (root/"INITIALIZED.json").exists()
    assert json.loads((root/"LOCALITY_APPROVED.json").read_text())["passed"]
    assert json.loads((root/"ANALYTIC_APPROVED.json").read_text())["passed"]
    assert json.loads((root/"SIM_LOCALITY_APPROVED.json").read_text())["passed"]
    source=Path(json.loads((root/"PARENT.json").read_text())["source"])
    plan=json.loads((source/"collection-plan.json").read_text())
    plan.update(corpus=str(source),intended_training_arms=ARMS,new_collection=False,
        maximum_new_sgd_per_arm_parent=1000,automatic_long_training=False,
        initial_only_controlled_mean_rollout=True)
    save(root/"collection-plan.json",plan)
    short=Interventions(root/"interventions",8,1,semantics="issued")
    long=Interventions(root/"interventions",16,1,semantics="issued")
    for i in range(16):
        a,b=short[i],long[i]
        assert torch.equal(a["pixels"],b["pixels"][:,:8])
        assert torch.equal(a["action"][:7],b["action"][:7]) and not a["action"][-1].any()
    save(root/"DATA_APPROVED.json",dict(passed=True,paired_windows=16,same_frame_prefixes=True,
        unchanged_corpus_sha256=sha(root/"interventions/dataset.json"),unknown_terminal_input_zero=True))
    records=[]
    for host,arm in ARMS.items():
        for seed in (1,2):
            family="long-rollout" if arm=="analytic" else "spatial"
            parent=Path(plan["parent"])/"training"/f"{family}-seed{seed}"
            old=torch.load(parent/"last.ckpt",map_location="cpu",weights_only=False);assert old["step"]==5000
            cfg=json.loads((parent/"config.json").read_text())
            cfg.update(arm=arm,host=host,parent=str(parent),data=str(root/"interventions"),
                max_steps=6000,phase_start=5000,stages=[5000,5002,6000],input_semantics="issued")
            cfg["model"].update(flow_force_t_one=False,local_encoder=arm=="spatial-local",analytic_prior=arm=="analytic")
            run=root/"training"/f"{arm}-seed{seed}";run.mkdir(parents=True,exist_ok=False);save(run/"config.json",cfg)
            model=build(cfg["model"],root/"code");model.load_state_dict(old["model"],strict=True)
            assert state_hash(model.state_dict())==old["nn_sha256"]
            opt,scheduler=make_optimizer(model,6000,5000)
            assert associations(model,opt)==old["parameter_associations"]
            opt.load_state_dict(old["optimizer"])
            fresh=[v*f(5000) for v,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
            sched=copy.deepcopy(old["scheduler"]);sched["_last_lr"]=fresh;scheduler.load_state_dict(sched)
            for group,lr in zip(opt.param_groups,fresh):group["lr"]=lr
            state=dict(old,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),config_sha256=sha(run/"config.json"))
            save_full_state(run,state)
            save(run/"ancestry.json",dict(passed=True,parent=str(parent),parent_checkpoint_sha256=sha(parent/"last.ckpt"),
                all_neural_tensors_equal=len(old["model"]),optimizer_RNG_cursor_preserved=True,
                original_scratch_unsaved_update_replay=1,new_updates=1000,independent_pretraining=False,
                observed_images_per_update=cfg["global_batch"]*cfg["model"]["window"]*2,
                common_schedule="1000-update restart/10warmup",new_neural_parameters=0))
            records.append(dict(run=run.name,host=host,parent=str(parent),source_nn=old["nn_sha256"]))
            del state,old,model,opt,scheduler
    save(root/"INITIALIZED.json",dict(passed=True,records=records,new_SGD=0,
        local_vs_global_exact_parameter_and_data_pair=True,analytic_vs_prior_issued_control_in_previous_root=True))


def entry(root,run,gpu):
    corpus=Path(json.loads((root/"collection-plan.json").read_text())["corpus"])
    case=json.loads((root/"VALIDATION_CASES_READY.json").read_text())["selected"][0]
    out=root/"entry"/run.name
    argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.validation",
        "--root",str(corpus),"--mode","evaluate","--run",str(run),"--step","5002","--technical",
        "--history-controls","issued","--output",str(out)]
    for key,value in case.items():argv.extend(["--"+key,str(value)])
    command(argv,run/"simulator-entry.log",environment(root,str(gpu),True),root/"code",900)
    assert json.loads((out/"summary.json").read_text())["passed"]


def evaluate_seed(root,run,gpus):
    corpus=Path(json.loads((root/"collection-plan.json").read_text())["corpus"])
    cohort=json.loads((root/"VALIDATION_CASES_READY.json").read_text())["selected"]
    metrics=("uniform","support") if run.name.startswith("spatial") else ("uniform",)
    def one(item):
        i,case,metric=item;label=f"{case['task']}-{case['kitchen']}-{case['seed']}"
        out=root/"source-evaluation"/run.name/metric/label
        argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.validation",
            "--root",str(corpus),"--mode","evaluate","--run",str(run),"--step","6000",
            "--history-controls","issued","--goal-metric",metric,"--output",str(out)]
        for key,value in case.items():argv.extend(["--"+key,str(value)])
        command(argv,run/("eval-"+metric+"-"+label+".log"),environment(root,str(gpus[i]),True),root/"code",7200)
        assert json.loads((out/"summary.json").read_text())["passed"]
    for metric in metrics:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,[(i,c,metric) for i,c in enumerate(cohort)]))
    save(run/"EVALUATION_COMPLETE.json",dict(passed=True,episodes=len(cohort)*len(metrics),goal_metrics=metrics,source_only=True))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["initialize","worker"],required=True);p.add_argument("--host",choices=ARMS)
    a=p.parse_args();phase.evaluate_seed=evaluate_seed
    if a.mode=="initialize":initialize(a.root)
    else:phase.worker(a.root,a.host,entry_check=entry)
