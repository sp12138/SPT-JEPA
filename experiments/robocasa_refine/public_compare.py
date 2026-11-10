"""Frozen reference collection on the common held-out-kitchen cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from experiments.robocasa_compare.acquire import save
from experiments.robocasa_compare.campaign import environment,command
from experiments.robocasa_refine.intervention_phase import ARMS


def fourth_host(root):
    """The extra eight GPUs own the full reference queue, freeing training."""
    plan=json.loads((root/"collection-plan.json").read_text());screen=Path(plan["screen"])
    cohort=json.loads((screen/"CASES_READY.json").read_text())["target"]
    deadline=time.time()+24*3600
    cache=root/"local-simulator-dev6-2c.json"
    while not cache.exists() and time.time()<deadline:
        if (root/"STOP").exists():return
        time.sleep(10)
    assert json.loads(cache.read_text())["passed"]
    assignments={0:("50m",0,2),1:("50m",1,2),2:("2b450k",0,2),3:("2b450k",1,2),
                 **{g:("8b300k",g-4,4) for g in range(4,8)}}
    def preflight(gpu):
        rows=[]
        for case in cohort[gpu::8]:
            label=f"{case['task']}-{case['kitchen']}-{case['seed']}"
            output=root/"newhost-entry"/label
            argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_compare.evaluate",
                "--mode","evaluate","--root",str(root),"--zero","--technical","--output",str(output)]
            for key,value in case.items():argv.extend(["--"+key,str(value)])
            try:
                command(argv,root/("newhost-entry-"+label+".log"),environment(root,str(gpu),True),root/"code",900)
                row=json.loads((output/"summary.json").read_text());assert row["passed"]
                rows.append(dict(case=case,passed=True,initial_rgb_mse=row["initial_rgb_mse"],physical_replay_errors=row["physical_replay_errors"]))
            except Exception as error:rows.append(dict(case=case,passed=False,error=repr(error),method_negative=False))
        return rows
    with ThreadPoolExecutor(max_workers=8) as pool:proofs=sum(pool.map(preflight,range(8)),[])
    valid=[row["case"] for row in proofs if row["passed"]]
    save(root/"NEW_HOST_ENTRY.json",dict(passed=len(valid)>=30,proofs=proofs,valid_cases=valid,
        expected_cases=len(cohort),technical_only=True,new_SGD=0,selected_without_model_scores=True))
    assert len(valid)>=30,"New host failed common-case reset/RGB entry"
    cohort=[case for case in cohort if case in valid]
    save(root/"public-plan-dev6-2c.json",dict(assignments=assignments,cases=cohort,
        all_reference_work_transferred=True,reference_only=True,new_candidate_SGD=0,
        precision="FP16 camera encoder; BF16 transition as Table14",maximum_hours=24))
    def one(gpu):
        size,index,parts=assignments[gpu];subset=cohort[index::parts]
        for pose in ("released","native",*(["rgb"] if size=="50m" else [])):
            variant=dict(id=f"robojepa-{size}-m{2 if pose=='rgb' else 3}-{pose}",reference=size,
                reference_mode=2 if pose=="rgb" else 3,reference_pose_source="native" if pose=="rgb" else pose)
            output=root/"public-evaluation"/variant["id"]/f"gpu{gpu}"
            argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.public_compare",
                "--mode","chunk","--root",str(root),"--variant-json",json.dumps(variant),
                "--case-json",json.dumps(subset),"--output",str(output)]
            command(argv,root/f"public-{variant['id']}-extra-g{gpu}.log",environment(root,str(gpu),True),root/"code",
                    min(22*3600,max(1,deadline-time.time())))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,range(8)))
    save(root/"PUBLIC_DONE-dev6-2c.json",dict(disposed=True,cases=len(cohort),models=["50m","2b450k","8b300k"]))


def worker(root,host):
    plan=json.loads((root/"collection-plan.json").read_text());screen=Path(plan["screen"])
    cohort=json.loads((screen/"CASES_READY.json").read_text())["target"]
    size={"dev13":"50m","dev15":"2b450k","dev6":"8b300k"}[host]
    configurations=[dict(reference=size,reference_mode=3,reference_pose_source=pose,
        id=f"robojepa-{size}-m3-{pose}") for pose in ("released","native")]
    if size=="50m":configurations.append(dict(id="robojepa-50m-m2-rgb",reference=size,reference_mode=2,reference_pose_source="native"))
    save(root/f"public-plan-{host}.json",dict(configurations=configurations,cases=cohort,
        gpu_indices=[4,5,6,7],zero_absolute_pose_for_mode2=True,reference_only=True,
        same_evaluation_not_equal_pretraining=True,maximum_hours=24))
    deadline=time.time()+24*3600
    # The already-owned source queue and new paired learner finish first.
    def phases_done():
        return (screen/f"source-done-{host}.json").exists() and all(
            (root/"training"/f"{ARMS[host]}-seed{s}"/"EVALUATION_COMPLETE.json").exists() or
            (root/"training"/f"{ARMS[host]}-seed{s}"/"ERROR.json").exists() for s in (1,2))
    while not phases_done() and time.time()<deadline:
        if (root/"STOP").exists():return
        time.sleep(10)
    assert phases_done(),"Earlier finite phases did not terminate"
    def one(gpu):
        subset=cohort[(gpu-4)::4]
        for variant in configurations:
            output=root/"public-evaluation"/variant["id"]/f"gpu{gpu}"
            argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.public_compare",
                "--mode","chunk","--root",str(root),"--variant-json",json.dumps(variant),
                "--case-json",json.dumps(subset),"--output",str(output)]
            command(argv,root/f"public-{variant['id']}-g{gpu}.log",environment(root,str(gpu),True),root/"code",
                min(18*3600,max(1,deadline-time.time())))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,(4,5,6,7)))
    save(root/f"PUBLIC_DONE-{host}.json",dict(disposed=True,model_sizes=[size],cases=len(cohort)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["worker","chunk","fourth"],required=True);p.add_argument("--host",choices=ARMS)
    p.add_argument("--variant-json");p.add_argument("--case-json");p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.mode=="worker":worker(a.root,a.host)
    elif a.mode=="fourth":fourth_host(a.root)
    else:
        from experiments.robocasa_refine.compare import evaluate_variant
        evaluate_variant(a.root,json.loads(a.variant_json),a.output,json.loads(a.case_json),source=False)
