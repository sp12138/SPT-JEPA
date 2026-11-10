"""Adopt live public-reference chunks; speed only subsequent chunks' Python GC."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.campaign import command, environment


def owned_process(record):
    try:
        p = Path("/proc")/str(record["pid"])
        start = int((p/"stat").read_text().rsplit(")", 1)[1].split()[19])
        cmd = (p/"cmdline").read_bytes()
        return start == record["start_ticks"] and hashlib.sha256(cmd).hexdigest() == record["cmd_sha256"]
    except (OSError, ValueError, IndexError):
        return False


def chunk(args):
    from experiments.robocasa_compare import evaluate as base
    base.install_public(args.root)
    from app.robo_jepa.mpc import mpc_utils
    from experiments.robocasa_refine.execution import cadenced_cem
    proof = json.loads((args.proofs/(args.variant["reference"]+"-v2.json")).read_text())
    assert proof["passed"] and max(proof["maximum_output_errors"]) == 0.
    mpc_utils.cem = cadenced_cem(mpc_utils.cem)
    original = base.evaluate

    def evaluate_with_execution_record(*positional, **kwargs):
        original(*positional, **kwargs)
        target = positional[0].output/"summary.json"
        result = json.loads(target.read_text())
        result.update(planner_forced_gc="generation1 each batch, full generation2 every8 batches",
            execution_proof_sha256=sha(args.proofs/(args.variant["reference"]+"-v2.json")),
            execution_parity_scope="128 candidates/H5/two refits;1024-candidate repeated-cache memory check",
            CEM_candidate_count_sampling_and_math_unchanged=True)
        save(target, result)

    base.evaluate = evaluate_with_execution_record
    from experiments.robocasa_refine.compare import evaluate_variant
    evaluate_variant(args.root, args.variant, args.output, args.cases, source=False)


def controller(root):
    adoption = json.loads((root/"ADOPTION.json").read_text())
    assert adoption["old_controllers_retired"] and adoption["all_chunks_preserved"]
    data = Path(adoption["data_root"])
    assert (Path("/proc/sys/kernel/random/boot_id").read_text().strip() == adoption["boot_id"])
    plan = json.loads((data/"public-plan-dev6-2c.json").read_text())
    proofs = Path(adoption["proof_root"])
    for size in ("50m", "2b450k", "8b300k"):
        result = json.loads((proofs/(size+"-v2.json")).read_text())
        assert result["passed"] and result["model_unchanged"]
    deadline = adoption["original_deadline_unix"]
    cohort = plan["cases"]

    def one(gpu):
        size, index, parts = plan["assignments"][str(gpu)]
        cases = cohort[index::parts]
        rows = []
        for pose in ("released", "native", *(["rgb"] if size == "50m" else [])):
            variant = dict(id=f"robojepa-{size}-m{2 if pose == 'rgb' else 3}-{pose}", reference=size,
                reference_mode=2 if pose == "rgb" else 3, reference_pose_source="native" if pose == "rgb" else pose)
            out = data/"public-evaluation"/variant["id"]/f"gpu{gpu}"
            live = [x for x in adoption["chunks"] if x["output"] == str(out)]
            if live:
                assert len(live) == 1
                while owned_process(live[0]):
                    assert time.time() < deadline; time.sleep(10)
                assert (out/"DONE.json").exists(), "Adopted chunk exited without its terminal record"
                rows.append(dict(variant=variant, inherited_live_process=live[0]["pid"]))
            elif not (out/"DONE.json").exists():
                argv = [str(data/"environment/bin/python"), "-u", "-m", "experiments.robocasa_refine.adopt_public",
                    "--mode", "chunk", "--root", str(data), "--proofs", str(proofs),
                    "--variant-json", json.dumps(variant), "--case-json", json.dumps(cases), "--output", str(out)]
                runtime_env = environment(data, str(gpu), True)
                runtime_env["PYTHONPATH"] = str(root/"code")+os.pathsep+runtime_env["PYTHONPATH"]
                command(argv, root/f"{variant['id']}-gpu{gpu}.log", runtime_env,
                        root/"code", max(1, deadline-time.time()))
                rows.append(dict(variant=variant, cadenced_gc=True))
            else:
                rows.append(dict(variant=variant, inherited_complete=True))
            terminal = json.loads((out/"DONE.json").read_text())
            assert terminal["disposed"] and len(terminal["rows"]) == len(cases)
        save(root/f"DONE-gpu{gpu}.json", dict(disposed=True, rows=rows))

    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(one, range(8)))
    values, failures = [], []
    for directory in (data/"public-evaluation").glob("*/*"):
        for path in directory.glob("*/summary.json"):
            row = json.loads(path.read_text()); assert row["passed"] and row["model_unchanged"]
            values.append(dict(variant=directory.parent.name, path=str(path), **{k:row[k] for k in
                ("task", "kitchen", "seed", "score", "case_sha256", "final_task_complete", "reference_pose_source")}))
        failures.extend(str(p) for p in directory.glob("*-execution-error.json"))
    save(root/"results/summary.json", dict(disposed=True, endpoints=len(values), expected=7*len(cohort),
        rows=values, technical_failures=failures, baseline_pretraining_not_equal_to_candidates=True,
        old_live_chunks_unchanged=True, future_chunks_cadenced_gc_only=True, new_SGD=0))
    save(data/"PUBLIC_DONE-dev6-2c-adopted.json", dict(disposed=True, collector=str(root/"results/summary.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["controller", "chunk"], required=True)
    p.add_argument("--proofs", type=Path); p.add_argument("--variant-json"); p.add_argument("--case-json")
    p.add_argument("--output", type=Path); a = p.parse_args()
    try:
        if a.mode == "controller": controller(a.root)
        else:
            a.variant = json.loads(a.variant_json); a.cases = json.loads(a.case_json); chunk(a)
    except BaseException as error:
        if a.mode == "controller": save(a.root/"ERROR-controller.json", dict(error=repr(error), method_negative=False))
        raise
