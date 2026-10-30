"""Complete-only CPU collection of the 34-job carrier diagnostic queue.

No polling, model loading, GPU work, fitting, or evaluation reruns. Large
candidate arrays remain on EFS; their hashes and descriptive summaries travel.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from experiments.ideas_validate_20261031.collect import paired_outcomes

METRICS=("identity","observation","increment")
METRIC_PAIRS=(("observation","identity"),("increment","identity"),("increment","observation"))


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb")as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def option(job,name):
    command=job["command"];flag="--"+name
    if flag not in command:raise AssertionError(f"Missing {flag}: {job['id']}")
    return command[command.index(flag)+1]


def summary(values):
    values=np.asarray(values,dtype=float);finite=values[np.isfinite(values)]
    return dict(count=len(values),defined=int(len(finite)),undefined=int(len(values)-len(finite)),
        mean=float(finite.mean())if len(finite)else None,
        median=float(np.median(finite))if len(finite)else None,
        p05=float(np.quantile(finite,.05))if len(finite)else None,
        p95=float(np.quantile(finite,.95))if len(finite)else None)


def planning_signature(report):
    # Normalization differs for released LeWM and is retained separately.
    return {key:report[key]for key in ["manifest_sha256","pairs","solver","plan","protocol","split"]}


def validate_outcomes(report):
    outcomes=report["metrics"]["episode_successes"]
    assert len(outcomes)==len(report["pairs"])==50
    assert all(v in [0,1,False,True]for v in outcomes)
    assert abs(100*sum(outcomes)/50-report["metrics"]["success_rate"])<1e-8
    assert report["checkpoint_audit"]["strict_equal"] is True
    return [bool(v)for v in outcomes]


def summarize_banks(output,report):
    diagnostic=report["metric_diagnostic"];audit=diagnostic["candidate_audit"]
    path=Path(output)
    assert audit["metric_order"]==list(METRICS)and audit["extra_model_rollouts"]==0
    assert audit["manual_identity_max_abs_error"]==0
    assert digest(path/"metric-costs.npz")==audit["costs_sha256"]
    assert digest(path/"metric-candidate-banks.json")==audit["candidate_banks_sha256"]
    banks=read(path/"metric-candidate-banks.json")
    with np.load(path/"metric-costs.npz")as f:
        costs=f["costs"];orders=f["stable_orders"];offsets=f["call_offsets"]
    assert costs.ndim==3 and costs.shape==orders.shape and costs.shape[0]==3
    assert np.isfinite(costs).all()and len(banks)==audit["calls"]==len(offsets)-1
    assert offsets[0]==0 and offsets[-1]==costs.shape[1]==audit["candidate_batches"]
    assert np.all(np.diff(offsets)>0)
    np.testing.assert_array_equal(np.argsort(costs,axis=-1,kind="stable"),orders)
    for i,bank in enumerate(banks):
        lo,hi=map(int,offsets[i:i+2]);assert bank["call"]==i and bank["shape"]==[3,hi-lo,costs.shape[-1]]
        for k,name in enumerate(METRICS):assert bank["selected_indices"][name]==costs[k,lo:hi].argmin(-1).tolist()
    solver=report["solver"];samples,steps,topk=[int(solver[k])for k in ["num_samples","n_steps","topk"]]
    assert costs.shape[-1]==samples and int(solver["batch_size"])==1
    assert 0<topk<=samples and len(banks)>=50*steps
    ranks=rankdata(costs,axis=-1,method="average")
    centered=ranks-ranks.mean(-1,keepdims=True)
    rank_norm=np.square(centered).sum(-1)**.5
    pairs={}
    for candidate,reference in METRIC_PAIRS:
        a,b=METRICS.index(candidate),METRICS.index(reference)
        denominator=rank_norm[a]*rank_norm[b]
        rho=np.full(len(denominator),np.nan)
        valid=denominator>0
        rho[valid]=(centered[a]*centered[b]).sum(-1)[valid]/denominator[valid]
        elite_a,elite_b=orders[a,:,:topk],orders[b,:,:topk]
        overlap=(elite_a[:,:,None]==elite_b[:,None,:]).any(-1).mean(-1)
        pairs[candidate+"__vs__"+reference]=dict(spearman=summary(rho),
            stable_top1_disagreement_fraction=float((orders[a,:,0]!=orders[b,:,0]).mean()),
            stable_full_order_agreement_fraction=float((orders[a]==orders[b]).all(-1).mean()),
            stable_topk_overlap_fraction=summary(overlap),topk=topk,
            normalized_average_rank_difference=summary(np.abs(ranks[a]-ranks[b]).mean(-1)/max(1,samples-1)))
    tie_rates={}
    for k,name in enumerate(METRICS):
        sorted_cost=np.take_along_axis(costs[k],orders[k],axis=-1)
        tie_rates[name]=dict(constant_bank_fraction=float((rank_norm[k]==0).mean()),
            topk_boundary_tie_fraction=float((sorted_cost[:,topk-1]==sorted_cost[:,topk]).mean())if topk<samples else 0.)
    # Pinned CEM processes every initial environment, all n_steps, in case order.
    # Later solves may have fewer active environments and are never aligned here.
    first_calls=np.arange(50)*steps
    first_rows=[banks[int(i)]for i in first_calls]
    assert all(row["shape"][1]==1 for row in first_rows)
    first_costs=np.stack([costs[:,int(offsets[i]),:]for i in first_calls],axis=1)
    result=dict(calls=len(banks),candidate_batches=costs.shape[1],samples_per_bank=samples,
        metric_pairs=pairs,ties=tie_rates,first_solve_first_iteration_call_indices=first_calls.tolist(),
        source_costs=str(path/"metric-costs.npz"),costs_sha256=audit["costs_sha256"],
        source_banks=str(path/"metric-candidate-banks.json"),candidate_banks_sha256=audit["candidate_banks_sha256"],
        scope="Within each bank the three costs see identical predictions. Aggregate banks are dependent and conditioned on this selected metric's search path. Stable diagnostic orders do not replace native torch.topk tie handling; no bank-bootstrap confidence intervals.")
    return result,dict(rows=first_rows,costs=first_costs)


def compare_first_banks(group,first_banks):
    ordered=[group[name]for name in METRICS]
    first=[first_banks[r["id"]]for r in ordered]
    keys=["action_candidates_sha256","goal_emb_sha256","pixels_sha256","action_history_sha256","endpoint_difference_sha256"]
    agreements={};mismatches={}
    for key in keys:
        present=all(key in bank for data in first for bank in data["rows"])
        if not present:
            agreements[key]=None;mismatches[key]=None
            continue
        different=[i for i in range(50)if len({data["rows"][i][key]for data in first})!=1]
        agreements[key]=50-len(different);mismatches[key]=different
    required_inputs=["action_candidates_sha256","goal_emb_sha256","pixels_sha256"]
    complete_input_signature=all(agreements[key]is not None for key in required_inputs)
    all_inputs=[key for key in keys if key!="endpoint_difference_sha256"and agreements[key]is not None]
    matched_inputs=[];matched_outputs=[];max_errors=[];exact_orders=[]
    for case in range(50):
        same_input=complete_input_signature and all(len({data["rows"][case][key]for data in first})==1 for key in all_inputs)
        same_endpoint=len({data["rows"][case]["endpoint_difference_sha256"]for data in first})==1
        if same_input:matched_inputs.append(case)
        if same_input and same_endpoint:
            matched_outputs.append(case)
            max_errors.append(max(float(np.max(np.abs(data["costs"][:,case]-first[0]["costs"][:,case])))for data in first[1:]))
            exact_orders.append(all(np.array_equal(np.argsort(data["costs"][:,case],axis=-1,kind="stable"),
                np.argsort(first[0]["costs"][:,case],axis=-1,kind="stable"))for data in first[1:]))
    return dict(checkpoint=ordered[0]["checkpoint"],case_set=ordered[0]["case_set"],
        job_ids=[r["id"]for r in ordered],case_count=50,agreement_counts=agreements,mismatch_case_indices=mismatches,
        complete_input_signature=complete_input_signature,
        matching_input_case_indices=matched_inputs,matching_input_and_endpoint_case_indices=matched_outputs,
        matched_output_cost_max_abs_errors=max_errors,
        matched_output_all_three_stable_orders_equal=exact_orders,
        scope="Only first planning solve, first CEM iteration for each case. Later search populations and executed trajectories are deliberately not treated as paired banks.")


def collect(root,bootstrap):
    root=Path(root);plan=read(root/"plan.json");protocol=read(root/"protocol.json")
    jobs=plan["jobs"]
    assert len(jobs)==len({j["id"]for j in jobs})==protocol["jobs"]==34
    assert (protocol["fresh_tworoom_jobs"],protocol["primary_cube_metric_jobs"],
        protocol["fresh_cube_baselines"],protocol["fresh_cube_metric_jobs"])==(7,12,3,12)
    expected_counts={("tworoom","fresh-validation-planning"):7,("cube","fresh-validation-planning"):3,
                     ("cube","metric-planning"):12,("cube","fresh-metric-planning"):12}
    assert dict(Counter((j["task"],j["phase"])for j in jobs))==expected_counts
    exits=Path(protocol["authoritative_exits"]);assert exits.resolve()==(root/"exits").resolve()
    # Finish check precedes all evaluation-output reads; no partial collection.
    completed=[]
    for job in jobs:
        status=read(exits/(job["id"]+".json"))
        assert status["returncode"]==0,(job["id"],status)
        assert job["source"]["source_commit"]==protocol["source_commit"]
        completed.append(dict(id=job["id"],returncode=0,exit_sha256=digest(exits/(job["id"]+".json"))))
    for path,sha in protocol["source_inputs"].items():assert digest(path)==sha,path
    calibrations={}
    for item in protocol["calibration_review"]:
        assert digest(item["path"])==item["sha256"]
        calibration=read(item["path"])
        assert calibration["step"]==600 and calibration["training_clips"]==128
        assert calibration["source"]["source_commit"]==protocol["source_commit"]
        assert digest(Path(item["path"]).parent/"matrices.npz")==calibration["matrices_sha256"]
        calibrations[str(Path(item["path"]).parent.resolve())]=calibration
    assert len(calibrations)==4
    assert len({c["train_manifest_sha256"]for c in calibrations.values()})==1
    rows=[];case_sets={};first_banks={};hash_cache={};identity_replay=[]
    def cached_hash(path):
        path=str(Path(path).resolve())
        if path not in hash_cache:hash_cache[path]=digest(path)
        return hash_cache[path]
    for job in jobs:
        output=Path(job["output"]);report=read(output/"metrics.json")
        outcomes=validate_outcomes(report)
        manifest_path=Path(option(job,"manifest"));manifest=read(manifest_path)
        manifest_sha=digest(manifest_path)
        assert manifest_sha==report["manifest_sha256"]
        assert report["pairs"]==manifest["pairs"]and report["protocol"]==manifest["protocol"]
        assert report["split"]==manifest["campaign_split"]=="validation"
        assert manifest["task"]==job["task"]and len({p["episode_id"]for p in manifest["pairs"]})==50
        case_set="primary"if job["phase"]=="metric-planning"else"fresh"
        key=job["task"]+"-"+case_set
        signature=planning_signature(report)
        if key in case_sets:assert canonical(signature)==canonical(case_sets[key]["planning_signature"]),job["id"]
        else:
            case_sets[key]=dict(manifest=str(manifest_path),manifest_sha256=manifest_sha,
                policy_seed=manifest["policy_seed"],planning_signature=signature,
                fresh_selection=manifest.get("fresh_selection"),scope=manifest.get("campaign_scope"))
        if case_set=="fresh":
            expected_sha=protocol["fresh_manifest_sha256"]if job["task"]=="tworoom"else protocol["fresh_cube_manifest_sha256"]
            assert manifest_sha==expected_sha
            primary_path=Path(manifest["fresh_selection"]["primary_manifest"])
            assert digest(primary_path)==manifest["fresh_selection"]["primary_manifest_sha256"]
            primary=read(primary_path)
            pair=lambda p:(p["episode_id"],p["start_step"],p["goal_step"])
            assert not {pair(p)for p in primary["pairs"]}.intersection({pair(p)for p in manifest["pairs"]})
            assert primary["policy_seed"]!=manifest["policy_seed"]
            assert primary["protocol"]==manifest["protocol"]
        checkpoint=Path(option(job,"checkpoint"));cfg=read(checkpoint/"config.json")
        assert checkpoint.resolve()==Path(report["checkpoint"]).resolve()
        weights_hash=cached_hash(checkpoint/"weights.pt");cfg_hash=cached_hash(checkpoint/"config.json")
        family=job.get("family","carrier");arm=job.get("arm",cfg.get("carrier_chart"))
        seed=job["training_seed"];metric=job.get("metric","identity")
        if "checkpoint_weights_sha256"in job:
            assert weights_hash==job["checkpoint_weights_sha256"]and cfg_hash==job["checkpoint_config_sha256"]
        data=Path(option(job,"data"));card=read(data/"dataset_card.json")
        if "metadata_card_sha256"in manifest:assert digest(data/"dataset_card.json")==manifest["metadata_card_sha256"]
        own=cfg.get("_target_","").startswith("as_jepa_lewm.")
        assert own==(family!="released-lewm")
        normalization="train episodes"if own else"released reference full-data normalization"
        assert report["normalization"]==normalization
        if own:
            stage=read(checkpoint.parent/"stage-complete.json");assert stage["stop_step"]==600
            size=read(checkpoint.parent/"model_size.json")
            assert size["total"]==report["checkpoint_audit"]["parameters"]
            run_cfg=read(checkpoint.parent/"config.json")
            assert stage["config_sha256"]==digest(checkpoint.parent/"config.json")
            carrier_calibration=run_cfg.get("carrier_calibration")
        else:size=dict(total=report["checkpoint_audit"]["parameters"]);carrier_calibration=None
        row=dict(id=job["id"],task=job["task"],case_set=case_set,family=family,arm=arm,seed=seed,
            metric=metric,checkpoint=str(checkpoint),checkpoint_weights_sha256=weights_hash,
            checkpoint_config_sha256=cfg_hash,success_rate=report["metrics"]["success_rate"],
            episode_successes=outcomes,manifest_sha256=manifest_sha,policy_seed=manifest["policy_seed"],
            metrics=report["metrics"],seconds=report["seconds"],model_size=size,
            normalization=normalization,carrier_initial_statistics=carrier_calibration,
            model_target=cfg["_target_"],checkpoint_audit=report["checkpoint_audit"],
            metric_diagnostic=report.get("metric_diagnostic"),output=str(output),
            metrics_sha256=digest(output/"metrics.json"),
            comparison_scope="Released pretrained reference has unmatched exposure/capacity and normalization."if not own else
                "Native600 model; cross-family carrier/TF capacity, frozen-parameter status and calibration access differ.")
        if "metric"in job:
            diagnostic=report["metric_diagnostic"]
            assert diagnostic["selected"]==metric
            assert diagnostic["native_objective"]==dict(class_name="GoalMSE",pred_key="predicted_emb",goal_key="goal_emb",reduction="sum")
            calpath=str(Path(option(job,"calibration")).resolve());cal=calibrations[calpath]
            assert cal["weights_sha256"]==weights_hash
            assert digest(Path(calpath)/"calibration.json")==diagnostic["calibration_sha256"]
            assert diagnostic["matrices_sha256"]==cal["matrices_sha256"]
            assert diagnostic["train_manifest_sha256"]==cal["train_manifest_sha256"]
            row["candidate_banks"],first_banks[job["id"]]=summarize_banks(output,report)
            if case_set=="primary"and metric=="identity":
                oldpath=checkpoint.parent.parent/("planning-"+checkpoint.parent.name)/"metrics.json"
                old=read(oldpath);old_outcomes=validate_outcomes(old)
                assert canonical(planning_signature(old))==canonical(signature)
                assert old["normalization"]==normalization
                changed=[i for i,(a,b)in enumerate(zip(outcomes,old_outcomes))if a!=b]
                assert not changed,f"Native identity replay differs on cases {changed}: {job['id']}"
                identity_replay.append(dict(id=job["id"],original=str(oldpath),original_sha256=digest(oldpath),
                    all50_outcomes_equal=True,normalization_equal=old["normalization"]==normalization))
        rows.append(row)
    assert len(identity_replay)==4
    expected=set()
    for case_set in ["primary","fresh"]:
        for seed in [1,2]:
            for arm in ["identity","bounded"]:
                for metric in METRICS:expected.add(("cube",case_set,"carrier",arm,seed,metric))
    for task in ["tworoom","cube"]:
        for seed in [1,2]:expected.add((task,"fresh","temporal-forecast",None,seed,"identity"))
        expected.add((task,"fresh","released-lewm",None,None,"identity"))
    for seed in [1,2]:
        for arm in ["identity","bounded"]:expected.add(("tworoom","fresh","carrier",arm,seed,"identity"))
    assert {(r["task"],r["case_set"],r["family"],r["arm"],r["seed"],r["metric"])for r in rows}==expected
    metric_groups=defaultdict(dict)
    for row in rows:
        if row["metric_diagnostic"]:metric_groups[row["case_set"],row["checkpoint"]][row["metric"]]=row
    paired=[];first_checks=[]
    for (case_set,checkpoint),group in metric_groups.items():
        assert set(group)==set(METRICS)
        assert len({r["checkpoint_weights_sha256"]for r in group.values()})==1
        assert len({r["manifest_sha256"]for r in group.values()})==1
        assert len({r["metric_diagnostic"]["matrices_sha256"]for r in group.values()})==1
        for candidate,reference in METRIC_PAIRS:
            a,b=group[candidate],group[reference]
            paired.append(dict(task="cube",case_set=case_set,checkpoint=checkpoint,arm=a["arm"],seed=a["seed"],
                candidate_metric=candidate,reference_metric=reference,candidate_job=a["id"],reference_job=b["id"],
                **paired_outcomes(a["episode_successes"],b["episode_successes"],bootstrap)))
        first_checks.append(compare_first_banks(group,first_banks))
    assert len(paired)==24 and len(first_checks)==8
    seed_averages=[]
    for case_set in ["primary","fresh"]:
        for arm in ["identity","bounded"]:
            for candidate,reference in METRIC_PAIRS:
                group=sorted([p for p in paired if(p["case_set"],p["arm"],p["candidate_metric"],p["reference_metric"])
                    ==(case_set,arm,candidate,reference)],key=lambda x:x["seed"])
                assert [p["seed"]for p in group]==[1,2]
                matrix=np.asarray([p["per_case_difference"]for p in group],dtype=float)
                rng=np.random.default_rng(1042)
                draws=matrix.mean(0)[rng.integers(50,size=(bootstrap,50))].mean(1)*100
                seed_averages.append(dict(case_set=case_set,arm=arm,candidate_metric=candidate,reference_metric=reference,
                    differences_by_seed_pp=[p["difference_pp"]for p in group],mean_difference_pp=float(matrix.mean()*100),
                    case_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                    scope="Shared-case bootstrap conditional on two fixed trained models. No training-seed uncertainty, multiple-comparison correction or pairing between primary/fresh sets."))
    return dict(completed_endpoint_count=34,completed_endpoints=completed,source_commit=protocol["source_commit"],
        protocol=protocol,plan_sha256=digest(root/"plan.json"),protocol_sha256=digest(root/"protocol.json"),
        rows=rows,case_sets=case_sets,calibrations=calibrations,primary_identity_exact_replays=identity_replay,
        paired_metrics=paired,paired_metric_seed_averages=seed_averages,first_bank_cross_run_checks=first_checks,
        bootstrap_repetitions=bootstrap,bootstrap_seed=1042,collector_sha256=digest(__file__),
        reused_pairing_helper_sha256=digest(paired_outcomes.__code__.co_filename),
        scope="Metric interventions change only the controller cost. Underlying sample-level world-model losses/weights are unchanged. Primary and fresh manifests/policy seeds are separate evaluation sets. Native TF/released references retain their original normalization/capacity/exposure. Raw candidate arrays stay on EFS; same-bank rankings are descriptive search-path-conditioned diagnostics, not physical outcome quality or independent samples.")


def write_report(result,path):
    lines=["# Carrier diagnostics: complete 34-job review","",
        f"All34 endpoints exit0; source `{result['source_commit']}`. All four primary native-identity evaluations reproduce the original50 case outcomes exactly.","",
        "| Task / set | Model | Seed | Planner metric | Success | Normalization |",
        "|---|---|---:|---|---:|---|"]
    for r in sorted(result["rows"],key=lambda x:(x["task"],x["case_set"],x["family"],x["arm"]or"",x["seed"] or 0,x["metric"])):
        lines.append(f"| {r['task']} / {r['case_set']} | {r['family']} {r['arm']or''} | {r['seed']if r['seed']is not None else'—'} | {r['metric']} | {r['success_rate']:g}% | {r['normalization']} |")
    lines += ["","Metric comparisons below are within the same frozen checkpoint and case set. Means retain both training seeds; intervals resample the shared50 cases jointly and do not measure training-seed stability.","",
        "| Case set | Carrier chart | Candidate − reference | Seed1 / seed2 | Mean difference [case CI] |",
        "|---|---|---|---:|---:|"]
    for r in result["paired_metric_seed_averages"]:
        d=r["differences_by_seed_pp"];lo,hi=r["case_bootstrap_ci95_pp"]
        lines.append(f"| {r['case_set']} | {r['arm']} | {r['candidate_metric']} − {r['reference_metric']} | {d[0]:+g}/{d[1]:+g} pp | {r['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] pp |")
    lines += ["","First-bank replay checks only align the first optimization iteration for each case in the first native CEM solve. Hash agreement and any mismatched cases are explicit in results.json. Each later bank's three metric scores share the same predictions, but banks across selected-metric runs can diverge and are not treated as common candidates.","",
        "Spearman statistics, stable-order agreement, top1 disagreement, elite overlap and tie rates are retained separately for each selected-metric search path. They have no inferential confidence intervals: CEM banks are dependent and are not new physical cases.","",
        "Fresh sets exclude exact primary pairs but come from the same validation episode pool; episode overlap and changed policy seeds are disclosed. Do not pair primary/fresh case vectors. Released LeWM retains unmatched training exposure/capacity and full-data normalization; F and TF also differ in frozen capacity and calibration access.","",
        "These are controller metric diagnostics. The sample-level learning objective and the600-step model weights remain unchanged. A rank change is not evidence of better physical outcomes; closed-loop success remains a separate outcome. All seeds and failures are retained; this collection does not authorize a long-run promotion.",""]
    Path(path).write_text("\n".join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default="/efs/prescottsun/SPT-JEPA/runs/carrier_diagnostics_20261031")
    p.add_argument("--output",required=True);p.add_argument("--bootstrap",type=int,default=10000)
    a=p.parse_args();assert a.bootstrap>=100
    result=collect(a.root,a.bootstrap)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    write_report(result,out/"RESULTS.md")
    print(json.dumps(dict(endpoints=34,exact_identity_replays=4,
        scores=[{k:r[k]for k in["id","task","case_set","success_rate"]}for r in result["rows"]]),indent=2))


if __name__=="__main__":main()
