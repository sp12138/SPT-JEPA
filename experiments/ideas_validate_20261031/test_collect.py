"""Synthetic CPU contract tests for the analysis-only collector; no real scores."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ideas_validate_20261031.collect import (
    ARMS, collect, digest, paired_outcomes, reduce_state_audit, validate_initial_identity,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def make_fixture(root):
    """Minimal complete schema, marked synthetic; not an experimental artifact."""
    root = Path(root)
    records, jobs, initial = [], [], []
    phases = ["train600", "health", "info", "transition", "planning"]
    for task in ["tworoom", "cube"]:
        windows = dict(count=64, task=task, synthetic_fixture=True)
        for seed in [1, 2]:
            for ai, (family, arm) in enumerate(ARMS):
                name=f"{task}-{family}-{arm}-seed{seed}"
                cfg=dict(model_class=family, model={}, carrier_calibration={"sha256": task+str(seed)})
                write(root/name/"config.json", cfg)
                records.append(dict(id=name,task=task,seed=seed,family=family,arm=arm,
                                    config_sha256=digest(root/name/"config.json")))
                initial.append(dict(id=name,all_original_tensors_equal=True,common_tensors=550))
                total=1000 if family=="carrier" else 1100
                trainable=100 if arm=="identity" else 200 if family=="carrier" else 1100
                write(root/name/"status.json",dict(state="finished",step=600))
                write(root/name/"model_size.json",dict(total=total,trainable=trainable,frozen=total-trainable))
                (root/name/"metrics.jsonl").write_text(json.dumps(dict(step=600,loss=1.))+"\n")
                health=dict(audit=dict(step=600,finite=True),parameters=total,cases=[dict(episode=3,start=0)])
                write(root/("health-"+name)/"health.json",health)
                probes=dict(train_episodes=[1,2],validation_episodes=[3,4],label_keys=["diagnostic"],
                            scores={"clean":{"mean_r2":.3,"r2_against_train_mean":[.3],"mse":[1.],"informative_target_dimensions":1}})
                write(root/("info-"+name)/"information-audit.json",dict(step=600,batches=[],probes=probes))
                wp=root/("transition-"+name)/"windows.json";write(wp,windows)
                modes=["observed_future","clean_rollout","teacher_forced_prefix","shifted_observed_carry",
                       "initial_carry","zero_prediction","other_initial","zero_initial","wrong_action_order","other_episode_actions"]
                state=dict(step=600,cases=64,config_sha256=records[-1]["config_sha256"],manifest_sha256=digest(wp),
                    actual_temporal_variation={"initial_mean":1.},
                    conditions={mode:dict(mse=.1*(ai+1)*seed,normalized_by_initial_variation=.1*(ai+1)*seed,
                                          mse_change_vs_clean_rollout=.1,per_window_mse=[.1]*64) for mode in modes})
                if family=="carrier":
                    state["carrier_chart_certificate"]=dict(residual_lipschitz_upper=[0,1.2,.3][ai],
                        input_normalization_active=arm=="bounded",output_normalization_active=False)
                else:
                    heads=["symbol","symbol_future_only","completion","transform"]
                    state["path_queries"]=dict(trained_mode=arm,query_step=0,query_indices=[[0,1,2]],
                        label_support={"symbols":{"counts":[1,2,3]}},
                        conditions={mode:{head:dict(ce=.5,relative_to_uniform_ce=.6,accuracy=.7,per_window_ce=[.5]*64,
                                                   per_position_ce=[.5]*3) for head in heads} for mode in modes})
                write(root/("transition-"+name)/"state-audit.json",state)
                successes=[i<(ai+seed)%5 for i in range(50)]
                write(root/("planning-"+name)/"metrics.json",dict(
                    metrics=dict(episode_successes=successes,success_rate=100*sum(successes)/50),
                    checkpoint_audit=dict(parameters=total,strict_equal=True),
                    pairs=[dict(episode_id=i,start_step=0) for i in range(50)],
                    manifest_sha256=task,solver={"batch_size":1},plan={"history_len":3},protocol={"task":task},
                    split="validation",seconds=1,checkpoint=str(root/name/"model-step0000600"),normalization="train episodes"))
                for phase in phases:
                    jobid=name if phase=="train600" else phase+"-"+name
                    jobs.append(dict(id=jobid,phase=phase,source={"source_commit":"synthetic-contract-fixture"}))
                    write(root/"exits"/(jobid+".json"),dict(returncode=0))
    write(root/"plan.json",dict(jobs=jobs))
    write(root/"protocol.json",dict(runs=records,source_commit="synthetic-contract-fixture",stop_step=600,
                                   preflight=str(root/"preflight")))
    write(root/"initial-tensor-identity.json",dict(passed=True,runs=initial,paired_family_tensors_equal=True))
    write(root/"preflight"/"review.json",dict(passed=True))
    return records


class CollectionContracts(unittest.TestCase):
    def test_paired_outcomes_preserve_gained_and_lost(self):
        result=paired_outcomes([1,0,1,0],[0,1,1,0],bootstrap=100)
        self.assertEqual(result["gained"],1);self.assertEqual(result["lost"],1)
        self.assertEqual(result["difference_pp"],0)
        self.assertEqual(result["per_case_difference"],[1,-1,0,0])
        with self.assertRaises(ValueError):paired_outcomes([1],[1,0])

    def test_query_reduction_preserves_primary_cases(self):
        state=dict(path_queries=dict(trained_mode="symbol",conditions={"clean_rollout":{
            head:dict(per_window_ce=[1,2],ce=1.5) for head in ["symbol","symbol_future_only","transform"]}}))
        reduced=reduce_state_audit(state)
        self.assertIn("per_window_ce",reduced["path_queries"]["conditions"]["clean_rollout"]["symbol_future_only"])
        self.assertNotIn("per_window_ce",reduced["path_queries"]["conditions"]["clean_rollout"]["transform"])
        self.assertIn("per_window_ce",state["path_queries"]["conditions"]["clean_rollout"]["transform"])

    def test_complete_grid_and_hidden_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=make_fixture(root)
            result=collect(root,bootstrap=100)
            self.assertEqual(len(result["rows"]),24)
            self.assertEqual(result["completed_endpoint_count"],120)
            self.assertEqual(len(result["paired"]),24)
            self.assertEqual(len(result["paired_seed_averages"]),12)
            self.assertTrue(all(len(r["episode_successes"])==50 for r in result["rows"]))
            # A changed planner or case order must never silently become paired.
            path=root/("planning-"+records[1]["id"])/"metrics.json"
            modified=json.loads(path.read_text());modified["pairs"]=list(reversed(modified["pairs"]))
            write(path,modified)
            with self.assertRaises(AssertionError):collect(root,bootstrap=100)

    def test_initial_review_must_explicitly_pass(self):
        with self.assertRaises(AssertionError):validate_initial_identity({"passed":False,"runs":[]},[])
        with self.assertRaises(AssertionError):
            validate_initial_identity({"passed":True,"runs":[dict(id="wrong",all_original_tensors_equal=True)],
                                       "paired_family_tensors_equal":True},[dict(id="expected")])


if __name__=="__main__":
    unittest.main()
