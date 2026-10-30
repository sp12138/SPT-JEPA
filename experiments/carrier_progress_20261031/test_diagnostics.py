"""Two new CPU contracts for recorded-bank ranking/alignment; no active data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.carrier_progress_20261031.collect_diagnostics import (
    METRICS,compare_first_banks,digest,summarize_banks,
)


def fixture(path,later_diverges=False):
    costs=np.tile(np.array([[0,1,2,3],[3,2,1,0],[2,2,2,2]],dtype=np.float32)[:,None,:],(1,100,1))
    if later_diverges:costs[:,1::2]=costs[:,1::2,::-1]+10
    orders=np.argsort(costs,axis=-1,kind="stable").astype(np.int32)
    np.savez_compressed(path/"metric-costs.npz",costs=costs,stable_orders=orders,call_offsets=np.arange(101))
    banks=[]
    for i in range(100):
        suffix="-diverged"if later_diverges and i%2 else""
        banks.append(dict(call=i,shape=[3,1,4],action_candidates_sha256=f"action{i}{suffix}",
            goal_emb_sha256=f"goal{i//2}",pixels_sha256=f"pixels{i//2}",action_history_sha256=f"history{i//2}",
            endpoint_difference_sha256=f"endpoint{i}{suffix}",
            selected_indices={name:[int(costs[k,i].argmin())]for k,name in enumerate(METRICS)}))
    (path/"metric-candidate-banks.json").write_text(json.dumps(banks))
    return dict(solver=dict(num_samples=4,n_steps=2,topk=2,batch_size=1),metric_diagnostic=dict(candidate_audit=dict(
        metric_order=list(METRICS),extra_model_rollouts=0,manual_identity_max_abs_error=0,calls=100,candidate_batches=100,
        costs_sha256=digest(path/"metric-costs.npz"),candidate_banks_sha256=digest(path/"metric-candidate-banks.json"))))


class DiagnosticContracts(unittest.TestCase):
    def test_ties_spearman_and_archive_consistency(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory);report=fixture(path);result,first=summarize_banks(path,report)
            pair=result["metric_pairs"]["observation__vs__identity"]
            self.assertAlmostEqual(pair["spearman"]["mean"],-1.,places=14)
            self.assertEqual(pair["stable_topk_overlap_fraction"]["mean"],0.)
            constant=result["metric_pairs"]["increment__vs__identity"]["spearman"]
            self.assertEqual(constant["undefined"],100);self.assertIsNone(constant["mean"])
            self.assertEqual(result["ties"]["increment"]["topk_boundary_tie_fraction"],1.)
            self.assertEqual(first["costs"].shape,(3,50,4))
            # An inconsistent declared ordering must be rejected.
            with np.load(path/"metric-costs.npz")as f:data=dict(f)
            data["stable_orders"][0,0]=data["stable_orders"][0,0,::-1]
            np.savez_compressed(path/"metric-costs.npz",**data)
            report["metric_diagnostic"]["candidate_audit"]["costs_sha256"]=digest(path/"metric-costs.npz")
            with self.assertRaises(AssertionError):summarize_banks(path,report)

    def test_only_initial_banks_are_aligned_and_hash_mismatches_excluded(self):
        with tempfile.TemporaryDirectory()as directory:
            groups={};first_banks={}
            for i,name in enumerate(METRICS):
                path=Path(directory)/name;path.mkdir();report=fixture(path,later_diverges=i>0)
                _,first=summarize_banks(path,report)
                groups[name]=dict(id=name,checkpoint="same-checkpoint",case_set="fresh")
                first_banks[name]=first
            result=compare_first_banks(groups,first_banks)
            self.assertEqual(len(result["matching_input_and_endpoint_case_indices"]),50)
            self.assertTrue(all(v==0 for v in result["matched_output_cost_max_abs_errors"]))
            # Later CEM banks differed, but are intentionally never compared.
            changed=copy.deepcopy(first_banks);changed["increment"]["rows"][3]["pixels_sha256"]="different-state"
            result=compare_first_banks(groups,changed)
            self.assertEqual(result["mismatch_case_indices"]["pixels_sha256"],[3])
            self.assertNotIn(3,result["matching_input_case_indices"])
            self.assertEqual(len(result["matched_output_cost_max_abs_errors"]),49)


if __name__=="__main__":unittest.main()
