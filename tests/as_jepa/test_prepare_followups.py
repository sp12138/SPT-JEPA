import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from as_jepa_tools.prepare_full_followups import prepare


class FollowupPreparationContracts(unittest.TestCase):
    def test_new_followups_are_gated_and_existing_frozen_jobs_stay_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)/"project";root=project/"runs/campaign";root.mkdir(parents=True)
            source=dict(source=str(root/"frozen"),source_commit="abc",manifest=str(root/"frozen.json"),pythonpath="explicit")
            source_file=root/"new-source.json";source_file.write_text(json.dumps(source))
            jobs=[]
            for family in ["L2","S2"]:
                for beta in [1,4]:
                    name=f"full-{family}-beta{beta}-stage3-seed0";out=root/name;out.mkdir()
                    if family=="L2":
                        cfg=dict(output=str(out),seed=0,max_steps=124910,stop_after=37473,model=dict(beta=beta,mode="L2",anchor_gap=8))
                        path=out/"config.json";path.write_text(json.dumps(cfg))
                    else:
                        cfg=dict(model=dict(extras=dict(as_jepa=dict(kind="S2",beta=beta),campaign_output=str(out))),
                                 runtime=dict(max_steps=37473,seed=0,output_dir=str(out)),
                                 checkpoint=dict(every_n_steps=12491),data=dict(sampler_seed=0),
                                 resume=dict(path="none"))
                        path=out/"config.yaml";path.write_text(yaml.safe_dump(cfg))
                    jobs.append(dict(id=name,host="dev13",gpu=0,output=str(out),source=source,config=str(path),
                                     config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),command=["python","explicit-entry"]))
            plan=root/"plan.json";plan.write_text(json.dumps(dict(root=str(root),jobs=jobs)))
            original=copy.deepcopy(jobs)
            registry=root/"registry.json";prepare(plan,source_file,registry)
            result=json.loads(plan.read_text());byid={x["id"]:x for x in result["jobs"]}
            for j in original:
                self.assertEqual(byid[j["id"]],j)
                self.assertEqual(hashlib.sha256(Path(j["config"]).read_bytes()).hexdigest(),j["config_sha256"])
            reg=json.loads(registry.read_text())
            for trial in reg["trials"]:
                for key in ["stage3_audit","full_job","final_audit","score8","score10"]:
                    self.assertIn(trial[key],byid)
            dormant=[j for j in result["jobs"] if j["id"].startswith("final-") and j["id"].endswith("-stage3")]
            self.assertTrue(dormant)
            for j in dormant:
                self.assertEqual(len(j["requires"]),1)
                self.assertFalse(Path(j["requires"][0]).exists())
                if "-S2-" in j["id"]:
                    self.assertEqual(yaml.safe_load(Path(j["config"]).read_text())["resume"]["path"],"none")
            ramp=[j for j in result["jobs"] if j["id"].startswith("full-S2-ramp") and j["id"].endswith("stage3-seed0")]
            self.assertTrue(ramp)
            self.assertTrue(all(yaml.safe_load(Path(j["config"]).read_text())["model"]["extras"]["ramp_steps"]==6245 for j in ramp))


if __name__=="__main__":
    unittest.main()
