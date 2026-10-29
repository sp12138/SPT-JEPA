"""Aggregate the fixed Cube first-window-epoch comparison without scheduling more work."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
b=Path("/efs/prescottsun/SPT-JEPA");root=b/"runs/hardtasks_cube_epoch1_20261030"
out=b/"reports/hardtasks-cube-epoch1-review-20261030";out.mkdir(exist_ok=True)
plan=json.loads((root/"plan.json").read_text());rows=[];signature=None
for j in plan["jobs"]:
    if j["phase"]!="report":assert json.loads((root/"exits"/(j["id"]+".json")).read_text())["returncode"]==0
for j in [x for x in plan["jobs"] if x["phase"]=="epoch1"]:
    run=Path(j["output"])
    assert json.loads((run/"stage-complete.json").read_text())["stop_step"]==11671
    h=json.loads((root/("health-"+j["id"])/"health.json").read_text())
    p=json.loads((root/("planning-"+j["id"])/"metrics.json").read_text())
    assert h["audit"]["step"]==11671 and all(v for k,v in h["audit"].items() if k not in ["step","gradient_max"])
    assert all(v>0 for v in h["audit"]["gradient_max"].values())
    sig=(p["manifest_sha256"],p["pairs"])
    if signature is None:signature=sig
    assert sig==signature and len(p["metrics"]["episode_successes"])==50
    rows.append(dict(method=j["method"],training_seed=j["training_seed"],success=p["metrics"]["success_rate"],health=h,planning=p))
result=dict(all18_exit0=True,stop_step=11671,records=rows,no_automatic_continuation=True,
            scope="Two training seeds per method, fixed 50 strict-validation episodes. One near-complete traversal of training windows; test episodes untouched. Released reference has different budget, size and data exposure.")
(out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
lines=["# Cube: first window epoch","","All 18 training, health and planning endpoints exit 0 and stop at 11,671 updates. Both seeds and every method are retained. These are validation results, not a final test claim.","",
       "| Method | Seed 1 % | Seed 2 % | Mean % |","|---|---:|---:|---:|"]
fig,ax=plt.subplots(figsize=(7,4))
for i,method in enumerate(["original","lower-beta","temporal-forecast"]):
    selected=sorted([x for x in rows if x["method"]==method],key=lambda x:x["training_seed"])
    vals=[x["success"] for x in selected]
    lines.append(f"| {method} | {vals[0]:.1f} | {vals[1]:.1f} | {np.mean(vals):.1f} |")
    ax.bar(i,np.mean(vals),alpha=.7);ax.scatter([i-.06,i+.06],vals,color="black",zorder=3)
ax.axhline(30,color="black",ls="--",label="Released LeWM (different budget)")
ax.set(ylim=(0,100),ylabel="Strict validation success (%)",xticks=range(3),xticklabels=["Original","Lower beta","Temporal forecast"],title="Cube: matched 11,671 updates")
ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2);fig.tight_layout()
for ext in ["png","svg"]:fig.savefig(out/("planning."+ext),dpi=170)
lines+=["","Original and lower-beta resume their exact 2000-step full checkpoints. Temporal-forecast reuses its unchanged ordinary 2600-step continuation. Total update counts, data, planner and cases match at this endpoint; parameter count matches within Cube. Update matching does not imply identical FLOPs.",
        "","The cap covers 1,493,888 of 1,494,000 natural windows at batch 128. LR horizon remains 124,910. No second epoch or candidate grid is automatically authorized. Released LeWM scored 30% and random 2% on these cases, with different model/training/data-exposure conditions.","","![Planning](planning.png)",""]
(out/"RESULTS.md").write_text("\n".join(lines))
print(json.dumps([{k:v for k,v in x.items() if k not in ["health","planning"]} for x in rows]),flush=True)
