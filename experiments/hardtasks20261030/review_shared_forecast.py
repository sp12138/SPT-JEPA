"""Compare a shared causal prior against the exact completed ordinary controls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

b=Path("/efs/prescottsun/SPT-JEPA")
root=b/"runs/hardtasks_sharedprior_20261030"
out=b/"reports/hardtasks-sharedprior2600-review-20261030"
plan=json.loads((root/"plan.json").read_text())
assert len(plan["jobs"])==12
for job in plan["jobs"]:
    assert json.loads((root/"exits"/(job["id"]+".json")).read_text())["returncode"]==0
copies=json.loads((root/"resume-audit.json").read_text())
rows=[]
for copy in copies:
    candidate=root/copy["id"]
    control=Path(copy["reused_control"])
    task=copy["id"].split("-")[0]
    seed=int(copy["id"].split("-seed")[1].split("-")[0])
    pair=[]
    for method,run in [("temporal-forecast",control),("shared-forecast-prior",candidate)]:
        health=json.loads((run.parent/("health-"+run.name)/"health.json").read_text())
        planning=json.loads((run.parent/("planning-"+run.name)/"metrics.json").read_text())
        assert health["audit"]["step"]==2600
        assert all(v for k,v in health["audit"].items() if k not in ["step","gradient_max"])
        assert all(v>0 for v in health["audit"]["gradient_max"].values())
        assert len(planning["metrics"]["episode_successes"])==50
        s=health["summary"];f=s["fm_t0.95_forecast"]
        row=dict(task=task,training_seed=seed,method=method,success=planning["metrics"]["success_rate"],
                 feature_std=health["feature_std"],state_effect_percent=100*(s["fm_t0.95_wrong_state"]-f)/f,
                 action_effect_percent=100*(s["fm_t0.95_wrong_actions"]-f)/f,health=health,planning=planning)
        rows.append(row);pair.append(row)
    p,q=[x["planning"] for x in pair]
    assert p["manifest_sha256"]==q["manifest_sha256"] and p["pairs"]==q["pairs"]
    original=np.asarray(p["metrics"]["episode_successes"],dtype=int)
    changed=np.asarray(q["metrics"]["episode_successes"],dtype=int)
    pair[1]["paired_changes"]=dict(gained=int(((changed==1)&(original==0)).sum()),lost=int(((changed==0)&(original==1)).sum()),
                                  difference_percentage_points=float(100*(changed-original).mean()))
out.mkdir(exist_ok=True)
report=dict(all12_exit0=True,initial_step=2000,stop_step=2600,full_initial_states_identical=True,
            reused_control_default_path_exact=True,records=rows,
            scope="Two seeds; 600-update adaptations from ordinary temporal-forecast. Fixed validation cases; test untouched. No fresh-training, novelty or broad stability claim.")
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
lines=["# Sharing the causal prior between rate and DiT","",
       "All 12 new endpoints exit 0. Four full-state and gradient audits pass. Each candidate starts from the exact full checkpoint used by its completed ordinary temporal-forecast control and receives 600 updates, ending at 2600. Default-loss, gradient and RNG equivalence is separately recorded.","",
       "| Task | Method | Seed 1 success % | Seed 2 success % |",
       "|---|---|---:|---:|"]
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,task in zip(axes,["tworoom","cube"]):
    for i,method in enumerate(["temporal-forecast","shared-forecast-prior"]):
        selected=sorted([x for x in rows if x["task"]==task and x["method"]==method],key=lambda x:x["training_seed"])
        vals=[x["success"] for x in selected]
        ax.bar(i,np.mean(vals),alpha=.7)
        ax.scatter([i-.05,i+.05],vals,color="black",s=22,zorder=3)
        lines.append(f"| {task} | {method} | {vals[0]:.1f} | {vals[1]:.1f} |")
    ax.axhline(72 if task=="tworoom" else 30,color="black",ls="--",label="Released LeWM (different budget)")
    ax.set(title=task,ylim=(0,100),xticks=[0,1],xticklabels=["Ordinary forecast","Shared prior"])
    ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
axes[0].set_ylabel("Strict validation success (%)")
fig.suptitle("A common causal trajectory still needs to improve control")
fig.tight_layout(rect=(0,.09,1,.93))
fig.text(.5,.025,"Same full initial states; two training seeds; 50 fixed cases per task. Short adaptation, not a fresh-training result.",ha="center",fontsize=8)
for ext in ["png","svg"]:fig.savefig(out/("planning."+ext),dpi=170)
lines+=["","| Task | Method | Seed | State substitution MSE increase % | Action substitution MSE increase % | Encoder std |",
        "|---|---|---:|---:|---:|---:|"]
for x in rows:
    lines.append(f'| {x["task"]} | {x["method"]} | {x["training_seed"]} | {x["state_effect_percent"]:.4f} | {x["action_effect_percent"]:.2f} | {x["feature_std"]:.6f} |')
lines+=["","Substitutions use identical posterior and rendering noise at corruption t=.95. These are denoising sensitivity measurements, not generated-sample quality or control success. Released LeWM differs in training budget, model size and possible evaluation-data exposure. Only paired internal controls support method comparisons.",
        "","The objective shares a free-running forecast between rendering and rate with no new head, coefficient or parameter. It is related to latent overshooting and multistep consistency; this experiment does not establish academic novelty.",
        "","![Planning](planning.png)",""]
(out/"RESULTS.md").write_text("\n".join(lines))
print(json.dumps([{k:v for k,v in x.items() if k not in ["health","planning"]} for x in rows]),flush=True)
