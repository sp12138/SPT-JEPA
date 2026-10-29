from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
b=Path("/efs/prescottsun/SPT-JEPA");r=b/"runs/hardtasks_anchor_20261030";out=b/"reports/hardtasks-anchor2600-review-20261030";plan=json.loads((r/"plan.json").read_text());rows=[];signatures={}
for j in plan["jobs"]:assert json.loads((r/"exits"/(j["id"]+".json")).read_text())["returncode"]==0
for j in [x for x in plan["jobs"] if x["phase"]=="adapt600"]:
 name=j["id"];h=json.loads((r/("health-"+name)/"health.json").read_text());p=json.loads((r/("planning-"+name)/"metrics.json").read_text());assert h["audit"]["step"]==2600 and len(p["metrics"]["episode_successes"])==50;sig=(p["manifest_sha256"],p["pairs"]);signatures.setdefault(j["task"],sig);assert signatures[j["task"]]==sig;s=h["summary"];f=s["fm_t0.95_forecast"]
 rows.append(dict(task=j["task"],method=j["method"],training_seed=j["training_seed"],success=p["metrics"]["success_rate"],state_effect_percent=100*(s["fm_t0.95_wrong_state"]-f)/f,action_effect_percent=100*(s["fm_t0.95_wrong_actions"]-f)/f,health=h,planning=p))
report=dict(all48_exit0=True,initial_step=2000,stop_step=2600,additional_updates=600,records=rows,scope="Same full initial state per task/seed, fixed2x2 clean-anchor access/noise factorial. Short adaptation only; not a fresh-training or novelty claim.");(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
methods=["observed-base","observed-highnoise","constant-base","constant-highnoise"];labels=["Observed","Observed\n+ high noise","Constant","Constant\n+ high noise"];fig,axes=plt.subplots(2,2,figsize=(12,8));lines=["# Clean visual information versus corruption","","All48 endpoints exit0. All branches start from the same ordinary-noise temporal-forecast2000 full checkpoint within task/seed, then receive exactly600 updates. All16 full-state audits pass.","","| Task | Method | Seed1% | Seed2% |","|---|---|---:|---:|"]
for column,task in enumerate(["tworoom","cube"]):
 vals=[];state=[];action=[]
 for i,m in enumerate(methods):
  selected=sorted([x for x in rows if x["task"]==task and x["method"]==m],key=lambda x:x["training_seed"]);y=[x["success"] for x in selected];vals.append(y);state.append([x["state_effect_percent"] for x in selected]);action.append([x["action_effect_percent"] for x in selected]);lines.append(f"| {task} | {m} | {y[0]:.1f} | {y[1]:.1f} |")
 vals=np.asarray(vals);ax=axes[0,column];ax.bar(range(4),vals.mean(1),alpha=.7,color=[plt.get_cmap("tab10")(i) for i in range(4)])
 for i,y in enumerate(vals):ax.scatter([i-.07,i+.07],y,color="black",s=20,zorder=3)
 ax.axhline(72 if task=="tworoom" else 30,color="black",ls="--",label="Released LeWM (different budget)");ax.set(title=task,ylim=(0,100),xticks=range(4),xticklabels=labels);ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
 ax=axes[1,column]
 for values,name in [(state,"Wrong initial state"),(action,"Wrong action sequence")]:
  values=np.asarray(values);ax.plot(range(4),values.mean(1),marker="o",label=name)
  for seed in [0,1]:ax.scatter(np.arange(4)+(seed-.5)*.08,values[:,seed],s=15,alpha=.4)
 ax.set(xticks=range(4),xticklabels=labels);ax.axhline(0,color="gray",lw=.5);ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0,0].set_ylabel("Strict validation success (%)");axes[1,0].set_ylabel("Denoising MSE increase at t=.95 (%)");fig.suptitle("State use, action use and actual control can diverge",fontsize=14);fig.tight_layout(rect=(0,.07,1,.95));fig.text(.5,.02,"Same2000-step source +600 updates; two training seeds.50 fixed validation cases. State substitutions preserve posterior noise.\nDenoising sensitivity is not a control-success or generated-sample-quality metric.",ha="center",fontsize=9)
for ext in ["png","svg"]:fig.savefig(out/("causal-controls."+ext),dpi=170)
lines += ["","TwoRoom: retaining clean anchor access produces action-sensitive denoising while the encoded initial-state intervention remains negligible. Removing that access suppresses the action effect without recovering useful state encoding or strong control in this short adaptation. This rules out treating a noise shift or anchor cut as an already validated cure; it does not settle a fresh no-anchor model's eventual performance.","","Cube: state/action interventions materially worsen denoising, yet all four control rates stay9–12% on average. Strong denoising state use is therefore insufficient to establish good latent-goal planning.","","Code inspection identifies a further mismatch: the rate constrains a teacher-forced one-step prior, whereas the rendering objective also trains free-running forecasts. The next minimal test replaces only the rate prior with that same causal forecast, preserving loss weights and parameters. This is related to known multistep latent consistency/overshooting and is not an established novelty claim.","","![Causal controls](causal-controls.png)",""]
(out/"RESULTS.md").write_text("\n".join(lines));print(json.dumps([dict(task=x["task"],method=x["method"],seed=x["training_seed"],success=x["success"]) for x in rows]),flush=True)
