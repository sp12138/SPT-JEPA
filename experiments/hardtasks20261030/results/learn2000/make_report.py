import pathlib,json,numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
b=pathlib.Path("/efs/prescottsun/SPT-JEPA");r=b/"runs/hardtasks_learn2000_20261030";out=b/"reports/hardtasks-learn2000-review-20261030";plan=json.loads((r/"plan.json").read_text());records=[];signatures={};references=[]
for j in plan["jobs"]:assert json.loads((r/"exits"/(j["id"]+".json")).read_text())["returncode"]==0
for j in [x for x in plan["jobs"] if x["phase"]=="learn2000"]:
 name=j["id"];h=json.loads((r/("health-"+name)/"health.json").read_text());p=json.loads((r/("planning-"+name)/"metrics.json").read_text());stage=json.loads((r/name/"stage-complete.json").read_text());assert h["audit"]["step"]==stage["stop_step"]==2000;assert len(p["metrics"]["episode_successes"])==len(p["pairs"])==50;assert len(set(x["episode_id"] for x in p["pairs"]))==50
 sig=(p["manifest_sha256"],p["pairs"]);signatures.setdefault(j["task"],sig);assert signatures[j["task"]]==sig
 records.append(dict(task=j["task"],method=j["method"],training_seed=j["training_seed"],success=p["metrics"]["success_rate"],planning=p,health=h,stage=stage,config=json.loads((r/name/"config.json").read_text())))
for task in ["tworoom","cube"]:
 for policy in ["random","released-lewm"]:
  p=json.loads((b/"runs/hardtasks_gate600_20261030"/f"reference-{task}-{policy}-strict-validation/metrics.json").read_text());assert signatures[task]==(p["manifest_sha256"],p["pairs"]);references.append(dict(task=task,method=policy,success=p["metrics"]["success_rate"],planning=p))
result=dict(step=2000,endpoints=60,all_exit_zero=True,all_seeds_retained=True,records=records,references=references,limits=["Early2000-update screen; not mature-task performance.","Two training seeds and50 held-out validation episodes per task; test split untouched.","Released reference has different training budget/model size and possible held-out-data exposure.","Health and action sensitivity alone do not establish useful planning.","No academic novelty claim."]);(out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
methods=["original","lower-beta","temporal-forecast","original-highnoise","temporal-forecast-highnoise"];fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True);lines=["# Harder-task first learning screen","","All60 endpoints exit0. Each method uses the same2000 updates, both training seeds, and the same50 validation episodes per task. Test episodes remain held out.","","| Task | Method | Seed1% | Seed2% |","|---|---|---:|---:|"]
for ax,task in zip(axes,["tworoom","cube"]):
 values=[]
 for i,m in enumerate(methods):
  rows=sorted([x for x in records if x["task"]==task and x["method"]==m],key=lambda x:x["training_seed"]);y=[x["success"] for x in rows];values.append(y);lines.append(f"| {task} | {m} | {y[0]:.1f} | {y[1]:.1f} |");print(task,m,y,flush=True)
 values=np.asarray(values);ax.bar(range(5),values.mean(1),color=[plt.get_cmap("tab10")(i) for i in range(5)],alpha=.7)
 for i,y in enumerate(values):ax.scatter([i-.07,i+.07],y,color="black",s=22,zorder=3)
 ref=next(x["success"] for x in references if x["task"]==task and x["method"]=="released-lewm");ax.axhline(ref,color="black",linestyle="--",label=f"Released LeWM {ref:.0f}% (different budget)");ax.set(title=task,ylim=(0,100),xticks=range(5),xticklabels=["Original","Lower\nbeta","Temporal\nforecast","Original\nhigh noise","Temporal\nhigh noise"]);ax.tick_params(axis="x",labelsize=8);ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
axes[0].set_ylabel("Strict validation success (%)");fig.suptitle("Harder tasks at 2000 updates: actual closed-loop control",fontsize=13);fig.tight_layout(rect=[0,.07,1,.95]);fig.text(.5,.02,"Bars = two-seed mean; dots = individual training seeds. Same50 cases per task; no mature-performance or novelty claim.",ha="center",fontsize=9)
for ext in ["png","svg"]:fig.savefig(out/("planning."+ext),dpi=170)
lines += ["","Released-reference scores on these same cases: TwoRoom72%, Cube30%; random0%/2%. Released weights are not training-budget/model-size/data-exposure matched controls.","","TwoRoom remains nearly state-independent at2000 despite denoising improvement; a larger timestep shift alone has not solved it. Cube learns appreciable state/action dependence, but that must be distinguished from task success. The next causal check changes only clean anchor availability and the same global noise shift, with identical2000-step temporal-forecast source state per task/seed and only600 further updates. All existing snapshots remain held.","","![Planning](planning.png)",""]
(out/"RESULTS.md").write_text("\n".join(lines))
