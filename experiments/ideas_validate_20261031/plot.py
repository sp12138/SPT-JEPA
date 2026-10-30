"""CPU plotting of the completed six-arm validation; all seeds stay visible."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ARMS=[("carrier","identity"),("carrier","unconstrained"),("carrier","bounded"),
      ("pathquery","symbol"),("pathquery","completion"),("pathquery","transform")]
LABELS=["F: identity","F: free chart","F: bounded","S3: symbol","S3: completion","S3: transform"]
COLORS={"carrier":"#3b6d8f","pathquery":"#c46c3b"}
MARKERS={1:"o",2:"^"}
TASKS=["tworoom","cube"]
TITLES={"tworoom":"TwoRoom","cube":"Cube"}


def clean_axis(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.18)
    ax.set_axisbelow(True)


def arm_axis(ax,labels=LABELS):
    ax.set_xticks(np.arange(len(labels)),labels,rotation=28,ha="right")
    if len(labels)==6:
        ax.axvline(2.5,color="#bbbbbb",lw=1,ls=":")
    clean_axis(ax)


def save(fig,out,name):
    for ext in ["png","svg"]:
        fig.savefig(out/(name+"."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)


def finite(value):
    return value is not None and np.isfinite(value)


def seed_legend():
    return [Line2D([0],[0],marker=MARKERS[seed],color="#333333",lw=0,label=f"Seed {seed}",markersize=7)
            for seed in [1,2]]


def plot_planning(result,lookup,out):
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.5),layout="constrained")
    fig.suptitle("Strict planning after 600 updates · 50 fixed cases per model",fontsize=15)
    for task,ax in zip(TASKS,axes):
        largest=0
        for x,(family,arm) in enumerate(ARMS):
            ys=[lookup[task,seed,family,arm]["success_rate"] for seed in [1,2]]
            largest=max(largest,*ys)
            ax.plot([x-.10,x+.10],ys,color=COLORS[family],alpha=.35,lw=1)
            ax.hlines(np.mean(ys),x-.23,x+.23,color=COLORS[family],lw=2)
            for seed,y in zip([1,2],ys):
                offset=-.10 if seed==1 else .10
                ax.scatter(x+offset,y,color=COLORS[family],marker=MARKERS[seed],s=65,zorder=4)
            ax.annotate(f"{ys[0]:g} / {ys[1]:g}",(x,max(ys)),xytext=(0,9),
                        textcoords="offset points",ha="center",fontsize=8)
        ax.set_title(TITLES[task]);ax.set_ylabel("Success (%)")
        ax.set_ylim(-3,min(105,max(22,largest+14)))
        arm_axis(ax)
        tf=result["historical_context"]["temporal_forecast600"][task]
        initial=result["historical_context"]["untrained"][task]
        ax.set_xlabel(f"Numbers show seed 1 / seed 2\nHistorical context only: TF600 {tf[0]}/{tf[1]}%; initial {initial[0]}/{initial[1]}%\nDifferent capacity / trainable parameters; excluded from paired tests",fontsize=9)
    axes[0].legend(handles=seed_legend(),loc="upper left",frameon=False,ncol=2)
    save(fig,out,"planning-all-arms")


def plot_forecast(result,lookup,out):
    modes=[("clean_rollout","Causal clean rollout"),
           ("teacher_forced_prefix","One-step with real prefix"),
           ("shifted_observed_carry","Copy latest real observation")]
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout="constrained")
    fig.suptitle("State transmission · MSE normalized by actual displacement from frame 0",fontsize=15)
    for row,task in enumerate(TASKS):
        for col,(mode,title) in enumerate(modes):
            ax=axes[row,col];values=[]
            for x,(family,arm) in enumerate(ARMS):
                for seed in [1,2]:
                    y=lookup[task,seed,family,arm]["state"]["conditions"][mode]["normalized_by_initial_variation"]
                    if finite(y):
                        ax.scatter(x+(-.09 if seed==1 else .09),y,color=COLORS[family],marker=MARKERS[seed],s=40)
                        values.append(y)
                    else:
                        ax.text(x,.04,"undefined",transform=ax.get_xaxis_transform(),rotation=90,ha="center",fontsize=7)
            ax.axhline(1,color="#777777",ls="--",lw=1,label="Initial carry = 1")
            if values and max(values)>max(.05,min(v for v in values if v>0) if any(v>0 for v in values) else .05)*40:
                if all(v > 0 for v in values):
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=min(values)*.6, top=max(values)*2)
                else:
                    ax.set_yscale("symlog",linthresh=.05)
                    ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(f"{TITLES[task]} · {title}")
            ax.set_ylabel("MSE / true temporal variation")
            arm_axis(ax)
    axes[0,0].legend(handles=seed_legend()+[Line2D([0],[0],color="#777777",ls="--",label="Initial carry = 1")],
                     frameon=False,fontsize=8)
    fig.supxlabel("Only the first column is a free rollout. The other columns receive additional real observations; lower error does not establish learned dynamics.",fontsize=10)
    save(fig,out,"forecast-transmission")


def plot_queries(result,lookup,out):
    conditions=[("observed_future","Observed states","#3378a4"),
                ("clean_rollout","Causal forecast","#c25e3e"),
                ("zero_prediction","Query / masks only","#777777"),
                ("shifted_observed_carry","Shifted observed carry","#569475")]
    arms=["symbol","completion","transform"]
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.7),layout="constrained")
    fig.suptitle("Path-query shortcut audit · identical prototypes and targets under every intervention",fontsize=14)
    for row,task in enumerate(TASKS):
        for col,metric in enumerate(["relative_to_uniform_ce","accuracy"]):
            ax=axes[row,col]
            for ci,(condition,label,color) in enumerate(conditions):
                for x,arm in enumerate(arms):
                    for seed in [1,2]:
                        q=lookup[task,seed,"pathquery",arm]["state"]["path_queries"]
                        mode="symbol_future_only" if arm=="symbol" else arm
                        y=q["conditions"][condition][mode][metric]
                        offset=(ci-1.5)*.13+(-.025 if seed==1 else .025)
                        ax.scatter(x+offset,y,color=color,marker=MARKERS[seed],s=35,
                                   label=label if x==0 and seed==1 else None)
            ax.set_title(f"{TITLES[task]} · {'CE / log(K)' if col==0 else 'Classification accuracy'}")
            ax.set_ylabel("Uniform-class CE = 1" if col==0 else "Accuracy")
            if col==0:
                ax.axhline(1,color="#999999",ls="--",lw=1)
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(-.02,1.04)
            arm_axis(ax,["Symbols (future only)","Scalar completion","Full transformation"])
    axes[0,0].legend(frameon=False,fontsize=8,loc="best")
    axes[0,1].legend(handles=seed_legend(),frameon=False,ncol=2)
    fig.supxlabel("Each arm has a different target. Compare conditions within an arm; observed/carry access real observations. A good zero-state score is a shortcut, not forecasting.",fontsize=10)
    save(fig,out,"query-shortcuts")


def plot_capacity(result,lookup,out):
    fig,axes=plt.subplots(2,2,figsize=(13.8,8.5),layout="constrained")
    fig.suptitle("Capacity accounting and the actual chart bound",fontsize=15)
    for row,task in enumerate(TASKS):
        ax=axes[row,0]
        for x,(family,arm) in enumerate(ARMS):
            for seed in [1,2]:
                size=lookup[task,seed,family,arm]["model_size"]
                xx=x+(-.08 if seed==1 else .08)
                ax.plot([xx,xx],[size["trainable"]/1e6,size["total"]/1e6],color=COLORS[family],alpha=.3)
                ax.scatter(xx,size["total"]/1e6,marker=MARKERS[seed],facecolors="none",edgecolors=COLORS[family],s=50)
                ax.scatter(xx,size["trainable"]/1e6,marker=MARKERS[seed],color=COLORS[family],s=40)
        ax.set(title=f"{TITLES[task]} · total vs trainable",ylabel="Parameters (millions)")
        ax.set_ylim(0, max(lookup[task,s,f,a]["model_size"]["total"]
                           for f,a in ARMS for s in [1,2])/1e6*1.1);arm_axis(ax)
        if row==0:
            ax.legend(handles=[Line2D([0],[0],marker="o",color="#555555",lw=0,label="Trainable"),
                               Line2D([0],[0],marker="o",markerfacecolor="none",color="#555555",lw=0,label="Total")],frameon=False)
        ax=axes[row,1]
        for x,arm in enumerate(["identity","unconstrained","bounded"]):
            for seed in [1,2]:
                certificate=lookup[task,seed,"carrier",arm]["state"]["carrier_chart_certificate"]
                value=certificate["residual_lipschitz_upper"]
                ax.scatter(x+(-.08 if seed==1 else .08),value,marker=MARKERS[seed],color=COLORS["carrier"],s=50)
        ax.axhline(.5,color="#65956d",ls="--",lw=1,label="Declared bounded-chart cap")
        ax.axhline(1,color="#c85c4e",ls=":",lw=1,label="Contraction certificate boundary")
        ax.set(title=f"{TITLES[task]} · residual Lipschitz upper bound",ylabel="L of chart residual (T = I + residual)")
        ax.set_ylim(bottom=0);arm_axis(ax,["Identity","Free chart","Bounded chart"])
        if row==0:
            ax.legend(frameon=False,fontsize=8)
    fig.supxlabel("Equal total parameters do not imply equal trainable capacity. A certified/inactive bound is not a planning gain; normalization activation flags are retained in results.json.",fontsize=10)
    save(fig,out,"capacity-and-chart")


def plot_interventions(result,lookup,out):
    modes=[("other_initial","Different real initial state"),
           ("other_episode_actions","Different recorded actions")]
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.4),layout="constrained")
    fig.suptitle("Do correct inputs improve the forecast?",fontsize=15)
    for row,task in enumerate(TASKS):
        for col,(mode,title) in enumerate(modes):
            ax=axes[row,col]
            for x,(family,arm) in enumerate(ARMS):
                for seed in [1,2]:
                    audit=lookup[task,seed,family,arm]["state"]
                    denom=audit["actual_temporal_variation"]["initial_mean"]
                    y=audit["conditions"][mode]["mse_change_vs_clean_rollout"]/denom if denom>0 else None
                    if finite(y):
                        ax.scatter(x+(-.08 if seed==1 else .08),y,marker=MARKERS[seed],color=COLORS[family],s=45)
            ax.axhline(0,color="#777777",ls="--",lw=1)
            ax.set(title=f"{TITLES[task]} · {title}",ylabel="Excess MSE / temporal variation\n(intervened − correct)")
            arm_axis(ax)
    axes[0,0].legend(handles=seed_legend(),frameon=False,ncol=2)
    fig.supxlabel("Positive values mean the correct input predicts better under this intervention. Sensitivity alone and off-support zero-state changes do not establish causal control competence.",fontsize=10)
    save(fig,out,"input-use")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",required=True)
    parser.add_argument("--output")
    args=parser.parse_args()
    path=Path(args.results);out=Path(args.output) if args.output else path.parent
    out.mkdir(parents=True,exist_ok=True)
    result=json.loads(path.read_text())
    if len(result["rows"])!=24 or result["completed_endpoint_count"]!=120:
        raise AssertionError("Plot expects the complete predeclared campaign")
    lookup={(r["task"],r["seed"],r["family"],r["arm"]):r for r in result["rows"]}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titleweight":"regular",
                         "svg.fonttype":"none","savefig.facecolor":"white"})
    for function in [plot_planning,plot_forecast,plot_queries,plot_capacity,plot_interventions]:
        function(result,lookup,out)
    files=sorted(p for p in out.iterdir() if p.suffix in {".png",".svg"})
    manifest=dict(results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files})
    (out/"figures.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output":str(out),"figures":list(manifest["files"])},indent=2))


if __name__=="__main__":
    main()
