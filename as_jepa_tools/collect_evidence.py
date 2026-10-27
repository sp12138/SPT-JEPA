"""Read-only collection of campaign evidence into small review artifacts."""
import argparse,json,time
from pathlib import Path
import numpy as np


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    campaign=Path(a.campaign);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    probes=[];plans=[];generations=[];swaps=[]
    for path in campaign.glob("probe-*/results/**/probes.json"):
        r=read(path)
        if r:probes.append(dict(path=str(path),**r))
    for path in campaign.glob("**/metrics.json"):
        if "/checkpoint/" in str(path):continue
        r=read(path)
        if not r:continue
        if r.get("split")=="heldout" and "episode_successes" in r.get("metrics",{}):
            plans.append(dict(path=str(path),**r))
        elif "h7_agent_ratio" in r:generations.append(dict(path=str(path),**r))
    for path in campaign.glob("**/swaps.json"):
        r=read(path)
        if r:swaps.append(dict(path=str(path),**r))
    paired={}
    for r in plans:
        key=(r["seed"],len(r["episodes"]))
        cases=list(zip(r["episodes"],r["start_steps"]))
        if key in paired and paired[key]!=cases:raise AssertionError(f"Unmatched planner cases: {r['path']}")
        paired[key]=cases
    index=dict(collected_at=time.time(),campaign=str(campaign),probes=probes,plans=plans,generations=generations,swaps=swaps)
    (out/"evidence.json").write_text(json.dumps(index,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":170})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    # Explicit checkpoint selections preserve comparison scope.
    def pick(records,fragment):
        rows=[r for r in records if fragment in r["path"]]
        return rows[-1] if rows else None
    selections=[
        ("L0 @25k","planning-batch2-dev13/results/L0-standard-seed0-step0025000"),
        ("L1 @25k","quality-planning-batch3-dev15/planning/L1-standard-seed0-model-step0025000"),
        ("L1 BN(64) @22k","quality-planning-batch3-dev15/planning/L1-standard-bn-dev2-model-step0022000"),
        ("L2 beta4 @8k","planning-batch2-dev13/results/L2-standard-b4-seed0-step0008000"),
        ("L2 beta.25 @18k","quality-planning-batch3-dev15/planning/L2-standard-calibrated-b025-resume-dev15-model-step0018000"),
    ]
    labels=[];scores=[];errors=[]
    for label,fragment in selections:
        r=pick(plans,fragment)
        if not r:continue
        result=np.array(r["metrics"]["episode_successes"],dtype=float);n=len(result);rate=result.mean();z=1.96
        center=(rate+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        labels.append(label);scores.append(rate*100);errors.append([(rate-center+half)*100,(center+half-rate)*100])
    ax=axes[0];ax.barh(labels,scores,color=["#375c93" if "L0" in x else "#4e9a89" for x in labels])
    if errors:ax.errorbar(scores,labels,xerr=np.array(errors).T,fmt="none",ecolor="#333333",capsize=3)
    ax.set(xlim=(0,100),xlabel="Success (%)",title="P0 planning: same 50 held-out cases");ax.invert_yaxis()
    for i,value in enumerate(scores):ax.text(min(94,value+3),i,f"{value:.0f}",va="center")
    ax.text(0,-.22,"One training seed; Wilson 95% intervals across cases.",transform=ax.transAxes,fontsize=8)
    fp_selections=[
        ("beta4, std1 @8k","L2-standard-calibrated1-b4-seed0-step0008000"),
        ("beta4, std.1 @12k","L2-standard-calibrated-b4-resume-dev15-step0012000"),
        ("beta0 @16k","L2-standard-b0-recovered-dev2-step0016000"),
        ("L1 BN(64) @14k","L1-standard-bn-dev2-step0014000"),
    ]
    ax=axes[1];labels=[];bf=[];fp=[]
    for label,fragment in fp_selections:
        b=pick(probes,"probe-batch2-dev15/results/"+fragment+"/mean/")
        f=pick(probes,"probe-precision-dev15/results/"+fragment+"/mean-fp32/")
        if b and f:labels.append(label);bf.append(b["feature_std"]);fp.append(f["feature_std"])
    y=np.arange(len(labels));ax.barh(y-.18,bf,.35,label="BF16",color="#e4a35b");ax.barh(y+.18,fp,.35,label="FP32",color="#375c93")
    ax.set_yticks(y,labels);ax.set(xscale="log",xlabel="Mean state standard deviation",title="Same weights: numerical variation");ax.invert_yaxis();ax.legend(frameon=False)
    ax.text(0,-.22,"Large BF16 variance can include rounding artifacts.",transform=ax.transAxes,fontsize=8)
    ax=axes[2];labels=[];anchor=[];state=[]
    for fragment,label in [("L2-color-calibrated1-b025-seed0-step0020000","With anchor"),("L2-color-noanchor-calibrated1-b025-seed0-step0020000","Without anchor")]:
        r=pick(swaps,fragment)
        if r:labels.append(label);anchor.append(r["anchor_style_transfer_projection"]);state.append(r["state_style_transfer_projection"])
    x=np.arange(len(labels));ax.bar(x-.18,anchor,.35,label="Change anchor",color="#375c93");ax.bar(x+.18,state,.35,label="Change state",color="#4e9a89")
    ax.set_xticks(x,labels);ax.set(ylim=(-.03,1.1),ylabel="Fraction of donor background palette transferred",title="Colour role: matched-noise interventions");ax.legend(frameon=False)
    ax.text(0,-.22,"128 cases, 50 Euler steps, one seed; dynamics imperfect.",transform=ax.transAxes,fontsize=8)
    fig.savefig(out/"interim-evidence.png",bbox_inches="tight");fig.savefig(out/"interim-evidence.pdf",bbox_inches="tight");plt.close(fig)
    print(json.dumps(dict(probes=len(probes),heldout_plans=len(plans),generation_records=len(generations),swaps=len(swaps),output=str(out)),indent=2))


if __name__=="__main__":main()
