"""Figures derived only from the committed short-adaptation numerical results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    data=json.loads(Path(a.results).read_text());out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    methods=list(data["protocol"]["methods"]);x=np.arange(len(methods))
    fig,ax=plt.subplots(figsize=(14,6))
    for i,seed in enumerate([str(s) for s in data["protocol"]["training_seeds"]]):
        bars=ax.bar(x+(i-(len(data["protocol"]["training_seeds"])-1)/2)*.18,[data["summaries"][m][seed]["p0_mean"] for m in methods],.16,label="Training seed "+seed)
        ax.bar_label(bars,fmt="%.0f",padding=3,fontsize=8)
    ax.set_xticks(x,methods,rotation=35,ha="right");ax.set_ylabel("P0 success (%)");ax.set_ylim(0,105)
    ax.set_title("600-update adaptations from the same two-epoch models")
    ax.legend(loc="upper left");ax.spines[["top","right"]].set_visible(False)
    fig.text(.5,.01,"51.1M parameters; five settings including controls; four existing training seeds. Adaptation only.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png","svg"]:fig.savefig(out/("actionrank-short-planning."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    seeds=[str(s) for s in data["protocol"]["training_seeds"]]
    candidates=methods[1:];fig,ax=plt.subplots(figsize=(10,7));y=np.arange(len(candidates))
    for i,seed in enumerate(seeds):
        rows=[data["paired_effects"][m][seed]["p0_delta_pp"] for m in candidates]
        values=np.array([r["mean_delta"] for r in rows]);intervals=np.array([r["ci95"] for r in rows])
        ax.errorbar(values,y+(i-(len(seeds)-1)/2)*.15,xerr=np.array([values-intervals[:,0],intervals[:,1]-values]),fmt="o",capsize=3,label="Training seed "+seed)
    ax.axvline(0,color="black",lw=1);ax.set_yticks(y,candidates);ax.invert_yaxis()
    ax.set_xlabel("P0 difference from matched original continuation (percentage points)")
    ax.set_title("Sample-local action ranking: paired planning effects")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.15),ncol=2,frameon=False);ax.spines[["top","right"]].set_visible(False)
    fig.text(.5,.01,"95% paired episode-cluster bootstrap; conditional on fixed weights. Exploratory comparisons.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.16,1,1))
    for ext in ["png","svg"]:fig.savefig(out/("actionrank-short-paired-effects."+ext),dpi=180,bbox_inches="tight")
    for path in out.glob("*.svg"):path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines()).rstrip()+"\n")
if __name__=="__main__":main()
