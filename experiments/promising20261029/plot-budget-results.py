"""Replot frozen budget reports with readable scientific labels."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument("--results",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 v=json.loads(Path(a.results).read_text());d=Path(a.output);d.mkdir(parents=True,exist_ok=False)
 methods=list(v["summary"]);x=np.arange(len(methods));fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,key,title in zip(axes,["primary","fresh"],["Matched primary planning cases","Additional evaluation cases"]):
  for i,seed in enumerate(["1","2"]):
   bars=ax.bar(x+(i-.5)*.35,[v["summary"][m][seed][key] for m in methods],.32,label="Training seed "+seed)
   ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
  ax.set_xticks(x,methods,rotation=25,ha="right");ax.set_ylim(0,107);ax.set_title(title);ax.spines[["top","right"]].set_visible(False)
 axes[0].set_ylabel("P0 success (%)");axes[0].legend(loc="upper left",fontsize=8)
 fig.text(.5,.01,f"Matched {v['training_updates']:,} updates; 51.1M parameters; two training seeds. Broader stability remains under evaluation.",ha="center",fontsize=9)
 fig.tight_layout(rect=(0,.04,1,1))
 for ext in ["png","svg"]:fig.savefig(d/("followup-planning."+ext),dpi=180,bbox_inches="tight")
 for f in d.glob("*.svg"):f.write_text("\n".join(x.rstrip() for x in f.read_text().splitlines()).rstrip()+"\n")
if __name__=="__main__":main()
