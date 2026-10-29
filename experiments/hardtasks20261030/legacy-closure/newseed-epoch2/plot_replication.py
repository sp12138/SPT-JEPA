from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;data=json.loads((root/"replication-results.json").read_text());summary=data["summary"];seeds=data["training_seeds"];methods=["original","lower-beta","temporal-forecast"];x=np.arange(len(seeds));fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,key,title in zip(axes,["primary","additional"],["Primary planning cases","Additional planning cases"]):
 for i,m in enumerate(methods):
  bars=ax.bar(x+(i-1)*.25,[summary[m][s][key] for s in seeds],.23,label=m);ax.bar_label(bars,fmt="%.0f",padding=3,fontsize=8)
 ax.set_xticks(x,["Training seed "+s for s in seeds]);ax.set_ylim(0,105);ax.set_title(title);ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("P0 success (%)");handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,fontsize=9)
fig.text(.5,.015,"Matched24,982 updates;51.1M parameters; all training seeds retained. Two-epoch evidence, not final mature stability.",ha="center",fontsize=9);fig.tight_layout(rect=(0,.05,1,.91))
for ext in ["png","svg"]:fig.savefig(root/("replication-planning."+ext),dpi=180,bbox_inches="tight")
