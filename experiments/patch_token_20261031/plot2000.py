"""Plot only verified existing cold-rollout errors; no model or data calls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=json.loads(Path(a.summary).read_text());assert s['passed']and s['all72_actual_exit0']
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
colors={'flat_reference':'#555555','identity':'#0072B2','orthogonal':'#D55E00'}
labels={'flat_reference':'Flat1024 reference','identity':'Spatial token','orthogonal':'Orthogonal token'}
fig,axes=plt.subplots(1,2,figsize=(7.5,3.2),sharey=True)
for ax,task,title in zip(axes,['tworoom','cube'],['TwoRoom','Cube']):
 for arm in colors:
  for seed in [1,2]:
   r=next(r for r in s['rows']if(r['task'],r['arm'],r['seed'])==(task,arm,seed))
   values=r['energy']['clean_rollout']['per_horizon']['mse_over_zero'];assert len(values)==7
   ax.plot(range(1,8),values,color=colors[arm],linestyle='-'if seed==1 else'--',marker='o'if seed==1 else's',markersize=3,linewidth=1.5)
 ax.axhline(1,color='#777777',linewidth=.8,linestyle=':')
 ax.set(title=title,xlabel='Predicted frame offset',xticks=range(1,8),ylim=(0,1.5))
 ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('MSE / fixed-zero MSE')
handles=[Line2D([0],[0],color=colors[k],label=labels[k],linewidth=1.8)for k in colors]
handles += [Line2D([0],[0],color='black',linestyle=ls,label=f'Seed {i}',marker=m,markersize=3)for i,ls,m in[(1,'-','o'),(2,'--','s')]]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=5,frameon=False,fontsize=8)
fig.suptitle('2000 updates: identical observed states, cold causal rollout',fontsize=11)
fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(out/'cold-horizon.png',dpi=220);fig.savefig(out/'cold-horizon.pdf');plt.close(fig)
(out/'figure-provenance.json').write_text(json.dumps(dict(summary=str(a.summary),summary_sha256=hashlib.sha256(Path(a.summary).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='All tasks/arms/seeds;64 fixed windows, existing cached errors only. One frame offset is5 raw environment steps. Cold rollout differs from native history3 planning.'),indent=2)+'\n')
