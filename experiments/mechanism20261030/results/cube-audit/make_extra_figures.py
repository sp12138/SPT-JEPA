from pathlib import Path
import json, os
os.environ.setdefault('MPLCONFIGDIR','/efs/prescottsun/SPT-JEPA/cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
d=json.loads((r/'results.json').read_text());labels=list(d['summary'])
fig,ax=plt.subplots(figsize=(12,5.2),layout='constrained');x=np.arange(len(labels))
for j,(key,label,color) in enumerate([('effector_encoded_partial_rank','Effector, controlling cube pose','#346cbb'),('position_encoded_partial_rank','Cube position, controlling effector + orientation','#d07b25')]):
 s=[d['summary'][n]['nonoracle'][key] for n in labels];mean=np.array([v['mean'] for v in s]);lo=np.array([v['lo'] for v in s]);hi=np.array([v['hi'] for v in s]);ax.bar(x+(j-.5)*.34,mean,width=.32,color=color,label=label,yerr=np.array([mean-lo,hi-mean]),capsize=3)
ax.axhline(0,color='#666',linewidth=.8);ax.set_ylim(-.12,1.05);ax.set_xticks(x,labels,rotation=20,ha='right');ax.set_ylabel('Mean per-case partial Spearman correlation');ax.legend(loc='upper left',bbox_to_anchor=(0,-.28),ncol=2);ax.grid(axis='y',alpha=.2);ax.set_title('Which physical factor drives actual latent goal distance?\n22 nonoracle candidates per case; 50 paired cases; 95% case bootstrap')
for ext in ['png','svg']:fig.savefig(r/f'goal-metric-factors.{ext}',dpi=160,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
p=json.loads((r/'pose-hold-decomposition.json').read_text())['rates_percent'];labels=list(p)
fig,ax=plt.subplots(figsize=(13,5.2),layout='constrained');x=np.arange(len(labels))
for j,(key,label,color) in enumerate([('final_pos','Endpoint position','#8797a7'),('final_pose','Endpoint pose','#26845f'),('any_pose','Any-step pose','#e4a13b'),('hold_success','Sustained strict success','#346cbb')]):
 ax.bar(x+(j-1.5)*.19,[p[n][key] for n in labels],width=.18,label=label,color=color)
ax.set_xticks(x,labels,rotation=20,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Cases passing (%)');ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(0,-.28),ncol=4);ax.set_title('Each model\'s own CEM first plan: position, orientation and hold\n25 raw steps; recorded actions are an oracle reference, not a learned controller')
for ext in ['png','svg']:fig.savefig(r/f'first-plan-pose-hold.{ext}',dpi=160,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
