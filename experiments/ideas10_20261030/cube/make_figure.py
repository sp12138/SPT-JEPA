"""Render CPU mechanism falsification results; no simulator or model execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent
r=json.loads((p/'frozen-regret.json').read_text())['models']
fig,axs=plt.subplots(1,3,figsize=(15,4.4))
colors=['#386ebc','#df922a','#25805d']
x=np.arange(2);w=.34
axs[0].bar(x-w/2,[np.exp(-2),1],w,label='Area-average distortion',color=colors[0])
axs[0].bar(x+w/2,[np.exp(-1),np.exp(-1)],w,label='Worst relative distortion',color=colors[1])
axs[0].set_xticks(x,['Large field (99%)','Small object (1%)'])
axs[0].set_ylabel('Normalized distortion')
axs[0].set_title('Same 1-nat rate: coverage changes allocation')
axs[0].legend(fontsize=8)
labels=['temporal-forecast-s1','temporal-forecast-s2','lewm'];x=np.arange(3)
for offset,k,c,desc in [(-.17,'random',colors[0],'Random-family mean regret'),(.17,'winner_nonoracle',colors[1],'Chosen nonoracle regret')]:
 axs[1].bar(x+offset,[100*r[l]['optimism'][k]['positive_fraction'] for l in labels],.34,color=c,label=desc)
axs[1].set_xticks(x,['TF seed 1','TF seed 2','LeWM'])
axs[1].set_ylim(0,110);axs[1].set_ylabel('Cases with positive regret (%)')
axs[1].set_title('Selection changes the error distribution')
axs[1].legend(fontsize=8)
for offset,k,c,desc in [(-.17,'original',colors[0],'Original cost'),(.17,'signed_regret_ridge',colors[1],'Signed-regret ridge, case cross-fit')]:
 axs[2].bar(x+offset,[r[l]['crossfit']['nonoracle'][k]['success_pct'] for l in labels],.34,color=c,label=desc)
axs[2].set_xticks(x,['TF seed 1','TF seed 2','LeWM'])
axs[2].set_ylim(0,36);axs[2].set_ylabel('Sustained success in 25-step library (%)')
axs[2].set_title('Simple correction does not rescue TF')
axs[2].legend(fontsize=8)
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Theory toy and exploratory frozen-bank checks — no new training or benchmark claim',fontsize=13)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(p/f'mechanism-falsification.{ext}',dpi=160,bbox_inches='tight')
