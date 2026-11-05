"""Standalone scientific figures from the frozen numerical evidence only."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(evidence,output):
    output.mkdir(parents=True,exist_ok=False)
    data=json.loads((evidence/'analysis.json').read_text())['tables']
    tasks=['bin-picking-v3','pick-place-v3','push-v3','drawer-open-v3','door-open-v3','peg-insert-side-v3']
    names=['Bin picking','Pick place','Push','Drawer','Door','Peg insert']
    fig,axes=plt.subplots(2,1,figsize=(11.8,7.8),sharex=True,sharey=True,layout='constrained')
    colors=['#10847e','#3869c7'];x=np.arange(len(tasks));width=.34
    for ax,cohort,title in zip(axes,['main4','fresh'],[
            'Exploratory main cohort: 4 independently trained CNNs, 20 physical cases per task',
            'Fresh physical cases: 2 predeclared frozen CNNs, 10 new cases per task']):
        group=data[cohort]
        for i,arm in enumerate(['local-mse','local-flow']):
            values=[group[arm]['tasks'][t]['success_percent'] for t in tasks]
            label=('MSE' if i==0 else 'Flow')+f' (six-task mean {group[arm]["manip_macro_percent"]:.1f}%)'
            bars=ax.bar(x+(i-.5)*width,values,width,color=colors[i],label=label)
            ax.bar_label(bars,labels=[f'{v:g}'for v in values],padding=3,fontsize=10)
        ax.set_ylim(0,117);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Actual success (%)')
        ax.set_title(title,loc='left',fontsize=11,pad=12)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.legend(loc='upper center',ncols=2,frameon=False,fontsize=10)
    axes[-1].set_xticks(x,names)
    fig.suptitle('Manipulation from RGB and logged actions, trained from scratch',fontsize=15)
    fig.supxlabel('TF and native LeWM at 20 epochs: 0% on all six tasks in both cohorts.\n'
                  '500 native steps, any success; independent LeRobot data. Training and compute budgets differ.',fontsize=10)
    fig.savefig(output/'manipulation.png',dpi=180);fig.savefig(output/'manipulation.pdf');plt.close(fig)
    with(output/'success-rates.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['cohort','method','task','successes','episodes','success_percent'])
        for cohort,groups in data.items():
            for arm,group in groups.items():
                for task,values in group['tasks'].items():writer.writerow([cohort,arm,task,values['successes'],values['episodes'],values['success_percent']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.evidence,a.output)
