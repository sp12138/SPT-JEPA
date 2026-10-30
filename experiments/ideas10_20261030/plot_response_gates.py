"""Scientific summary: every seed, native task score, exact initialization audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--initial-results');args=p.parse_args()
    data=json.loads(Path(args.results).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    initial=json.loads(Path(args.initial_results).read_text()) if args.initial_results else None
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.3),gridspec_kw={'width_ratios':[1.1,1.1,1.]})
    colors=['#4c78a8','#f28e2b','#59a14f'];operators=['direct_relative','annihilator_relative','random_relative']
    for ax,task,ref in zip(axes[:2],['tworoom','cube'],[7.,12.]):
        labels=[];heights=[]
        for oi,operator in enumerate(operators):
            for ri,route in enumerate(['forecast','mixed']):
                i=2*oi+ri;rs=[r for r in data['rows']if r['task']==task and r['method']==operator and r['route']==route]
                assert len(rs)==2
                ys=[r['success_rate']for r in sorted(rs,key=lambda r:r['seed'])]
                ax.bar(i,np.mean(ys),color=colors[oi],alpha=.55 if ri==0 else .95,width=.72)
                ax.scatter([i-.12,i+.12],ys,color='#202020',s=23,zorder=3)
                heights.extend(ys);labels.append(['Direct','Action-null','Random-null'][oi]+'\n'+('Forecast'if ri==0 else'Mixed'))
        ax.axhline(ref,color='#666666',ls='--',lw=1.2,label='TF600 reference mean')
        if initial:
            rates=[r['result']['metrics']['success_rate']for r in initial['rows']if r['task']==task]
            assert len(rates)==2 and rates[0]==rates[1]
            ax.axhline(rates[0],color='#888888',ls=':',lw=1.5,label='Untrained carry')
        ax.set_ylim(0,min(100,max(20,max(heights+[ref])+5)));ax.set_xticks(range(6),labels,fontsize=8)
        ax.set_ylabel('Strict success (%)');ax.set_title(task.title()+' | 50 fixed cases')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.legend(fontsize=8,loc='upper left')
    ax=axes[2];audits=data['preflight']['audits']
    for ti,task in enumerate(['tworoom','cube']):
        values=[]
        for route in ['forecast','mixed']:
            matches=[a for a in audits if Path(a['run']).name==f'{task}-annihilator_relative-{route}-seed1']
            assert len(matches)==1
            zero=next(q for q in matches[0]['query_only_gradients']if q['step']==0)
            values.append(zero['gradient_norms']['encoder'])
        ax.bar([3*ti,3*ti+1],values,color=['#b9cadb','#4c78a8'],width=.75)
    ax.set_yscale('log');ax.set_ylabel('Encoder gradient norm (query only)')
    ax.set_xticks([0,1,3,4],['TwoRoom\nForecast','TwoRoom\nMixed','Cube\nForecast','Cube\nMixed'],fontsize=8)
    ax.set_title('Initialization | action-null query')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Grounded response queries: bounded 600-update mechanism gate',fontsize=14,y=.99)
    fig.text(.02,.005,'Dots: both training seeds. Dashed TF references have no extra query head. All candidate arms share the head and native planner.\nMixed = half observed-code loss + half causal-forecast loss; future observations never enter the rollout. Initialization gradients do not establish learning or task gains.',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.94]);fig.savefig(out/'relative-gates.png',dpi=180);fig.savefig(out/'relative-gates.svg');plt.close(fig)


if __name__=='__main__':main()
