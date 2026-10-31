"""Plot the audited2000 coordinate control; no fitting or new model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    src=Path(__file__).parent/'results/basis2000/summary.json'
    data=json.loads(src.read_text());rows=data['rows'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    colors={'identity':'#24738c','orthogonal':'#c46735'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,7.8),gridspec_kw={'height_ratios':[1,1.25]})
    for col,task in enumerate(('tworoom','cube')):
        top,bottom=axes[:,col]
        for i,arm in enumerate(('identity','orthogonal')):
            selected=sorted([r for r in rows if r['task']==task and r['arm']==arm],key=lambda r:r['seed']);assert len(selected)==2
            v=np.array([r['rollout'] for r in selected]);top.bar(i,v.mean(),width=.56,color=colors[arm],alpha=.8)
            top.scatter(i+np.array([-.07,.07]),v,color='#192a35',s=29,zorder=3)
            for group,case in enumerate(('primary','fresh')):
                v=np.array([r['values'][case]['success_rate'] for r in selected]);x=group+(i-.5)*.32
                bottom.bar(x,v.mean(),width=.30,color=colors[arm],alpha=.8,label=arm if group==0 else None)
                bottom.scatter(x+np.array([-.035,.035]),v,color='#192a35',s=29,zorder=3)
                bottom.text(x,v.max()+4,f'{v.mean():g}',ha='center',fontsize=10)
        top.set_title('TwoRoom' if task=='tworoom' else 'Cube',fontweight='bold',fontsize=13)
        top.set_xticks([0,1],['Original local','Orthogonal rotation']);top.set_ylim(0,.45)
        top.set_ylabel('Causal rollout / initial-copy MSE\n(lower is better)');top.grid(axis='y',alpha=.15);top.set_axisbelow(True)
        bottom.set_xticks([0,1],['Primary50','Fresh50']);bottom.set_ylim(0,100);bottom.set_ylabel('Strict planning success (%)')
        bottom.grid(axis='y',alpha=.15);bottom.set_axisbelow(True)
        c=next(x for x in data['comparison_summaries'] if x['task']==task and x['case_set']=='fresh')
        lo,hi=c['shared_case_bootstrap_ci95_pp'];delta=c['mean_difference_pp']
        bottom.set_title(f'Fresh rotation effect: {delta:+g} pp [{lo:+g}, {hi:+g}]',fontsize=10,pad=12)
    axes[1,1].legend(handles=[Patch(facecolor=colors[a],alpha=.8,label=n) for a,n in
        [('identity','Original local'),('orthogonal','Orthogonal rotation')]],frameon=False,loc='upper right',fontsize=9)
    fig.suptitle('Preserved observation geometry, better factual prediction,\nyet worse TwoRoom control',x=.07,y=.98,ha='left',fontweight='bold',fontsize=16)
    fig.text(.07,.875,'2000 updates; fixed orthogonal map; same model initialization, data and planner.',fontsize=10,color='#40515c')
    fig.text(.07,.045,'Bars: two-seed means. Dots: individual training seeds. 95% intervals: paired case bootstrap, conditional on these seeds.\nFresh cases are different goal pairs from the same validation pool. This is a mechanism control, not a new algorithm.',fontsize=9,color='#40515c')
    fig.subplots_adjust(top=.81,bottom=.15,left=.09,right=.97,hspace=.47,wspace=.30)
    fig.savefig(out/'basis2000-mechanism.png',dpi=180);fig.savefig(out/'basis2000-mechanism.svg');plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_model_calls=0),indent=2)+'\n')
if __name__=='__main__':main()
