from pathlib import Path
import json, os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/efs/prescottsun/SPT-JEPA/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
d=json.loads((r/'results.json').read_text())
labels=['temporal-forecast-s1','temporal-forecast-s2','lewm']
methods=['original','remove_effector','remove_random','block_readout']
pretty={'original':'Original metric','remove_effector':'Delete effector readout span','remove_random':'Delete random rank-3 span','block_readout':'Diagnostic block-xyz metric'}
colors=['#697785','#d17b29','#75a99a','#376fc0']
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
x=np.arange(3)
for ax,group in zip(axes[0],['nonoracle','all']):
 for j,method in enumerate(methods):
  stats=[d[name]['summary'][group]['predicted'][method]['strict_success'] for name in labels]
  y=np.array([s['mean'] for s in stats])*100;lo=np.array([s['lo'] for s in stats])*100;hi=np.array([s['hi'] for s in stats])*100
  ax.bar(x+(j-1.5)*.2,y,width=.19,color=colors[j],label=pretty[method],yerr=np.array([y-lo,hi-y]),capsize=2)
 ax.set_xticks(x,labels);ax.set_ylim(0,80);ax.set_ylabel('Selected sustained success (%)');ax.set_title(('22 nonoracle' if group=='nonoracle' else 'All 39, including recorded-action oracles')+' candidates\nPredicted endpoints; 95% case bootstrap');ax.grid(axis='y',alpha=.2)
axes[0,0].legend(loc='upper left',fontsize=8)
ax=axes[1,0]
for j,method in enumerate(methods[:3]):
 y=[d[name]['probes'][method]['block']['validation']['r2_variance_weighted'] for name in labels]
 bars=ax.bar(x+(j-1)*.25,y,width=.23,color=colors[j],label=pretty[method])
 for bar,val in zip(bars,y):ax.text(bar.get_x()+bar.get_width()/2,val+.012,f'{val:.3f}',ha='center',fontsize=8)
ax.set_xticks(x,labels);ax.set_ylim(0,1.08);ax.set_ylabel('Validation block xyz readout R²');ax.set_title('Object information retained after deletion\nReadouts refit on the same 128 training episodes');ax.grid(axis='y',alpha=.2)
ax=axes[1,1]
for j,(source,name,col) in enumerate([('true','Actual endpoint encoding','#26845f'),('predicted','Predicted endpoint embedding','#376fc0')]):
 y=[d[label]['counterfactual_probes']['nonoracle'][source]['original']['block']['r2_variance_weighted'] for label in labels]
 bars=ax.bar(x+(j-.5)*.3,y,width=.28,color=col,label=name)
 for bar,val in zip(bars,y):ax.text(bar.get_x()+bar.get_width()/2,val+.012,f'{val:.3f}',ha='center',fontsize=9)
ax.set_xticks(x,labels);ax.set_ylim(0,1.08);ax.set_ylabel('Counterfactual block xyz readout R²');ax.set_title('Physical object fidelity on the common nonoracle pool\nTrain-only original readout; evaluated against simulated xyz');ax.legend(loc='upper left',fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('Cube frozen-metric intervention: simple direction deletion does not rescue planning\nDiagnostic privileged-state readouts; frozen models and candidate pool; 25-step first plans',fontsize=14)
for ext in ['png','svg']:fig.savefig(r/f'metric-intervention.{ext}',dpi=170)
plt.close(fig)

def fstat(s,scale=1):
 return f"{s['mean']*scale:.1f} [{s['lo']*scale:.1f}, {s['hi']*scale:.1f}]"
lines=['# Cube metric intervention: a negative simple-fix result and a clearer representation gap','',
'Three jobs exit 0. The analysis uses the existing 50 cases and identical 39-candidate library. Each frozen model fits diagnostic xyz readouts on the same 128 training episodes x 8 frames, with a disjoint 128-episode validation set. Privileged coordinates are used only to diagnose a frozen representation and metric, not to train a policy or propose a deployable algorithm. Results are 25-step first-plan selection, not the original 50-step closed-loop benchmark.','',
'## Integrity','',
'Frozen commit b5227ca; script SHA 87e5bb6961b55c48f8981f5e2cf639e6a9ab227df00165572141c21b2ee973a4. All row episode IDs match the authoritative metadata and split; train and validation episodes/rows are disjoint and selected rows match across all three models. Goal embeddings are recomputed from actual goal images. Recomputed original costs differ by at most 1.26e-7 relative to each case maximum and change zero original selected indices. Projection ranks are exactly 64→61 for temporal and 192→189 for LeWM, with idempotence error below 1.4e-15. The fitted effector readout is numerically annihilated (max residual below 3e-17). Full checks are in audit.json.','',
'## Deleting the fitted effector span does not remove effector information','',
'| Model | Original effector val R² | After effector-span deletion, refit R² | Original block val R² | Block after deletion | Block after random deletion |','|---|---:|---:|---:|---:|---:|']
for name in labels:
 p=d[name]['probes'];v=lambda method,target:p[method][target]['validation']['r2_variance_weighted']
 lines.append('| '+name+' | '+' | '.join(f'{z:.4f}' for z in [v('original','effector'),v('remove_effector','effector'),v('original','block'),v('remove_effector','block'),v('remove_random','block')])+' |')
lines += ['',
'The deletion successfully destroys the specific fitted linear map, yet a newly fitted map recovers nearly all effector information for temporal-forecast. Therefore this is not semantic erasure of the effector, and its weak task effect cannot refute the earlier effector-dominance association. It does reject the simple hypothesis that one removable rank-3 readout span is enough to fix the metric. The apparent redundancy is a property of this linear intervention/readout family; it does not prove a complete factorization theorem.','',
'## Predicted-endpoint selection','',
'Percent sustained success, with 95% paired-case bootstrap intervals. All intervals are conditional on trained weights, the readout fit, and the candidate library.','']
for group in ['nonoracle','all']:
 lines += ['### '+group,'','| Metric | Temporal seed1 | Temporal seed2 | LeWM |','|---|---:|---:|---:|']
 for method in methods:
  lines.append('| '+pretty[method]+' | '+' | '.join(fstat(d[name]['summary'][group]['predicted'][method]['strict_success'],100) for name in labels)+' |')
 lines += ['']
lines += [
'On the nonoracle library, effector-span deletion leaves temporal success exactly at 20%/12%, and random deletion does likewise. On the full library, deletion changes 20→22% and 22→26%; these are one and two paired case changes, with bootstrap intervals including zero. There is no substantial rescue.','',
'Using the diagnostic block-xyz readout metric does not consistently rescue temporal-forecast either: nonoracle predicted selection becomes 10%/18%, and full-library selection becomes 14%/26%. In contrast, LeWM full-library selection changes 40→58%, a paired +18pp interval [6,30]. This gain uses a privileged probe and a library containing recorded-action oracles; it is not a deployable benchmark gain. On the nonoracle library its gain is only 22→24%.','',
'## Actual endpoint information and predicted physical fidelity','',
'| Model | Nonoracle actual-encoding block R² | Nonoracle predicted-embedding block R² | Nonoracle actual-encoding effector R² | Nonoracle predicted-embedding effector R² |','|---|---:|---:|---:|---:|']
for name in labels:
 c=d[name]['counterfactual_probes']['nonoracle'];vals=[c[src]['original'][target]['r2_variance_weighted'] for src,target in [('true','block'),('predicted','block'),('true','effector'),('predicted','effector')]]
 lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines += ['',
'Temporal encodes and forecasts the effector extremely well. LeWM preserves substantially more linearly accessible block-position information, both on held-out recorded frames and on counterfactual executed candidates. This strengthens a representation-allocation explanation beyond merely a wrong distance scale. It does not establish that all missing object information is absent: a stronger nonlinear probe or a larger fit set could recover more, and the models differ in state width and training budget.','',
'Actual-embedding selectors use the observed simulated endpoint; predicted selectors use the model rollout. On all 39 candidates, the encoded-goal metrics nearly always select the recorded endpoint, explaining their near-100% endpoint-pose score but only ~56% hold success. Replacing the encoded goal by privileged physical xyz can reduce performance because readout calibration error becomes relevant; that auxiliary control must not be pooled with the encoded-goal metric.','',
'## Updated interpretation','',
'The earlier association remains: temporal goal geometry is strongly effector-centered. The new intervention rules out a simple three-direction subtraction fix, while revealing that object information is less accessible than in LeWM. A method that improves its own latent rollout error may primarily become better at forecasting the factors it already allocates most representation to. This suggests separating three questions: which factors survive encoding, how their distances express a goal, and how accurately action-conditioned predictions preserve those factors. None can be certified by denoising state sensitivity alone.','',
'No new training method is established by this diagnostic. The exact causal contribution of the training objective to object underrepresentation remains open; the deletion intervention acts on the metric, not on learning or physical scene factors. Additional planning duration or full MPC can also change hold outcomes.','',
'![Frozen metric interventions](metric-intervention.png)','',
'Full numeric tables, train/validation/counterfactual probes, and paired intervals: results.json. Integrity: audit.json. Exact per-case intervention records and fitted matrices remain in /efs/prescottsun/SPT-JEPA/runs/mechanism_metric_20261030.','']
(r/'RESULTS.md').write_text('\n'.join(lines))
print('report and figure written')
