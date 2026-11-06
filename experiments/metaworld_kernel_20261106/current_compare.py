"""Read-only actual-trace verification and paired physical-case bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA/runs')


def load(name,arm,step=40000):
    root=BASE/name;training=(root/'training-plan.json').exists();plan=json.loads((root/('training-plan.json'if training else'plan.json')).read_text());rows={};evidence=[]
    for r in plan['recipes']:
        if r['arm']!=arm:continue
        path=Path(r['run']if training else r['output'])/(f'eval-step{step:07d}'if training else'pilot')/'summary.json'
        if not path.exists():return None
        value=json.loads(path.read_text());assert value['passed']and value['model_unchanged']
        evidence.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=value['model_sha256']))
        for row in value['rows']:
            assert hashlib.sha256(Path(row['trace']).read_bytes()).hexdigest()==row['trace_sha256']
            key=(r['seed'],row['task'],row['seed']);assert key not in rows;rows[key]=row
    assert rows
    return rows,evidence


def compare(left,right,scope):
    a,ae=left;b,be=right;assert a.keys()==b.keys()
    assert all(a[k]['case_sha256']==b[k]['case_sha256']for k in a)
    results={};rng=np.random.default_rng(20261107)
    for label,tasks in [('all_tasks',sorted({k[1]for k in a})),('manipulation',sorted({k[1]for k in a}-{'reach-v3','reach-wall-v3'}))]:
        delta=[];bootstrap=np.zeros(10000);table={};keys=[k for k in a if k[1]in tasks]
        for task in tasks:
            case_seeds=sorted({k[2]for k in keys if k[1]==task});models=sorted({k[0]for k in keys})
            diff=np.array([np.mean([int(a[(m,task,s)]['success'])-int(b[(m,task,s)]['success'])for m in models])for s in case_seeds])
            delta.append(float(diff.mean()));bootstrap+=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)/len(tasks)
            table[task]=dict(left_successes=sum(a[k]['success']for k in keys if k[1]==task),right_successes=sum(b[k]['success']for k in keys if k[1]==task),episodes=len(case_seeds)*len(models),physical_cases=len(case_seeds))
        results[label]=dict(left_percent=100*np.mean([a[k]['success']for k in keys]),right_percent=100*np.mean([b[k]['success']for k in keys]),
            delta_pp=100*np.mean(delta),physical_case_cluster_ci95_pp=(100*np.quantile(bootstrap,[.025,.975])).tolist(),episodes=len(keys),
            trained_seeds=sorted({k[0]for k in keys}),training_seed_delta_pp={str(m):100*np.mean([int(a[k]['success'])-int(b[k]['success'])for k in keys if k[0]==m])for m in sorted({k[0]for k in keys})},tasks=table)
    return dict(scope=scope,paired_case_and_trace_hashes_pass=True,results=results,evidence_left=ae,evidence_right=be,
        limitation='CIconditionsonthelistedtrainedlearners;resamplesphysicalcasesWITHINtask,keepsalllearnersinonecasecluster.Doesnotestimatepopulationtrainingvariance.Tasksspecified,notrandomtaskdraws.')


def run(a):
    comparisons=[
        ('native_world_vs_static_1000',('metaworld_native_control1000_20261107','world-native4',40000),('metaworld_native_control1000_20261107','static-native4',40000),'ExpandedEXISTING1000physicalcases/50tasks,2trainedactors;notuntouchedconfirmationforthenewmethod.Same4DNNs/5policycalls/realsensor5,own20kFcostextra.'),
        ('native_mseworld_vs_flowworld_full8',('metaworld_native_clock_controls_20261107','mse-world-native4',40000),('metaworld_current_control_full_20261107','world-native4',40000),'SAMEown20kworldNNs/originaldata/currenta_t4Dtargets/40kactorbudget/realsensor5/physicalcases. Deterministicregressionvs12stepflow,inferencecomputeDIFFERENT;noGaussianactiondrawsinMSE.'),
        ('native_mseworld_vs_mseheld_full8',('metaworld_native_clock_controls_20261107','mse-world-native4',40000),('metaworld_one_native_mse_20261107','all-executed',40000),'EXACTsame4DMSEactorNNs/goals/physicalcases/realsensor5. Modelfeedbackaddsown20kworld/4forecasts/5policycallsversus1.Noa_t+1labels;noGaussianactiondrawsinMSE.'),
        ('native_mseworld_vs_copy_full8',('metaworld_native_clock_controls_20261107','mse-world-native4',40000),('metaworld_native_mse_copy_20261107','mse-copy-native4',40000),'EXACTsameMSEactor/worldNNs/128->224processing/5deterministicpolicycalls/realsensor5. Onlyown-actionworldevolutiondisabled. No newSGD/stochasticactiondraws.'),
        ('native_mseworld_vs_flowworld_fifty',('metaworld_native_mse_world50_20261107','mse-world-native4',40000),('metaworld_current_control50_20261107','world-native4',40000),'50task4caseSCREEN:originaldata/a_t4Dlabels/40kCNNs/own20kF/physicalcases/sensor5matched. MSE1fieldpassvsFlow12percall;not equalinferenceFLOPs.'),
        ('native_world_vs_static_full8',('metaworld_current_control_full_20261107','world-native4',40000),('metaworld_current_control_full_20261107','static-native4',40000),'EXACTsamea_t4DNNs/5ownGaussianpolicydraws/realsensor5/physicalcases/500native. Worldadvances4OWNpredictedimages;staticreusesactual224RGB.Worldown20kpretrainandforecastcomputeadditional.Noa_t+1labels/strongpriors.'),
        ('native_world_vs_copy_full8',('metaworld_current_control_full_20261107','world-native4',40000),('metaworld_current_control_full_20261107','copy-native4',40000),'EXACTsameactor/worldNNs/128->224intermediateprocessing/5draws/realsensor5/physicalcases.Bothloadsame20kforwardNN;onlyworldadvancesbeliefusingOWNcontrols.0SGD;notfutureactionreferences.'),
        ('native_world_vs_held_full8',('metaworld_current_control_full_20261107','world-native4',40000),('metaworld_one_native_20261107','all-executed',40000),'EXACTsamea_t4Dactorweights/goals/physicalcases/realsensor5.Worldsequentiallygenerates5OWNnativecontrols;holdrepeatsonecurrentdraw5times.Extra5xpolicycalls/world20kpretrainexplicit.'),
        ('native_world_vs_static_fifty',('metaworld_current_control50_20261107','world-native4',40000),('metaworld_current_control50_20261107','static-native4',40000),'50task4caseSCREEN,2trainedlearners.SAMEa_t4DNNs/5ownpolicydraws/realsensor5/physicalcases.Worldown20kand4forecastsadditional;noactualintermediateRGB/futureactionlabels.'),
        ('native_world_vs_macro_best_full8',('metaworld_current_control_full_20261107','world-native4',40000),('metaworld_current_goal_20261107','directed-flow',40000),'Samephysicalcases/realsensor5/500budget;DIFFERENTtargets:ONLYa_t4Dvs5currentnativecontrols20D. Nativeworld40kactor+20kworld/5calls,macro40k/1call.Notmatchedinformation/budget/FLOPs;reportwhetherbestscoreactuallyexceeded.'),
        ('native_world_vs_zero_full8',('metaworld_current_control_full_20261107','world-native4',40000),('metaworld_world_action_full_20261107','world-zero',40000),'SAMENNs/realobservationcadence/actionlabels/5owndraws;negativeonlysetsOWNIMAGINEDforwardcontrols0(notactualactuation).Same20physicalcases/task.0SGD.'),
        ('native_world_vs_negate_full8',('metaworld_current_control_full_20261107','world-native4',40000),('metaworld_world_action_full_20261107','world-negate',40000),'SAMENNs/realobservationcadence/actionlabels/5owndraws;negativeonlynegatesOWNIMAGINEDforwardcontrols(notactualactuation).Same20physicalcases/task.0SGD.'),
        ('directed_vs_absolute_confirm1000',('metaworld_confirm50_20261107','directed',40000),('metaworld_confirm50_20261107','absolute',40000),'Same2000episodes/50tasks/20NEWphysicalcasespertaskx2learners;newcasebankcommittedbeforeconfirmationoutcomes.Original40kpairedlearners/sensor5/500native.No newSGD;notpublicJEPA-WMs/MT50setting.'),
        ('directed_vs_absolute_newseeds40k',('metaworld_relation_long_20261107','directed',40000),('metaworld_relation_long_20261107','absolute',40000),'FourNEWindependentscratchseedpairs3..6,8tasksx20physicalcases;withineachpairoriginaldata/current5-native20Dtargets/sampler/initialCNN-decoder/budgetmatch.120kLRhorizon,notold40khorizon;all4learnerskeptineachphysicalcasebootstrapcluster.'),
        ('directed_vs_absolute_40k',('metaworld_current_goal_20261107','directed-flow',40000),('metaworld_current_inverse_20261107','episodes',40000),'Same320episodes/20physicalcasespertaskx2learners,CNN/decoderinitialstates/labels/sampler/budget;earlyorderedpairingplusfusion/halfdecoder-memorytokens.Not equalFLOPs.'),
        ('directed_vs_absolute_fifty40k',('metaworld_goal40_fifty_20261107','directed-flow',40000),('metaworld_current50_20261107','flow',40000),'Same400episodes/50tasks/4NEWphysicalcasespertaskx2learners;40kexactdata/label/sampler/initialCNN-decoderbudget.Matchedcontroller/sensor5/500native.NotcanonicalMT50/publicJEPA-WMs.'),
        ('ordered_equivalent_vs_directed40k',('metaworld_current_roles_20261107','ordered-equivalent',40000),('metaworld_current_goal_20261107','directed-flow',40000),'EXACTsamecapacity/initialfunction/currenttargets/sampler/budget.OnlylinearfeatureparameterizationandresultingAdamWcoordinatesdiffer;notdifferentfunctionclasses.'),
        ('native4_full_vs_truncated',('metaworld_one_native_20261107','all-executed',40000),('metaworld_one_native_20261107','exclude-last4',40000),'ONLYsingleCURRENTa_t4Dtarget;pairedallinitialtensors/dataexceptlast4knowncurrentstates.Bothholdowncommand5native,sensor5/500budget;no fasterfeedback/fakepadding/futurecommandlabels.'),
        ('current_flow_vs_mse',('metaworld_current_inverse_20261107','episodes',40000),('metaworld_current_inverse_mse_20261107','episodes',40000),'Sameoriginaldata/currenttargets/architecture/physicalcases,budget;flowvsregression.'),
        ('current_fresh_vs_fixed',('metaworld_current_inverse_20261107','episodes',40000),('metaworld_current_feedback_20261107','episodes-fixed',40000),'EXACTsameNNs,currentintervaltargets,feedback5;onlyownnoiseconsistencychanges.'),
        ('current_vs_future_chunk',('metaworld_current_inverse_20261107','episodes',40000),('metaworld_chunk_feedback_full_20261107','chunk6-execute1-fresh',40000),'SamephysicalRGBcase/feedback/noiseprotocol;differentanchor/sampling/labels/querychunk.Notpairedinitialweights/equalFLOPs.Nofuturechunkancestry.'),
        ('directed_vs_absolute_10k',('metaworld_current_goal_20261107','directed-flow',10000),('metaworld_current_inverse_20261107','episodes',10000),'SAME64-episode10kscreen,4physicalcases/taskx2learners;matchedCNN/decoderinitialization/labels/anchors/budget,extraorderedfusion/halfdecoder-memorytokens.Notmatureconfirmation/equalFLOPs.'),
        ('fifty_flow_vs_mse',('metaworld_current50_20261107','flow',40000),('metaworld_current50_20261107','mse',40000),'50-taskimage-goal4NEWphysicalcases/taskx2learners;notcanonicalMT50/publicJEPA-WMs.')]
    results={}
    for name,left,right,scope in comparisons:
        l=load(*left);r=load(*right)
        if l is not None and r is not None:results[name]=compare(l,r,scope)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,comparisons=results,new_model_updates=0,new_simulator_runs=0),indent=2)+'\n')
    print(json.dumps({k:{s:{f:v for f,v in x.items()if f!='tasks'}for s,x in r['results'].items()}for k,r in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
