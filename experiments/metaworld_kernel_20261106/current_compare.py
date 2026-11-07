"""Read-only actual-trace verification and paired physical-case bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA/runs')


def load(name,arm,step=40000,seeds=None,cohort='standard'):
    root=BASE/name;training=(root/'training-plan.json').exists();plan=json.loads((root/('training-plan.json'if training else'plan.json')).read_text());rows={};evidence=[]
    for r in plan['recipes']:
        if r['arm']!=arm:continue
        if seeds is not None and r['seed']not in seeds:continue
        location={'final':'confirm-final','hold5':f'hold-step{step:07d}'}.get(cohort,f'eval-step{step:07d}')
        path=Path(r['run']if training else r['output'])/(location if training else'pilot')/'summary.json'
        if not path.exists():return None
        value=json.loads(path.read_text());assert value['passed']and value['model_unchanged']
        evidence.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=value['model_sha256']))
        for row in value['rows']:
            assert hashlib.sha256(Path(row['trace']).read_bytes()).hexdigest()==row['trace_sha256']
            key=(r['seed'],row['task'],row['seed']);assert key not in rows;rows[key]=row
    assert rows
    return rows,evidence


def combine(*values):
    if any(v is None for v in values):return None
    rows={};evidence=[]
    for r,e in values:
        assert not rows.keys()&r.keys();rows.update(r);evidence.extend(e)
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
        ('factor_coupled_vs_native_oldactors',('metaworld_native_feedback_factor_20261107','coupled-oldactor',40000),('metaworld_current_control_full_20261107','world-native4',40000),'SAMEq1/2andF1/2NNs/cases/realcam5/5policycalls/4F/noisevariance1. Onlycouplingchanges:ONEownlatentperREALimagevs5independentnative-latents. No temporalhyperparameter;notfixedwholetrajectory.0SGD.'),
        ('factor_coupled_vs_native_newactors',('metaworld_native_feedback_factor_20261107','coupled-newactor',40000),('metaworld_native_feedback_long_20261107','directed',40000),'SAMEq3/4at40kof120kLR/F3/4/physicalcases/sensor5/inferencecalls.OnlyONEownlatentperreal-observationintervalvsnativeIID. Notcompareold40kLRschedule/seedalone.'),
        ('factor_worldswap_oldactors',('metaworld_native_feedback_factor_20261107','cross-oldactor-newworld',40000),('metaworld_current_control_full_20261107','world-native4',40000),'SAMEq1/2andownGaussianstreams/cases/realsensor5;ONLYF1/2replacedbyF3/4(ownsameoriginaldata/scratch20karchitecture). Isolatesworld-weightvariation,notnewdata/teacher.'),
        ('factor_worldswap_newactors',('metaworld_native_feedback_factor_20261107','cross-newactor-oldworld',40000),('metaworld_native_feedback_long_20261107','directed',40000),'SAMEq3/4at40kof120kLR/cases/noise/realsensor5;ONLYF3/4replacedbyF1/2. Actor-groupdifferencesalsocontainLRscheduleandseeddifferences;notuniquelyaseedcausaltest.'),
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
    mature1000='metaworld_native_mature1000_20261107'
    if (BASE/mature1000/'plan.json').exists():
        for right in ['static','actual','current-only']:
            l=load(mature1000,'world');r=load(mature1000,right)
            if l is not None and r is not None:results['mature_native_world_vs_'+right+'_1000']=compare(l,r,
                'TwoNEWtrainedactorseeds3/4,40kbudget/LR40k/ownF3/4. Existing1000-case50taskexpandedcomparison,notuntouchedconfirmation. Matcheda_t4Dtargets/5nativepolicycalls/500budget;world/static/actualuseSAMEactorNNs,current-onlyisaseparatelytrainedinformationablation. '+
                ('Static5drawsusesSAMEactualcamera5;worldadds4ownforecasts/20kFpretrain.'if right=='static'else
                 'EXTRAactualRGBEVERYnativecontrol,explicitdifferentinformationtrack/diagnosticONLY;notfairfixedsensor5baseline.'if right=='actual'else
                 'Capacity/initialization/data/sampler/40kLR40k/worldmatchedCURRENT-ONLYinformationbaseline: futuregoalneverencoded. NotSAMEtrainedactorNNs inthisinformationablation.'))
    extra=[
        ('mature_schedule_vs_120k_schedule_at40k',('metaworld_native_maturity40_20261107','directed'),('metaworld_native_feedback_long_20261107','directed'),
         'SAMEactorseeds3/4/allrandominitialtensors/data/currenta_t-onlytargets/40kactualupdates/F3/4/cases/sensor5. ONLYmax_stepsschedulehorizon40kvs120k(andderived1%warmup)changes;notaseedonlytest.'),
        ('goal_information_vs_current_only',('metaworld_native_maturity40_20261107','directed'),('metaworld_native_currentonly40_20261107','current-only'),
         'Samecapacity/allinitialtensors/seeds3/4/currentactionlabels/40kbudget/LR40k/ownF3/4/cases. Current-onlycontrolduplicatesCURRENTbeforeCNN,neverencodesfuturegoal. GoalRGBmaycarrytaskinformation;notgeneralgoalreachabilityproof.'),
        ('censored_vs_point_four_components',('metaworld_action_measure_20261107','censored'),('metaworld_action_measure_20261107','point'),
         'Exactsamewholeinitialnetwork/seeds3/4/data/a_t4D/40k/LR40k/ownF/cases/sensor5/one-passsampling. ONLYtrainlikelihoodtreatsactuatorboundaryvaluesasCDFmassvsinteriordensity. Standardcensoring/mixturestatistics;notclaimednovel.'),
        ('censored_vs_point_single_component',('metaworld_action_measure_20261107','censored-single'),('metaworld_action_measure_20261107','point-single'),
         'Matchedsingle-logisticjoint4Dhead;entireinitialstates/labels/budget/world/casesmatch. ONLYsaturated-actionlikelihoodchanges. Bothsamplelatentthenclip.'),
        ('censored_mixture_vs_single',('metaworld_action_measure_20261107','censored'),('metaworld_action_measure_20261107','censored-single'),
         'SameCNN/decoder/fusioninitials/labels/data/40kLR40k/ownF/cases. Fourmixturecomponentsvsone:headcapacityandconditionalmultimodalitydiffer. No per-taskselection.'),
        ('censored_vs_flow',('metaworld_action_measure_20261107','censored'),('metaworld_native_maturity40_20261107','directed'),
         'SameCNN/decoder/fusioninitialtensors/currenta_tlabels/originaldata/sampler/40kLR40k/ownF3/4/cases/sensor5. One-passcensoredlogisticmixturevs12-passflow;finalhead/loss/ownrandomdrawsdiffer,not equalinferenceFLOPs.'),
        ('correspondence_vs_fixed_pairing',('metaworld_action_measure_20261107','correspondence'),('metaworld_native_maturity40_20261107','directed'),
         'SameCNN/decoderinitialtensors/a_t4DFlowobjective/originaldata/sampler/40kLR40k/F3/4/realcam5/cases. Learnedsample-levelpatchcorrespondence+imagecoordinateoffsetvsfixedsame-locationpairing;extraQ/K/fusionparameters/computeexplicit,not equalFLOPs.')]
    median='metaworld_action_measure_median50_20261107'
    if (BASE/median/'plan.json').exists():
        for arm in ['censored','point','censored-single','point-single']:
            l=load(median,arm);r=load('metaworld_action_measure_20261107',arm,cohort='final')
            if l is not None and r is not None:results[f'mixture_componentmedian_vs_stochastic_{arm}_screen50']=compare(l,r,
                'EXACTSAMEfrozen40kactor/F3/4NNs/cases/camera5/5policycalls/4F. Onlydecisionrulechanges: highest-weightlatentcomponentmedianthenclipvsowncategorical/logisticsampling. No newSGD/threshold/taskrule. NotclaimedfullmixtureMAP/optimalaction;50task4caseSCREEN.')
    for name,left,right,scope in extra:
        if not all((BASE/x[0]/'training-plan.json').exists()for x in [left,right]):continue
        for cohort in ['standard','final']:
            l=load(*left,cohort=cohort);r=load(*right,cohort=cohort)
            if l is not None and r is not None:results[name+('_screen50'if cohort=='final'else'_full8')]=compare(l,r,scope+(' 50task4caseSCREEN.'if cohort=='final'else' Full8task20physicalcases.'))
    solver='metaworld_native_solver_control_20261107'
    if (BASE/solver/'plan.json').exists():
        for arm,parent,parent_arm in [('old','metaworld_current_control_full_20261107','world-native4'),('new','metaworld_native_feedback_long_20261107','directed')]:
            for kind in ['euler24','midpoint12']:
                l=load(solver,f'{kind}-{arm}');r=load(parent,parent_arm)
                if l is not None and r is not None:results[f'{kind}_vs_euler12_{arm}']=compare(l,r,'SAMEfrozenactors/worlds/cases/sensor5/ownnoise;24vs12actionfieldpassesONLY.Euler768ODEaccuracyreferenceisnotphysicaltruth.0SGD.')
    native=load('metaworld_native_goalclock_20261107','native-goal')if(BASE/'metaworld_native_goalclock_20261107/training-plan.json').exists()else None
    if native is not None:
        macro=combine(load('metaworld_native_goalclock_20261107','macro-goal'),load('metaworld_native_feedback_long_20261107','directed'))
        if macro is not None:results['goalclock_native_vs_macro_full8']=compare(native,macro,'Fourpairedactorseeds3..6 ONLYhindsightgoalRGBgrid1vs5changes;EXACTa_t4Dtargets/currentanchors/sampler/architecture/init/budget/120kLRhorizon/ownFwithinpair/sensor5. ReuseALREADYtrained3/4macrogoalcontrols;noSGDrepeat.World1/2reusedfor5/6explicit.40kCAP,note120kfinalperformance.')
        left=load('metaworld_native_goalclock_20261107','native-goal',40000,[5,6],'final');right=load('metaworld_native_goalclock_20261107','macro-goal',40000,[5,6],'final')
        if left is not None and right is not None:results['goalclock_native_vs_macro_fifty']=compare(left,right,'50task4caseSCREEN,ONLYnewpairedactorseeds5/6;matchedcase/initialNN/currenta_t/40kbudget/LR120k/ownF/sensor5.3/4nativegoalbroadrowsnotmixedwithunmatchedcontrols.No publicSOTA/equalFLOPclaim.')
    replica='metaworld_native_feedback_replicas_20261107'
    if(BASE/replica/'training-plan.json').exists():
        for step in [40000,80000,120000]:
            left=load(replica,'directed',step);right=load(replica,'directed',step,None,'hold5')
            if left is not None and right is not None:results[f'eight_native_replicas_world_vs_hold_{step}']=compare(left,right,'EightNEWactorseeds7..14,ownF1..4reusedtwice(CORRELATEDworldpretraining). SAMEactorweights/cases/realsensor5;world5calls/4forecastsversushold1draw. CIconditionalonlistedactorsand4frozenworlds;not8independentfullpipelinepretrainings/equalFLOPs.')
    diagnostics={};goal=BASE/'metaworld_goal_input_diagnostic_20261107/results/goal-input-dependency.json'
    if goal.exists():diagnostics['goal_input_dependency']=dict(path=str(goal),sha256=hashlib.sha256(goal.read_bytes()).hexdigest(),scope='InconsistentRGBgoal diagnosticONLY,notbenchmarkperformance. Physicalcontext/NN/RNGsame;propergoalrefsandfirst-commanddifferencesverifiedseparately.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,comparisons=results,diagnostics=diagnostics,new_model_updates=0,new_simulator_runs=0),indent=2)+'\n')
    print(json.dumps({k:{s:{f:v for f,v in x.items()if f!='tasks'}for s,x in r['results'].items()}for k,r in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
