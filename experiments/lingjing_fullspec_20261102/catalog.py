"""All predeclared primary recipes;question coverage is distinct from maturity."""
import argparse
from pathlib import Path

from common import write


def catalog():
    recipes=[]
    def add(track,block,arm,task,seeds,model_class,changes=None,calibration=None,anchor='default',tier=1,questions=()):
        for seed in seeds:
            base=dict(mode='L2',beta=.25,sigma0=.1,window=8,anchor_gap=8,renderer_width=384,renderer_depth=12,renderer_heads=6,
                      fp32_pool=False)
            if model_class=='forecast_scored':base.update(lam=1.,horizon='one_step',forecast_noise='sample',rho=0.,forecast_grad='joint',forecast_seed=seed)
            if model_class=='decoder_free':base.update(beta=.3,lam=1.,horizon='one_step',forecast_noise=False,prior_input='sample',estimator='logdet',conditioning='anchor',forecast_seed=seed)
            if model_class=='nce':base.update(beta=1.,lam=1.,horizon='one_step',forecast_noise='sample',rho=0.,forecast_seed=seed)
            if model_class=='base':base.update(mode='L0',beta=4.,window=4)
            base.update(changes or {})
            if task!='pusht':base['action_dim']=10 if task=='tworoom' else 25
            target=(152650 if track=='round2' else 45795) if model_class=='base' else (6000 if task=='tworoom' else 12000 if task=='cube' else 37473 if track=='decoder_free' or model_class=='nce' else 24982)
            recipes.append(dict(id=f'{track}-{block}-{arm}-s{seed}',track=track,block=block,arm=arm,task=task,seed=seed,
                model_class=model_class,model=base,calibration=calibration,anchor=anchor,tier=tier,questions=list(questions),
                nominal_target_step=target,conditional_target=task=='tworoom',status='planned',
                insight_checkpoints=[200,1000,2000],scope='Do not call an early-stop result a completed nominal training budget.'))
    # Round2 rev3:64 primary recipes,including everycell of the forecast factorial.
    add('round2','P1','L0','pusht',[0,1,2],'base',questions=['matched_baseline'])
    cells=[(.5,'rollout','sample',0),(.5,'rollout','sample',.8),(.5,'one_step','mean',0),(.5,'one_step','mean',.8),
           (.5,'one_step','sample',0),(.5,'one_step','sample',.8),(1.,'one_step','sample',0),(1.,'one_step','sample',.8),
           (1.,'rollout','sample',0),(1.,'rollout','sample',.8)]
    for n,(lam,horizon,noise,rho) in enumerate(cells,1):
        add('round2','P2',f'F{n}','pusht',[1,2],'forecast_scored',dict(lam=lam,horizon=horizon,forecast_noise=noise,rho=rho),
            tier=1 if n in [1,2,5,7,8] else 2,questions=['noise','horizon','rho','lambda','priced_channel'])
    add('round2','P3','a','pusht',[3,5,6,7],'forecast_scored',dict(lam=0.),calibration=.1,tier=2,questions=['initial_scale'])
    add('round2','P3','sham','pusht',[3,5],'forecast_scored',dict(lam=0.),calibration='P3a_original_std',tier=2,questions=['initial_scale_sham'])
    for arm,sigma in [('b',.05),('c',.2)]:add('round2','P3',arm,'pusht',[3,5],'forecast_scored',dict(lam=0.,sigma0=sigma),calibration=sigma,tier=2,questions=['code_resolution'])
    add('round2','P4','I2','pusht',[1],'nce',tier=2,questions=['reconstruction_free'])
    for n in range(1,10):
        change={};anchor='past' if n in [2,4,6,7,8] else 'near';sigma=.05 if n==7 else .2 if n==8 else .1
        change.update(lam=0. if n in [1,2] else 1.,sigma0=sigma)
        if n==6:change['beta']=1/16
        if n in [5,9]:change.update(lam=.5,horizon='rollout',forecast_noise='mean' if n==5 else 'sample',rho=.8)
        add('round2','T',f'T{n}','tworoom',[1,2],'forecast_scored',change,calibration=sigma,anchor=anchor,tier=2 if n in [7,8] else 1,
            questions=['initial_scale','anchor','priced_channel','code_resolution','state_use'])
    for n in range(1,7):
        sigma=.05 if n==5 else .2 if n==6 else .1
        change=dict(sigma0=sigma,lam=0. if n==4 else 1.)
        anchor='near' if n==2 else 'zero' if n==3 else 'past'
        if n==3:change['renderer_anchor_mode']='zero'
        add('round2','K',f'K{n}','cube',[1,2],'forecast_scored',change,calibration=sigma,anchor=anchor,tier=2 if n in [5,6] else 1,
            questions=['anchor','code_resolution','cube_goal_geometry'])
    assert len(recipes)==64
    # Decoder-free pinned rev2:71 primary recipes,not including conditional Phase2.
    for beta in [.1,.3,1.]:
        for lam in [0.,.5,1.]:add('decoder_free','A',f'b{beta:g}-l{lam:g}','pusht',[1,2],'decoder_free',dict(beta=beta,lam=lam),calibration=.1,
            tier=1 if (beta,lam) in [(.3,1.),(.3,0.),(1.,0.),(.1,1.)] else 2,questions=['price_prediction_plane','purity_completeness'])
    for lam in [.5,1.]:
        for suffix,horizon,noise in [('1s','one_step',True),('om','rollout',False),('os','rollout',True)]:
            add('decoder_free','B',f'l{lam:g}-{suffix}','pusht',[1,2],'decoder_free',dict(lam=lam,horizon=horizon,forecast_noise=noise),calibration=.1,tier=2,questions=['noise_effective_weight','open_loop'])
    for lam in [0.,1.]:add('decoder_free','C',f'mean-l{lam:g}','pusht',[1,2],'decoder_free',dict(lam=lam,prior_input='mean'),calibration=.1,tier=3,questions=['prior_input'])
    for arm,change in [('win-l1',dict(estimator='logdet_window')),('win-l0',dict(estimator='logdet_window',lam=0.)),('batch',dict(conditioning='batch'))]:
        add('decoder_free','D',arm,'pusht',[1,2],'decoder_free',change,calibration=.1,tier=3,questions=['estimator_gaming','slow_components'])
    dot=[(.3,0.,'one_step',False),(.3,.5,'one_step',False),(.3,1.,'one_step',False),(.3,1.,'one_step',True),
         (.3,1.,'rollout',False),(1.,0.,'one_step',False),(.6,.5,'one_step',False),(.3,1.,'rollout',True)]
    for n,(beta,lam,horizon,noise) in enumerate(dot,1):add('decoder_free','E',f'E{n}','pusht',[1],'decoder_free',dict(domain='dot',beta=beta,lam=lam,horizon=horizon,forecast_noise=noise),calibration=.1,tier=1 if n in [1,3,4] else 2,questions=['transient_threshold'])
    for n in [1,2,3,5]:
        change=dict(domain='color',lam=0. if n==2 else 1.)
        if n==3:change['conditioning']='batch'
        if n==5:change['estimator']='logdet_window'
        add('decoder_free','F',f'F{n}','pusht',[1],'decoder_free',change,calibration=.1,questions=['appearance_invariance','estimator_gaming'])
    add('decoder_free','F','F4-L0','pusht',[1],'base',dict(domain='color'),questions=['appearance_positive_control'])
    for arm,change,cal,anchor in [('noinit',{},None,'default'),('init3',{},.3,'default'),('gap0',dict(anchor_gap=0),.1,'near'),('noinit-b1',dict(beta=1.),None,'default')]:
        add('decoder_free','G',arm,'pusht',[1,2],'decoder_free',change,calibration=cal,anchor=anchor,tier=3,questions=['initial_trap','reference_distance'])
    add('decoder_free','H','s0.3','pusht',[1,2],'decoder_free',dict(sigma0=.3,beta=1/30,lam=.7667),calibration=.3,tier=3,questions=['effective_parameter_equivalence'])
    for arm,change,cal in [('df',{},.1),('df-b0.1',dict(beta=.1),.1),('df-noinit',{},None),('df-l0',dict(lam=0.),.1)]:
        add('decoder_free','T',arm,'tworoom',[1,2],'decoder_free',change,calibration=cal,anchor='past',tier=1 if arm in ['df','df-noinit'] else 3,questions=['state_trap','decoder_removal'])
    assert len(recipes)==135
    return dict(recipes=recipes,counts=dict(round2=64,decoder_free=71),
        conditional=['P2 selected continuation+seeds3/4','P2 prior-gradient fallback at most2cells','P3 sham reads P3a calibration',
                     'T4 beta0 paired diagnostic when calibrated state contracts','T6000 only after use/state gate','K23342 only when maturity rule triggers',
                     'Round2 Stage2 selected transfers','DF one-time trap/gaming rescues','DF Phase2 only if viable candidate'],
        user_priority='Insight first. Early collapse is a recorded outcome,not a completed nominal budget. Do not infer collapse from low early planning alone.',
        gate_policy=dict(technical_smoke_step=200,post_warmup_checkpoints=[2000],
          numerical_error='Nonfinite/OOM/source mismatch are implementation or numerical outcomes,not validated algorithm failure.',
          calibrated_early_collapse='Require FP32 signal/noise and noisy information probes to confirm severe loss of a previously calibrated signal;retain both seeds and positive controls.',
          uncalibrated='Initially small spread alone is expected and is not an early-stop proof;compare mature reference learning trajectories.',
          healthy='Continue prescribed maturity budget when representation remains live;no arbitrary2000-step negative verdict.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();d=catalog();write(a.output,d);print(d['counts'])
