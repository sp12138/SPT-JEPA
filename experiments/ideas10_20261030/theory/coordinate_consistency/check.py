"""Small scalar coordinate-consistency identities; no learned model or data."""
import json,math,random
from pathlib import Path

sigma=.1;rho=.8;theta=-.3
entry_constant=.5*(sigma*sigma-1-math.log(sigma*sigma))
transition_constant=.5*(-rho*rho-math.log1p(-rho*rho))

def normal_log(x,mean,variance):
    return -.5*(math.log(2*math.pi*variance)+(x-mean)**2/variance)

def transform(x,mode):
    if mode=='identity':return x
    if mode=='linear':return .5*x
    return x+theta*math.tanh(x)

def derivative(x,mode):
    if mode=='identity':return 1.
    if mode=='linear':return .5
    return 1+theta/(math.cosh(x)**2)

def inverse(z,mode):
    if mode=='identity':return z
    if mode=='linear':return 2*z
    x=z
    for _ in range(80):x=z-theta*math.tanh(x)
    return x

def f(x,a):return .65*x+.08*x*x+a

def decoder(x,y,t):return .2*y+.3*x+t

rng=random.Random(1042);u=[.2,.6,.4,1.,.7,1.1,.9,1.2]
eps=[rng.gauss(0,1)]
for _ in range(7):eps.append(rho*eps[-1]+math.sqrt(1-rho*rho)*rng.gauss(0,1))
s=[x+sigma*e for x,e in zip(u,eps)];actions=[.1,-.2,.3,.1,-.1,.2,0.]
checks=[]
for mode in ('identity','linear','nonlinear'):
    T=lambda x:transform(x,mode);I=lambda z:inverse(z,mode)
    F=lambda z,a:T(f(I(z),a))
    errors=[];density_errors=[];mixture_errors=[]
    for t in range(1,8):
        m=f(s[t-1],actions[t-1]);mz=F(T(s[t-1]),actions[t-1]);mback=I(mz)
        qmean=u[t]+rho*(s[t-1]-u[t-1])
        raw=(qmean-m)**2/(2*sigma*sigma)+transition_constant
        transformed=(qmean-mback)**2/(2*sigma*sigma)+transition_constant
        errors.append(abs(raw-transformed))
        logq=normal_log(s[t],qmean,sigma*sigma*(1-rho*rho))
        logp=normal_log(s[t],m,sigma*sigma)
        jac=math.log(derivative(s[t],mode))
        density_errors.append(abs(((logq-jac)-(logp-jac))-(logq-logp)))
    for initial in (u[0],s[0]):
        raw=initial;chart=T(initial)
        for t,a in enumerate(actions):
            raw=f(raw,a);chart=F(chart,a)
            errors.append(abs(I(chart)-raw))
            condition=s[t+1] if t%2 else raw
            chart_condition=T(s[t+1]) if t%2 else chart
            mixture_errors.append(abs(decoder(condition,.4,.7)-decoder(I(chart_condition),.4,.7)))
        errors.append(abs((I(chart)-u[-1])**2-(raw-u[-1])**2))
    checks.append(dict(mode=mode,max_rate_rollout_goal_error=max(errors),max_log_density_ratio_error=max(density_errors),max_forecast_mixture_decoder_error=max(mixture_errors)))
    assert max(errors+mixture_errors+density_errors)<1e-11
# Expected native rate for u0=1,u1=2, carry predictor, no frame normalization.
# Native chart-space sigma remains fixed; consistent variant transports noise.
shrink=[]
for a in (1.,.5):
    old=a*a*(.5+1/(2*sigma*sigma))+.5*(rho-1)**2+entry_constant+transition_constant
    consistent=.5+1/(2*sigma*sigma)+.5*(rho-1)**2+entry_constant+transition_constant
    shrink.append(dict(a=a,old_total_rate=old,consistent_total_rate=consistent,clean_raw_forecast_squared_error=1.))
# For z(theta)=T_theta(x), inverse differentiation must include direct theta.
x=1.2;jac=1+theta/math.cosh(x)**2;zprime=math.tanh(x)
correct=(zprime-math.tanh(x))/jac;wrong=zprime/jac
assert correct==0 and abs(wrong)>.1
result=dict(sigma=sigma,rho=rho,posterior_conditional_variance=sigma*sigma*(1-rho*rho),transition_covariance_kl_per_dimension=transition_constant,checks=checks,scalar_shrink_counterexample=shrink,inverse_gradient=dict(correct_total_derivative=correct,omitting_inverse_parameter_derivative=wrong),scope='Scalar algebra and deterministic numerical checks only. No native model/gradient/RNG contract or benchmark claim.')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
