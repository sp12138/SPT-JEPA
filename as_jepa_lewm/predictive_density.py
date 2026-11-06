"""Control from a goal-conditioned / marginal future-observation density ratio."""
import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.predictive_kernel import PredictiveKernel


@torch.no_grad()
def quadratic_box(matrix, linear, center=None):
    """Global minimum of a four-variable quadratic on a box, including indefinite H.

    A minimizer is stationary on some face. Singular optimal faces have a
    minimizer on their boundary, which is also enumerated. Pseudoinverses keep
    flat faces well-defined; minimum-norm tie breaking leaves unused controls zero.
    """
    import itertools
    matrix, linear = matrix.double(), linear.double()
    center = torch.zeros_like(linear) if center is None else center.double()
    faces = matrix.new_tensor(list(itertools.product([-1., 0., 1.], repeat=4)))[None]
    free = (faces == 0).double()
    m = matrix[:, None] * free[:, :, :, None] * free[:, :, None, :] + torch.diag_embed(1-free)
    rhs = (linear[:, None] - torch.einsum('nij,nkj->nki', matrix, faces.expand(len(matrix), -1, -1))) * free
    inverse = torch.linalg.pinv(m, hermitian=True)
    null = torch.eye(4,device=matrix.device)[None,None]-inverse@m
    candidate = (torch.einsum('nkij,nkj->nki', inverse, rhs)+torch.einsum('nkij,nkj->nki',null,center[:,None]*free))*free+faces
    feasible = (candidate.abs() <= 1+1e-7).all(-1)
    candidate = candidate.clamp(-1, 1)
    costs = torch.einsum('nki,nij,nkj->nk', candidate, matrix, candidate) - 2*torch.einsum('nki,ni->nk', candidate, linear)
    costs = costs.masked_fill(~feasible, float('inf'))
    minimum = costs.min(-1).values
    tolerance = torch.finfo(torch.float64).eps * 128 * (1+minimum.abs())
    norm = (candidate-center[:,None]).square().sum(-1).masked_fill(costs > minimum[:, None]+tolerance[:, None], float('inf'))
    chosen = candidate[torch.arange(len(matrix), device=matrix.device), norm.argmin(-1)]
    assert torch.isfinite(chosen).all()
    return chosen.float()


class PredictiveDensity(PredictiveKernel):
    def __init__(self, variance='point', passive='free', actuation_history=True, control_steps=1, **kwargs):
        assert variance in ['point', 'scalar', 'diagonal']
        assert passive in ['free','zero','velocity','transport','stencil']
        kwargs.update(boundary=True, affine=True, gaussian=variance != 'point')
        super().__init__(**kwargs)
        self.variance, self.passive, self.actuation_history = variance, passive, actuation_history
        assert control_steps in[1,5];self.control_steps=control_steps
        assert control_steps==1 or passive in['free','zero','stencil']
        if control_steps>1:self.effect_head=nn.Conv2d(self.mean_head.in_channels,self.mean_head.out_channels*4*control_steps,1)
        if passive=='transport':
            self.transport_head=nn.Conv2d(self.mean_head.in_channels,self.mean_head.out_channels*9,1)
            nn.init.zeros_(self.transport_head.weight);nn.init.zeros_(self.transport_head.bias)
        if passive=='stencil':
            self.transient_head=nn.Conv2d(self.mean_head.in_channels,self.mean_head.out_channels*4*control_steps,1)
            nn.init.zeros_(self.transient_head.weight);nn.init.zeros_(self.transient_head.bias)
        if variance == 'scalar':
            self.variance_head = nn.Conv2d(self.mean_head.in_channels, 1, 1)

    def fields(self, previous, current, goal, control, role, history_valid, previous_control=None,sequence=False):
        # role -1: unknown-control marginal;0: known current control;1: future goal.
        boundary = (role == 1).float().reshape(-1, 1, 1, 1)
        x = self.predictor(torch.cat([previous, current, (goal-current)*boundary], 1))
        previous_control=torch.zeros_like(control)if previous_control is None else previous_control
        if not self.actuation_history:previous_control=torch.zeros_like(previous_control)
        context = torch.cat([previous_control, role[:, None], history_valid[:, None]], -1)
        gamma, beta = self.context(context.float()).chunk(2, -1)
        x = x*(1+.1*gamma[:, :, None, None])+beta[:, :, None, None]
        mean = self.mean_head(x).float()
        if self.passive!='free':
            velocity=(current-previous)*history_valid[:,None,None,None]
            passive=torch.zeros_like(mean)if self.passive in['zero','stencil']else velocity
            if self.passive=='transport':
                neighbors=F.unfold(velocity.float(),3,padding=1).reshape(len(mean),mean.shape[1],9,*mean.shape[-2:])
                weights=self.transport_head(x).float().unflatten(1,(mean.shape[1],9))
                passive=velocity+(weights*neighbors).sum(2)
            mean=torch.where((role==0)[:,None,None,None],passive,mean)
        effect = self.effect_head(x).float().unflatten(1, (self.control_steps,4, current.shape[1]))
        if self.passive=='stencil':
            transient=self.transient_head(x).float().unflatten(1,(self.control_steps,4,current.shape[1]))
            passive=-torch.einsum('na,nachw->nchw',previous_control,transient[:,0])
            mean=torch.where((role==0)[:,None,None,None],passive,mean)
            following=torch.cat([transient[:,1:],torch.zeros_like(transient[:,:1])],1)
            effect=effect+transient-following
        if not sequence:effect=effect.sum(1)
        if self.variance == 'point':
            variance = torch.ones_like(mean)
        else:
            raw = self.variance_head(x).float()
            if self.variance == 'scalar':
                raw = raw.mean((1, 2, 3), keepdim=True)
            variance = (F.softplus(raw)+torch.finfo(torch.float32).eps).expand_as(mean)
        return mean, effect, variance

    def loss(self, batch, step=0, ramp_steps=0):
        current, target = batch['current_z'].float(), batch['next_z'].float()
        control, valid = batch['executed_control'].float(), batch['history_valid']
        total, count, metrics = current.new_zeros(len(current)), current.new_zeros(len(current)), {}
        for role, name in [(0, 'control'), (-1, 'marginal'), (1, 'goal')]:
            flag = current.new_full((len(current),), role)
            command = control if role == 0 else torch.zeros_like(control)
            drift, effect, variance = self.fields(batch['previous_z'], current, batch['goal_z'], command, flag, valid,batch['previous_control'],sequence=self.control_steps>1)
            if role == 0:
                if self.control_steps>1:
                    drift=drift+torch.einsum('nta,ntachw->nchw',batch['executed_sequence'],effect)
                else:
                    if self.passive in ['velocity','transport']:command=command-batch['previous_control']
                    drift=drift+torch.einsum('na,nachw->nchw',command,effect)
            error = (current+drift-target).square()
            score = (.5*(error/variance+variance.log())).flatten(1).mean(1)
            mask = batch['goal_valid'].float() if role == 1 else torch.ones_like(score)
            total = total+score*mask
            count = count+mask
            metrics[name+'_future_mse'] = (error.flatten(1).mean(1)*mask).sum()/mask.sum().clamp_min(1)
        loss = (total/count).mean()
        return loss, dict(loss=loss, **metrics)

    @torch.no_grad()
    def command(self, previous, current, goal, valid, objective,previous_control=None):
        zero = current.new_zeros(len(current), 4)
        previous_control=zero if previous_control is None else previous_control
        flag = current.new_zeros(len(current))
        drift, effect, forward_variance = self.fields(previous, current, torch.zeros_like(goal), zero, flag, valid,previous_control)
        desired, _, goal_variance = self.fields(previous, current, goal, zero,flag-1 if objective=='marginal'else flag+1, valid,previous_control)
        incremental=self.passive in ['velocity','transport']
        if incremental:drift=drift-torch.einsum('na,nachw->nchw',previous_control,effect)
        center=previous_control if self.passive in['zero','stencil','velocity','transport'] else zero
        if objective in ['projection','marginal']:
            basis=effect.flatten(2).transpose(1,2).double();weight=(forward_variance+goal_variance).flatten(1).double().reciprocal()
            matrix=torch.einsum('npi,np,npj->nij',basis,weight,basis)
            linear=torch.einsum('npi,np,np->ni',basis,weight,(desired-drift).flatten(1).double())
            action=quadratic_box(matrix,linear,center)
        else:
            assert objective == 'ratio'
            marginal, _, marginal_variance = self.fields(previous, current, torch.zeros_like(goal), zero, flag-1, valid,previous_control)
            precision_goal = goal_variance.flatten(1).double().reciprocal()
            precision_marginal = marginal_variance.flatten(1).double().reciprocal()
            difference = precision_goal-precision_marginal
            basis = effect.flatten(2).transpose(1, 2).double()
            rhs = precision_goal*desired.flatten(1).double()-precision_marginal*marginal.flatten(1).double()-difference*drift.flatten(1).double()
            matrix = torch.einsum('npi,np,npj->nij', basis, difference, basis)
            linear = torch.einsum('npi,np->ni', basis, rhs)
            action = quadratic_box(matrix, linear,center)
        future = current+drift+torch.einsum('na,nachw->nchw', action, effect)
        return action, future

    def audit_expected_gradients(self):
        result=super().audit_expected_gradients()
        if self.passive=='transport':result['transport_head']='active'
        if self.passive=='stencil':result['transient_head']='active'
        return result
