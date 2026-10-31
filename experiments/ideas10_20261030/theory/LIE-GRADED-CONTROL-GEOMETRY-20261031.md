# Lie-graded control geometry: valid distinction, unsupported Cube mechanism

Scope: theory and a small CPU analytic witness only. No model changes, fitting, GPU use, or new benchmark result. This extends the [Gramian review](../../hardtasks20261030/GRAMIAN-READOUT-ANALYSIS-20261030.md); it does not establish a new algorithm.

**Verdict.** Nonlinear reachable-path geometry can distinguish directions invisible to a first-order Gramian without assigning an arbitrary large cost to its entire nullspace. However, neither our factual prediction results nor Cube's contact mechanics establish the smooth, locally identifiable control fields needed to estimate this geometry. “Learn fields, take brackets, use a Carnot distance” is classical control machinery applied to a world model, not yet a substantive new learning mechanism.

## Exact constructive distinction

Choose the convention
\[
X=\partial_x-\tfrac y2\partial_z,\quad
Y=\partial_y+\tfrac x2\partial_z,\quad [X,Y]=\partial_z,
\qquad \dot s=uX+vY.
\]
Measure control length \(L=\int\sqrt{u^2+v^2}\,dt\), with symmetric controls. The direct endpoint \((a,0,0)\) has distance \(|a|\): projection gives the lower bound, a straight path achieves it. For \((0,0,b)\), the planar projection is closed and
\[
b=\tfrac12\oint(x\,dy-y\,dx),\qquad |b|\le L^2/(4\pi).
\]
A circle through the origin attains equality, hence **\(d(0,(0,0,b))=2\sqrt{\pi|b|}\)**. The dilation \((x,y,z)\mapsto(\lambda x,\lambda y,\lambda^2z)\) scales distance by \(|\lambda|\): the vertical direction has degree two. A primitive square loop of amplitude \(\epsilon\) gives exactly \(z=\epsilon^2\), length \(4|\epsilon|\). Thus residuals \(A=(.04,0,0)\), \(B=(0,0,.001)\) satisfy Euclidean \(B<A\), but control length \(.04<.11210\). At fixed horizon \(T\), minimum quadratic energy is \(d^2/T\); length, energy and travel time must not be conflated.

At regular bracket-generating points, analogous local ball-box estimates use privileged coordinates and bracket degrees, not arbitrary powers of latent coordinates. Consistently pushing forward the fields and retaining the same action cost preserves control distance under smooth state reparameterization. Changing action units or introducing a latent Euclidean regularizer changes the problem.

**No action-invariance loss.** If a smooth representation component \(f\) satisfies \(Xf=Yf=0\) on an open neighborhood, then \([X,Y]f=X(Yf)-Y(Xf)=0\); induction removes every iterated bracket direction. Bracket generation then makes \(f\) constant on connected reachable regions. Pointwise invariance is insufficient: \(f=z\) has \(Xf=Yf=0\) at the origin but \([X,Y]f=1\). Quotienting out all primitive actions would erase the very object changes the proposal aims to preserve.

## What factual sample-level learning can and cannot identify

A sample-level transition loss can estimate supported control fields without optimal-action labels, provided observations form a Markov chart, actions locally excite the system, dynamics are sufficiently smooth, and learned state derivatives—not merely next-state values—are accurate. These are substantive conditions. The exact constant-action flow above is
\(F_H((x,y,z),(a,b))=(x+a,y+b,z+(xb-ya)/2)\).
The commuting model \(F_C=(x+a,y+b,z)\) agrees with it at the origin for **every** constant action, yet their four-action loops end at \(z=\epsilon^2\) and zero. One-step agreement there cannot identify the bracket. Expert-policy support can impose further off-support ambiguity. Autodifferentiating a learned extension does not supply missing factual evidence; fixed-duration action blocks are not automatically infinitesimal fields.

The missing operation is supported identification and validation of reachable directions and their scaling, including contact modes. Factual neighboring-state/action-order observations could supply it; current audits do not establish that support. A learned fixed update subspace is another obstruction: if \(V_i(s)=W h_i(s)\), then \([V_i,V_j]=W(Dh_jWh_i-Dh_iWh_j)\). All higher brackets remain in the same fixed span. Taking brackets cannot repair the audited fixed-span predictor bottleneck.

## Cube boundaries and prior-art audit

Before contact, if every field's object component vanishes throughout an open neighborhood, **all finite brackets also have zero object component there**. Finite arm motion must first cross a contact boundary. During contact, object displacement may instead be first order. Unilateral contact, friction, gravity and irreversibility violate the smooth driftless symmetric-control example. A slow direction \(\dot z=\kappa u\) is degree one, not a higher degree. In a scaled Heisenberg system, vertical cost is \(2\sqrt{\pi|b|/|\kappa|}\); uncertain near-zero coupling recreates the Gramian's “ignorance mistaken for difficulty” failure. Rank/depth changes invalidate a single regular grading. Partial observation adds ambiguity. None of these costs identifies which parts of a goal image are task-relevant.

Multi-level bracket steering is classical: [Murray and Sastry, *Nonholonomic Motion Planning: Steering Using Sinusoids*](https://murray.cds.caltech.edu/Nonholonomic_Motion_Planning:_Steering_Using_Sinusoids). Learning dynamics using Lie structure also has direct precedent in [GEM](https://arxiv.org/abs/2104.02844). Action-order sensory differences overlap [Sensory Commutativity](https://arxiv.org/abs/2002.05630), already recorded in our rejected D5; they are not a fresh mechanism.

The closest verified world-model metric precedent is [*Latent World Models Should Plan with Distances Along the Latent Manifold*](https://arxiv.org/abs/2610.25013): NGC replaces LeWM's terminal Euclidean cost with shortest paths on a training-latent mutual-neighbor graph, without retraining. It is geometrically distinct: dense Euclidean-neighbor graphs on the full Heisenberg state manifold approximate ordinary manifold length, whereas control distance restricts paths to horizontal velocities and scales vertical displacement as its square root. Restricting edges to supported action transitions changes the graph problem and its coverage requirements. No published performance numbers are transferred to our cases. This focused review did not verify an exact learned Lie-graded latent terminal-cost precedent; that absence is not evidence of novelty.

**Decision:** retain the analytic counterexample, do not promote a Lie-metric candidate from present evidence. If later tested, it is a controller/readout intervention applicable equally to LeWM, TF and carriers, with matched action units, data exposure, model-call and search budgets. A better readout would not by itself establish better world-model learning.

CPU evidence: [check_lie_geometry.py](check_lie_geometry.py) and [lie-geometry-witness.json](lie-geometry-witness.json): 81 exact one-step equivalences, eight exact loop/scaling witnesses, the reversed ranking, and the pointwise-invariance caveat. These check algebraic examples, not Cube assumptions.
