# Sample-level objective research with short adaptation gates

This is an exploratory set of9candidate/control modifications plus an exact original control, each on2diagnostic training seeds. All20trials start from the same paired2epoch full state and run only600additional updates before health/task checks. This uses existing planning-capable models to test whether an intervention helps quickly; it is not proof of from-scratch stability.

All51,097,134trainable parameters and inference architecture are identical. No cross-sample loss/BatchNorm/EMA/target network is added. See [prespecified protocol and sources](protocol.json) for objective definitions, controls, budgets and limitations. Original ideas/prior methods are attributed explicitly; no groundbreaking claim is made before evidence.

The20GPU trials complement4selected bounded stage2runs, filling24authorized cards. Every new trial stops at600updates; no full-budget extension is automatic.
