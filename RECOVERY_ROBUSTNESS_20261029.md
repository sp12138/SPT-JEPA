# User redirected to short-budget screening

Read [current recovery](RECOVERY_SHORTSCREEN_20261029.md). The old16run long matrix is gracefully paused; do not restart from historical statements.

---

# Active same-size robustness campaign — 2026-10-29

The requested full-data beta=.25 three-epoch reproduction finished training seeds1-4. Primary P0 means88.0/74.67/86.0/87.33%, mean84.0%,sampleSD6.28pp; fresh cases88.0/74.0/86.0/86.67%,mean83.67%. Seed0 selected reference remains89.33%. The predeclared stable-high-performance criteria fail. Preserve every seed.

Current root:/efs/prescottsun/SPT-JEPA/runs/robustness_20261029; protocol.json and plan.json are authoritative. Four methods: lower-beta(.0625), temporal-noise(rho=.8), forecast-render(fraction=.5), temporal-forecast(both). Seeds1-4, identical51,097,134parameters, batch128, sigma=.1,37473updates with124910-step LR schedule. No batch-level regularizer or BatchNorm. Forecast extra prior calls cost compute; report time.

Source-v1 is frozen at6b83290. Existing Lightning train/checkpoint ownership remains. All133 planned jobs comprise4training smokes,1loading/planning smoke,16training,96planning,16endpoint evaluations. Full training is blocked on gates/all-smokes-PASS.json and all four training smoke exits and the planning/loading smoke exit0; do not bypass. Original completion_20261028 remains held, and the stability_20261029 evaluation phase will be closed after all its endpoints finish.

Three detached per-host dispatchers run the new root; inspect workers/launch-*.json and actual processes before doing anything. Do not duplicate workers or edit a launched source/config. Recovery from full-state checkpoint must enforce the37473-step cap despite original resume behavior bypassing stop_after. No ten-epoch continuation is authorized by this plan.

When no independent work remains, inspect roughly every30-60minutes and sleep between checks. Stop numerical failures/persistent independently confirmed collapse promptly; do not cull finite nonempty seeds for weak early planning. Commit and push every useful new conclusion with full numerical results and scientificPNG/SVG.

If a simple winner improves/stabilizes the matched four seeds, validate only that selected method on reserved training seeds5/6 and evalseeds2042/2043/2044 before a strong claim. If no method helps, diagnose before another intervention. User authorization for evidence-driven same-size research and publication remains active.
