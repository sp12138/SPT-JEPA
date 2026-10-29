# Forecast-only short gates → first epoch

I05 forecast-render alone reached P0 90%/90% after two epochs for training seeds 1/2. This motivates testing the simpler method in six further seeds, rather than assuming the correlated-noise component is necessary.

All six short gates passed the full-state audit in [gate600/audit.json](gate600/audit.json): seeds 3/4 completed 600 additional updates from preserved old states; seeds 5–8 completed 600 updates from scratch. Model, optimizer and scheduler are finite; portable/full weights match; gradients and posterior state interventions are nonzero. New-seed initial model tensors match the original AS-JEPA controls. These are health checks, not mature planning evidence.

New bounded outputs under `/efs/prescottsun/SPT-JEPA/runs/forecast_firstepoch_20261030` resume the exact full Lightning state and stop at update 12,491. All six seeds receive the same P0 cases (evaluation seeds 42/43), health, generation and counterfactual checks before any further training decision. Thirty-six jobs use dev2 GPUs 2/3 and dev13 GPUs 0–3. No automatic second-epoch continuation.

The two forecast-only third-epoch runs remain in the separate parent campaign. Other seed replications retain their existing slots. All parent snapshots, STOP markers and completed short endpoints remain unchanged.
