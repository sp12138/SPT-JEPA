# Dynamic-readout three-update preflight

All 16 endpoints passed (8 native three-update stages and 8 full-state audits). Both tasks, both seeds, STATIC/STATE share exactly the original 553 initial tensors and the same four new head tensors (5,376 parameters). All 210 frozen tensors stayed unchanged. Native optimizer, scheduler, RNG, sample cursor and portable/full checkpoint equality passed. Peak allocation: 107.500–107.501 GiB.

| Run | Head input weight max gradient | Head output weight max gradient | State-input max gradient | Peak GiB |
|---|---:|---:|---:|---:|
| tworoom-readout-static-seed1 | 3.79e-15 | 1.67e-08 | 0 | 107.500 |
| tworoom-readout-state-seed1 | 8.79e-13 | 4.33e-07 | 8.79e-13 | 107.500 |
| tworoom-readout-static-seed2 | 1.7e-15 | 2.54e-08 | 0 | 107.500 |
| tworoom-readout-state-seed2 | 1.4e-12 | 5.1e-07 | 1.4e-12 | 107.500 |
| cube-readout-static-seed1 | 7.46e-14 | 9.68e-08 | 0 | 107.501 |
| cube-readout-state-seed1 | 3.26e-13 | 4.11e-07 | 3.26e-13 | 107.501 |
| cube-readout-static-seed2 | 7.99e-14 | 9.05e-08 | 0 | 107.501 |
| cube-readout-state-seed2 | 8.46e-13 | 5.05e-07 | 8.46e-13 | 107.501 |

These are startup checks, not evidence of useful learned dynamics. All head input weights are still byte-identical after the tiny warmup updates; output weights changed. The positive current input gradients establish a live path only. STATIC state columns have exactly zero gradient by design. Training emitted NFS temporary-file cleanup warnings, but all native saved-state checks passed. The source retains the previous raw inverse implementation; every tested arm uses chartnone and does not execute that inverse.

Source: `d6d0a9f85127a8af3e65bd31b29c161c3584867c`. Dispatcher1178789 closed at03:12:52UTC after exact idle verification; all8 GPUs free. The three-step root is held. Root subsequently authorized a separate eight-run600-step gate; no automatic continuation beyond600.


Startup logging caveat: the unchanged native training logger selects parameter names with `name.startswith(prefix + ".")`. A prefix that is already a complete parameter name therefore matches nothing, so its four individual head keys display artificial zeros. The aggregate `grad_norm_dynamic_readout_module` is valid. Actual three-step and600-step full-state audits access each `p.grad` directly and are authoritative for individual tensors. Preserve the immutable logs; this observation neither changes running source nor warrants stopping healthy training. Framework logger repair is deferred.
