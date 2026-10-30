# Ordered query: CPU evidence

No model was trained. The labels use only existing RGB frames. Full implementation contract and prior-art limits: `S3_REFINED.md`.

The exhaustive oracle checks all 3,280 words over {A,B,O} through length7: oracle, composition and hard-symbol stuttering errors are all0. The five matrices are closed under all25 compositions. A correlated AA/BB mixture gives true ordered-hit0; independently multiplying its marginals gives0.25. This rejects a naive marginal-hazard implementation.

| Task | Queries | Interior shuffle changes result | Interior reversal changes result | Duplicate prototypes |
|---|---:|---:|---:|---:|
| tworoom | 1024 | 22.07% | 29.20% | 0.00% |
| cube | 1024 | 18.36% | 25.68% | 0.00% |

Each task uses128 existing training windows,8 random prototype queries per window. Initial/final symbols are held fixed during the diagnostic shuffle/reversal. The perturbed symbol paths are never treated as action-conditioned training outcomes.

**Necessary correction:** selecting all3 prototype frames from a complete window guarantees their occurrence, so its full-window target has only AB/BA classes. There the full transfer adds no information over scalar completion. The proposed objective therefore includes its2 factual half-segments under the same query. Their targets do contain transformations collapsed by one scalar.

| Task / segment | I | A | B | AB | BA |
|---|---:|---:|---:|---:|---:|
| tworoom / whole | 0 | 0 | 0 | 678 | 346 |
| tworoom / first_half | 66 | 280 | 258 | 250 | 170 |
| tworoom / second_half | 65 | 260 | 256 | 258 | 185 |
| cube / whole | 0 | 0 | 0 | 771 | 253 |
| cube / first_half | 51 | 214 | 224 | 293 | 242 |
| cube / second_half | 51 | 213 | 201 | 313 | 246 |

Under a scalar completion target I,A,B,BA are all0. Under an already-A prefix, B and BA complete the query while I and A do not. Across half-segments, both kinds actually occur:

- tworoom: incomplete but contains B = 42.43% of halves; incomplete without B = 32.76%.
- cube: incomplete but contains B = 44.58% of halves; incomplete without B = 25.83%.

This verifies an available compositional target distinction, not physical event semantics, a causal action effect, learning superiority or absolute novelty. Episode-level dependence remains; query counts are not independent experimental seeds. All targets and controls still need a native-planner gate before a performance claim.
