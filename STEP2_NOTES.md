# Step 2 delivery — 29 October 2026

Implemented a stochastic benchmark, not a new diagnosis algorithm.

## Pilot evidence
- 10 authored scenario templates.
- 7 profiles: matched, prior shift, likelihood shift, correlated tests,
  inconclusive tests, unknown cause, and combined stress.
- 20 replicates per true cause and scenario/profile.
- 8,800 sampled episodes, each shared across 7 baseline policies.
- 61,600 policy runs. These are not 61,600 independent industrial cases.

| Profile | Policy | Weighted accuracy | Mean modeled total loss (Rs) |
|---|---|---:|---:|
| Matched | Monetary VOI | 72.6% | 7,046 |
| Matched | Information gain per cost | 76.0% | 9,652 |
| Combined stress | Monetary VOI | 40.2% | 14,063 |
| Combined stress | Information gain per cost | 41.0% | 18,680 |

Under the authored loss assumptions, monetary VOI reduces modeled cost relative
to IG/cost, with an accuracy trade-off. This observation is expected to depend on
the configured losses and simulator. It establishes neither algorithmic novelty
nor industrial usefulness. Both methods deteriorate severely under combined
stress. Unknown causes cannot be named by either policy because the candidate
set has no unknown class. Low confident-error rate alone is not success: a
policy can simply stop at low belief while still making a wrong decision.

## Changed files
- `active_diagnosis/engine.py`: final probability vector, stop reason,
  prior-only baseline, explicit inconclusive outcomes and outcome validation.
  Existing guided monetary VOI selection is retained.
- `benchmark/simulation.py`: separate models, authored stress configurations,
  paired outcomes, stable seeds and stratified sampling weights.
- `benchmark/evaluation.py`: truth-isolated policy calls, costs and scoring.
- `benchmark/statistics.py`: weighted metrics, calibration bins, scenario-cluster
  intervals and paired comparisons.
- `evaluate_stochastic.py`: CLI runner and persisted configuration/results.
- `replay_episode.py`: replay saved evidence under a selected policy.
- `validate_benchmark.py`: audit stored traces, weights, metrics and replay.
- `tests/test_stochastic.py`: 11 additional regression tests.
- README, research plan and benchmark protocol explain usage and limitations.

## Validation
21 tests pass, including the original 10 baseline tests. The output audit checks
all 61,600 traces against shared episode evidence and costs, verifies per-scenario
weights and performs 49 exact replays (one per profile/policy). See validation
logs. No external LLM, browser interaction or live plant validation was performed.

## Next milestone
Build the provenance registry and leakage-safe benchmark split before adding a
proposed uncertainty-aware policy. The project owner must establish access to a
qualified steel/metallurgy reviewer and usable incident evidence. Do not relabel
authored scenarios or synthetic unknown outcomes as real plant data.
