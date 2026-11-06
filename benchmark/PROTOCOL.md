# Synthetic benchmark protocol (Step 2)

## Scope
This experiment tests known policies under authored assumptions and deliberate
model errors. It does not test a proposed uncertainty-aware policy, validate the
likelihoods, measure engineer usefulness, or establish real-world accuracy.
The study has 10 authored scenario templates, not thousands of independent cases.
The pre-existing deterministic runner remains a separate logic demonstration.

## Reproduce
From the project directory:

```bash
python -m unittest discover -s tests -v
python evaluate_stochastic.py --replicates 20 --seed 20261029 --bootstrap 2000 --output results/new_run
```

The output folder must be empty. The runner refuses to overwrite an existing run.
No LLM calls, API key, GPU or new Python dependencies are required. The actual
pilot is in `results/stochastic`. Increase replicates to reduce Monte Carlo noise,
not to claim additional independent incidents.

## Policy and environment separation
`models.json` records complete copies of both models for every scenario/profile.
The policy never receives the environment likelihoods or true cause. Its only
outcome access is a callback for the selected test. A regression test confirms
that changing the scoring truth while holding evidence fixed cannot change the
policy's selected tests or posterior. The outcome callback prevents the runner
from falling back to the original most-likely-outcome simulator.

For each scenario/profile, every true cause gets 20 replicates. Each replicate
pre-samples outcomes for ALL tests. Every policy sees the same outcome if it
requests the same test, regardless of order. Stable SHA-256-derived seeds avoid
Python hash randomization. Random-order policies have a separate seeded stream.
Common random numbers also couple matching cause/test IDs across profiles; the
reported pairwise intervals compare policies within a profile only.

This is stratified sampling. A row has within-scenario weight
`environment_prior[true_cause] / replicates`. The sum is one per scenario/method.
All scenario templates have equal weight in aggregate metrics. Thus matched
simulation uses the authored prior, despite equal draw counts per cause.

## Profiles
| Profile | Environment change | Policy assumption |
|---|---|---|
| matched | Sample authored outcome distributions | Authored prior and likelihoods |
| prior_shift | Authored prevalence and tests unchanged | Prior values cyclically rotated over cause IDs |
| likelihood_shift | Each conditional distribution becomes 0.7 * authored + 0.3 * uniform | Original likelihoods |
| correlated | Each test uses a common episode uniform with probability 0.6, otherwise its independent uniform | Tests treated as conditionally independent |
| inconclusive | Each requested test has 0.2 probability of an inconclusive result | Cost paid, test consumed, belief unchanged; missingness not anticipated |
| unknown_cause | Unknown prevalence 0.2, known priors scaled to 0.8; unknown test outcomes uniform | Unknown class absent |
| combined | All the above interventions together | Rotated prior, original likelihoods, no unknown class |

The 0.6 correlation setting is a shared-uniform probability, NOT a Pearson
correlation coefficient. It preserves each test's marginal distribution, but
correlation direction depends on categorical outcome ordering. Inconclusiveness
is independent of cause/test results. Neither mechanism is expert-calibrated.
Unknown episodes are explicitly synthetic; they do not represent observed plant
failure modes. An always-known-class policy necessarily fails those episodes.

## Policies and costs
The six existing methods plus a prior-only `no_test` baseline are evaluated.
Monetary VOI calls the shared engine ranking also used by the guided workflow.
All test-using policies share a 0.85 belief threshold; VOI can stop earlier when
no remaining single test has positive net value. `no_test` always decides at the
prior. These different stopping rules are intrinsic to the baseline definitions.
No policy abstains or escalates. Do not call threshold attainment "coverage".

Default total loss = direct test cost + 20 * serial test minutes +
20,000 * wrong-decision indicator. Components are saved separately. CLI settings
`--loss-inr`, `--time-weight`, and `--threshold` permit sensitivity runs, which must
be reported as separate experiments. No escalation cost applies because there
is no escalation action yet. Action-dependent loss belongs to a later milestone.

## Metrics and uncertainty
- Accuracy: weighted correct-decision fraction, not raw unweighted draw accuracy.
- Confident-error rate: weighted fraction both wrong and at/above 0.85 belief.
- Error given confident: weighted errors among threshold-reaching decisions.
- Brier: sum of squared errors over cause probabilities, not divided by class
  count (range 0–2). For unknown truth, append a class with predicted probability
  zero. Known-only Brier is also included.
- Calibration: 10 equal-width confidence bins; report bin weights, accuracy,
  confidence and expected calibration error. Empty bins remain empty.
- Test number, direct cost, time, effective test cost, decision loss and total loss.

Paired differences are monetary VOI minus each comparator on the same episodes.
First compute weighted differences within each scenario, then average scenarios.
The 95% percentile interval resamples 10 whole scenario estimates 2,000 times.
There is no iid bootstrap over thousands of synthetic episodes. These intervals
are descriptive sensitivity intervals across the authored templates, NOT evidence
that templates are a random sample of steel plants. They do not separately
quantify finite-draw Monte Carlo uncertainty; repeat with more draws/seeds before
strong synthetic claims. No multiplicity-adjusted hypothesis tests are reported.

## Outputs
- `episodes.jsonl.gz`: every true cause, weight, and pre-sampled test outcome.
- `traces.jsonl.gz`: every policy's selected tests and posterior sequence.
- `episodes_metrics.csv`: all per-policy episode metrics.
- `summary.csv`: aggregate metrics and selected scenario-bootstrap intervals.
- `paired_comparisons.csv`: paired differences and intervals.
- `calibration.csv`: reliability bins.
- `models.json`: separate policy and simulator models.
- `manifest.json`: seed, configuration, counts, Python version and source hashes.
- `REPORT.md`: human-readable pilot table and limitations.

Replay any ID copied from `episodes_metrics.csv`:

```bash
python replay_episode.py results/stochastic 'EPISODE_ID' --strategy cost_aware_voi
```

Replaying uses stored models/outcomes but the current engine. Compare source
hashes in the manifest before treating it as an exact historical reproduction.
Next evidence milestone: traceable incidents with leakage-safe splits and expert
review. Novel policy development should follow identified failures, not precede
measurement of them.
