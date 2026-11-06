# Stochastic benchmark pilot

Synthetic sensitivity analysis only. No real plant cases or trained models.

20 independent draws per cause per scenario/profile, with all causes covered.
Metrics reweight the stratified draws to the stated environment prior; scenarios receive equal weight.
Intervals resample the 10 authored scenario groups (2.5th–97.5th percentiles). They do not establish industrial generalization.

| Profile | Method | Accuracy | Mean tests | Mean total loss (Rs) | Confident-error rate |
|---|---|---:|---:|---:|---:|
| combined | cheapest_first | 41.2% | 6.43 | 18707 | 6.1% |
| combined | cost_aware | 41.0% | 6.33 | 18680 | 7.0% |
| combined | cost_aware_voi | 40.2% | 3.38 | 14063 | 0.8% |
| combined | fixed_order | 41.6% | 6.30 | 18661 | 7.3% |
| combined | greedy_info | 41.4% | 6.17 | 18745 | 8.1% |
| combined | no_test | 14.4% | 0.00 | 17120 | 0.0% |
| combined | random | 41.3% | 6.42 | 19219 | 6.1% |
| correlated | cheapest_first | 78.1% | 5.74 | 9041 | 4.3% |
| correlated | cost_aware | 77.8% | 5.44 | 8862 | 5.0% |
| correlated | cost_aware_voi | 75.3% | 3.10 | 6589 | 0.4% |
| correlated | fixed_order | 78.8% | 5.43 | 9293 | 3.9% |
| correlated | greedy_info | 78.1% | 5.09 | 8850 | 5.2% |
| correlated | no_test | 25.5% | 0.00 | 14900 | 0.0% |
| correlated | random | 78.2% | 5.90 | 11081 | 4.0% |
| inconclusive | cheapest_first | 69.9% | 6.19 | 12317 | 4.6% |
| inconclusive | cost_aware | 69.8% | 6.07 | 12254 | 5.0% |
| inconclusive | cost_aware_voi | 68.3% | 3.48 | 8361 | 0.6% |
| inconclusive | fixed_order | 70.4% | 6.04 | 12429 | 4.3% |
| inconclusive | greedy_info | 70.1% | 5.84 | 12248 | 5.0% |
| inconclusive | no_test | 25.5% | 0.00 | 14900 | 0.0% |
| inconclusive | random | 70.4% | 6.27 | 13148 | 3.7% |
| likelihood_shift | cheapest_first | 56.2% | 6.04 | 14747 | 12.2% |
| likelihood_shift | cost_aware | 56.2% | 5.77 | 14440 | 14.0% |
| likelihood_shift | cost_aware_voi | 52.1% | 2.58 | 10918 | 1.1% |
| likelihood_shift | fixed_order | 56.9% | 5.75 | 14701 | 13.9% |
| likelihood_shift | greedy_info | 56.1% | 5.47 | 14317 | 15.3% |
| likelihood_shift | no_test | 25.5% | 0.00 | 14900 | 0.0% |
| likelihood_shift | random | 55.9% | 6.14 | 15871 | 10.5% |
| matched | cheapest_first | 76.2% | 5.84 | 9846 | 6.2% |
| matched | cost_aware | 76.0% | 5.58 | 9652 | 7.2% |
| matched | cost_aware_voi | 72.6% | 2.95 | 7046 | 0.6% |
| matched | fixed_order | 77.3% | 5.45 | 9886 | 6.9% |
| matched | greedy_info | 76.3% | 5.12 | 9443 | 8.7% |
| matched | no_test | 25.5% | 0.00 | 14900 | 0.0% |
| matched | random | 76.8% | 5.91 | 11380 | 6.0% |
| prior_shift | cheapest_first | 73.6% | 6.08 | 10774 | 5.1% |
| prior_shift | cost_aware | 73.7% | 5.82 | 10573 | 6.4% |
| prior_shift | cost_aware_voi | 72.1% | 3.18 | 7441 | 0.4% |
| prior_shift | fixed_order | 73.9% | 5.56 | 10546 | 7.8% |
| prior_shift | greedy_info | 73.4% | 5.26 | 10871 | 10.0% |
| prior_shift | no_test | 18.0% | 0.00 | 16400 | 0.0% |
| prior_shift | random | 73.4% | 6.02 | 12272 | 6.4% |
| unknown_cause | cheapest_first | 60.9% | 5.92 | 13252 | 9.2% |
| unknown_cause | cost_aware | 60.8% | 5.62 | 12991 | 12.0% |
| unknown_cause | cost_aware_voi | 58.1% | 2.77 | 9838 | 1.3% |
| unknown_cause | fixed_order | 61.9% | 5.61 | 13324 | 9.6% |
| unknown_cause | greedy_info | 61.0% | 5.27 | 12830 | 12.7% |
| unknown_cause | no_test | 20.4% | 0.00 | 15920 | 0.0% |
| unknown_cause | random | 61.6% | 6.01 | 14696 | 8.6% |

## Paired monetary VOI minus information-gain/cost

Negative loss difference favors monetary VOI; intervals crossing zero do not establish a consistent advantage across these templates.

| Profile | Mean total-loss difference (Rs) | Scenario-bootstrap 95% interval |
|---|---:|---:|
| combined | -4617 | [-6565, -2845] |
| correlated | -2273 | [-3261, -1339] |
| inconclusive | -3893 | [-5557, -2324] |
| likelihood_shift | -3522 | [-5132, -1992] |
| matched | -2606 | [-3662, -1529] |
| prior_shift | -3132 | [-4534, -1678] |
| unknown_cause | -3154 | [-4456, -1861] |

## Interpretation limits

- No uncertainty-aware proposed policy or escalation action is implemented yet.
- All current methods always output a candidate cause. A high-belief rate is not abstention coverage.
- Unknown-cause episodes cannot be correctly classified by this closed candidate set. Their failures are a deliberate stress test.
- Unknown outcomes are synthetic uniform draws, not observations of real omitted causes.
- Correlation uses shared inverse-CDF uniforms; strength and sign depend on outcome ordering. This is a sensitivity model, not a calibrated joint plant model.
- Inconclusive tests consume their full cost and cannot be repeated; belief stays unchanged. Policies do not anticipate missingness.
- Likelihood mismatch, missingness, correlations and unknown prevalence are authored intervention settings.
- Costs are summed serial test time; parallel testing and action-dependent consequences are outside this benchmark.
- Brier scores are unnormalized multiclass scores in [0,2]. Unknown truth is an additional class assigned probability zero; known-only scores are also reported.
- Confidence intervals cover variation among these authored templates; template count remains 10.
- No CAPA text quality, citation accuracy, external LLM or live UI evaluation is included.
- This is a pilot. More random draws reduce simulation noise but cannot replace independently sourced incidents.
