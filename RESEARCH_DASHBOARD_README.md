---
title: Steel CAPA Copilot
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# Steel CAPA Copilot

## New: CAPA Effectiveness Lab (main research contribution)
Statistical design and Bayesian verification of CAPA effectiveness for steel defects. Tab 1 designs the
monitoring window, tab 2 gives a close / extend / reopen decision from weekly defect counts, tab 3 benchmarks
the method against fixed 30/60/90-day plant practice on simulated CAPAs. Code: `effectiveness/`. Tests:
`python -m unittest tests.test_effectiveness`. Synthetic data only; validate on real plant counts before claiming performance.

AI assistant for steel-plant quality engineers: guided root-cause investigation (picks the next
test by value for money), evidence-checked CAPA drafting, tracked actions, audit trail and Word export.
Research prototype: outputs are drafts for a qualified engineer to review.

## Reliable-baseline milestone
See `RESEARCH_PLAN.md` for implemented fixes, the next stages and limitations.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate_baseline.py
python app.py
```

The comparison includes monetary VOI (the guided method), information gain per
cost, maximum information gain, cheapest first, fixed order and random order.
All use the same authored priors. Comparison uses a fixed Rs 20,000 loss; the
guided workflow exposes an adjustable loss. Results are deterministic logic
checks under authored assumptions, not an industrial performance benchmark.

Regenerating a draft retains an existing tracked action plan. Review the revised
report manually before changing existing actions. Names in audit logs are
self-entered; this prototype has no authentication or role-based authorization.


## Step 2: stochastic benchmark

```bash
python evaluate_stochastic.py --replicates 20 --output results/new_run
```

The supplied pilot includes seven stress profiles and seven policies, paired
pre-sampled evidence, distinct simulator/policy models, calibration metrics and
scenario-grouped uncertainty intervals. See `benchmark/PROTOCOL.md` and
`results/stochastic/REPORT.md`. `replay_episode.py` replays stored evidence.
This command evaluates diagnosis only; it does not call an LLM or modify cases.
