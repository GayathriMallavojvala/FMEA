# Validation — 2026-10-29

10 regression tests passed. Includes shared-policy equivalence across all 10
scenarios, state preservation, action regeneration/reopening, low-match rejection,
loss-change persistence, and report generation callbacks with a mocked LLM.
60 deterministic traces were produced (10 scenarios x 6 strategies).
All Python sources passed AST parsing; Gradio 6.30.0 app construction succeeded.
Gradio emitted non-fatal unclosed-event-loop resource warnings in tests.
Browser interaction, external LLM generation, and deployment were not tested.
Test process proxy variables were unset to work around a host missing socksio;
no application proxy configuration was changed.


## Step 2 addition
See `../STEP2_NOTES.md` and `stochastic_test_report.txt` for the stochastic
benchmark and 21-test regression suite. `stochastic/manifest.json` records the
actual experiment settings and source hashes. The original deterministic
results above remain a separate Step 1 check.
