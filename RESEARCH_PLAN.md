# Research specification and staged implementation

## Working question
Can diagnostic test selection reduce total investigation loss while controlling
wrong decisions when the supplied probabilities are inaccurate?

Primary research unit: independently sourced incident/scenario, not a generated
Monte Carlo episode. Current library: 10 authored scenarios. Current results are
logic checks, not measured plant performance, novelty evidence or calibration.

## Step 1 — reliable baseline (implemented)
- Shared monetary VOI ranking used by guided workflow and evaluation.
- Six comparison policies use authored priors, Rs 20,000 wrong-decision loss,
  Rs 20/min time cost and 0.85 model-belief stopping threshold.
- Session extensions survive result updates without mutating the input state.
- Low similarity (<0.15, heuristic and uncalibrated) abstains before case creation.
- Loss-slider changes recompute and persist recommendations immediately.
- High belief is explicitly distinct from verified causation; no forced 5-Why.
- Guided verifier receives original case description and recorded test evidence.
- Report regeneration retains existing action IDs, status and ownership. A revised
  report's plan is NOT automatically imported over an existing plan; review it.
- Reopened actions reset case status; report regeneration preserves action status.
- Original case description stays attached to the investigation even if the input
  textbox is edited later. Start a new investigation to change the case scope.
- Reproducible runner, regression tests and configuration manifest included.

## Step 2 — stochastic benchmark (implemented synthetic pilot)
Separate policy/environment models, pre-sampled paired outcomes, all candidate
causes, seven stress profiles, seven baseline policies, weighted metrics,
scenario-cluster intervals and per-episode replay are implemented. See
`benchmark/PROTOCOL.md` and `results/stochastic/REPORT.md`. This remains synthetic:
probabilities, costs and stress settings are authored, not measured or validated.
Pilot sample counts quantify draws, not independent industrial evidence.

## Step 3 — evidence and provenance
Build an incident registry with source document, incident ID, initial information,
requested tests, revealed results, established cause and acceptable actions.
Label every field observed / expert-estimated / simulated. Keep unresolved fields
missing. Split by source incident/document; prevent retrieval of held-out answers.
Preserve passage IDs/URLs in retrieval; add relevance rejection, competing-cause
retrieval and citation/contradiction evaluation. Audit the non-steel references.

## Step 4 — proposed method
Implement action-dependent loss as an established baseline, then uncertain
priors/likelihoods and explicit escalation. Compare average-over-models and
conservative policies. Add two-step lookahead only if pilot failures justify it.
Predeclare settings and ablations before held-out evaluation. No novelty claim yet.

## Step 5 — validation and manuscript
Blinded engineer review with a fixed rubric: unsupported factual claims, citation
correctness, contradictions and action suitability. Report reviewer disagreement
and correction time. Compare no RAG, RAG, RAG + evidence and RAG + verification,
controlling generation model/budget. External LLM judgments are supplementary.

## Inputs still required from project owner
1. Review/submission/publication deadlines and required publication category.
2. Whether this is the approved thesis or an alternative project.
3. Available metallurgy/quality reviewer and permission to use incident records.
These affect the study scope; they do not block the software fixes above.

## Remaining limitations
No authenticated users or role enforcement; audit actors are self-entered.
No effectiveness approval workflow. Lexical/LLM faithfulness is heuristic, not a
factual guarantee. No claim repair/gating, calibrated abstention, action-dependent
loss, robust policy or real-data validation is implemented in this milestone.
The 0.15 scenario cutoff can reject valid descriptions and accept unsuitable ones.
The original demo still uses most-likely outcomes. The new stochastic benchmark
is separate; neither establishes real-plant accuracy.
