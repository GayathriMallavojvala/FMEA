"""
Guided investigation workflow: the stateful layer that joins cost-aware diagnosis
to CAPA drafting. Pure Python (no LLM calls) so it is fully unit-testable.

A case `state` is a plain JSON-serialisable dict so it can live in gr.State:
    {"scenario_id", "belief": {cause: p}, "history": [{"test_id","outcome"}]}
"""
from copy import deepcopy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from active_diagnosis import engine

CONFIDENCE = engine.CONFIDENCE_THRESHOLD


def match_scenarios(text, scenarios, top_k=3):
    """Rank diagnostic scenarios by similarity to the engineer's description.
    Returns [(scenario_id, score)]. Limitation: only scenarios in the library
    can be diagnosed -- low similarity must be rejected by the calling UI."""
    ids = list(scenarios)
    docs = [f"{scenarios[i]['defect_type'].replace('_', ' ')}. {scenarios[i]['presenting_symptom']}"
            for i in ids]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(docs + [text])
    sims = cosine_similarity(vec.transform([text]), vec.transform(docs))[0]
    ranked = sorted(zip(ids, sims), key=lambda kv: -kv[1])[:top_k]
    return [(i, float(s)) for i, s in ranked]


DEFAULT_LOSS_INR = 20000.0  # assumed cost of acting on a wrong root cause; user-adjustable


def new_case(scenario, loss_inr=DEFAULT_LOSS_INR):
    """Start belief from the scenario's prior (authored relative frequencies) if present."""
    causes = [c["id"] for c in scenario["candidate_causes"]]
    prior = scenario.get("prior") or {}
    return {"scenario_id": scenario["id"],
            "belief": engine.normalize({c: prior.get(c, 1.0) if prior else 1.0 for c in causes}),
            "history": [], "loss_inr": float(loss_inr)}


def _tests_left(state, scenario):
    used = {h["test_id"] for h in state["history"]}
    return [t for t in scenario["tests"] if t["id"] not in used]


def top_cause(state):
    cause, p = max(state["belief"].items(), key=lambda kv: kv[1])
    return cause, p


def is_conclusive(state):
    return top_cause(state)[1] >= CONFIDENCE


def recommend(state, scenario, cost_kwargs=None):
    """All remaining tests ranked by expected NET benefit in rupees: the expected drop in
    wrong-diagnosis cost (state['loss_inr'] x gain in P(correct)) minus the test's cost.
    Returns list of dicts, best first. net > 0 means the test is worth running."""
    return engine.rank_tests_voi(state["belief"], _tests_left(state, scenario),
                                 state.get("loss_inr", DEFAULT_LOSS_INR),
                                 **(cost_kwargs or {}))


def stage(state, scenario):
    """'conclusive' | 'continue' | 'stop_voi' | 'exhausted'"""
    if is_conclusive(state):
        return "conclusive"
    ranked = recommend(state, scenario)
    if not ranked:
        return "exhausted"
    return "continue" if ranked[0]["net"] > 0 else "stop_voi"


def record_result(state, scenario, test_id, outcome):
    test = next(t for t in scenario["tests"] if t["id"] == test_id)
    if outcome not in test["outcomes"]:
        raise ValueError(f"'{outcome}' is not a valid outcome for {test_id}")
    if any(h["test_id"] == test_id for h in state["history"]):
        raise ValueError(f"{test_id} was already recorded")
    updated = deepcopy(state)
    updated["belief"] = engine.bayes_update(state["belief"], test, outcome)
    updated["history"].append({"test_id": test_id, "outcome": outcome})
    return updated


def _label(scenario, cause_id):
    return next(c["label"] for c in scenario["candidate_causes"] if c["id"] == cause_id)


def belief_table(state, scenario):
    rows = sorted(state["belief"].items(), key=lambda kv: -kv[1])
    return [[_label(scenario, c), f"{p*100:.0f}%"] for c, p in rows]


def evidence_lines(state, scenario):
    out = []
    for h in state["history"]:
        t = next(t for t in scenario["tests"] if t["id"] == h["test_id"])
        out.append(f"{t['label']} -> {h['outcome'].replace('_', ' ')} "
                   f"(cost Rs {t['cost_inr']}, {t['time_minutes']} min)")
    return out


def spend(state, scenario):
    ts = [next(t for t in scenario["tests"] if t["id"] == h["test_id"]) for h in state["history"]]
    return sum(t["cost_inr"] for t in ts), sum(t["time_minutes"] for t in ts)


def next_steps(state, scenario):
    """Plain-language guidance on what to do next, given where the case stands."""
    cause, p = top_cause(state)
    st = stage(state, scenario)
    lead = f"**{_label(scenario, cause)}** ({p*100:.0f}%)"
    loss = state.get("loss_inr", DEFAULT_LOSS_INR)
    if st == "conclusive":
        return [f"Model confidence threshold reached; leading hypothesis: {lead}.",
                "Contain now: quarantine and re-inspect affected coils/batches before release.",
                "Generate the CAPA report and action plan below, then have the responsible engineer review and approve it.",
                "Confirm the cause on the plant floor (a confirmatory check or trial) before closing the CAPA."]
    if st == "continue":
        best = recommend(state, scenario)[0]
        t = best["test"]
        return [f"Not yet conclusive. Leading hypothesis: {lead}.",
                f"Run next: **{t['label']}** (about Rs {t['cost_inr']}, {t['time_minutes']} min; "
                f"expected net benefit about Rs {best['net']:,.0f}), then record the result.",
                "Keep containment in place while the investigation continues."]
    if st == "stop_voi":
        return [f"Stop testing: no remaining test is worth its cost if a wrong root cause costs about Rs {loss:,.0f}.",
                f"Best current explanation: {lead}.",
                "Seek engineer review and treat the CAPA as provisional and build a verification step into the corrective action. "
                "Raise the wrong-diagnosis cost above if you think the stakes are higher; the system will then suggest more tests.",
                "Contain affected material in the meantime."]
    ranked = sorted(state["belief"].items(), key=lambda kv: -kv[1])[:2]
    alts = "; ".join(f"{_label(scenario, c)} ({q*100:.0f}%)" for c, q in ranked)
    return [f"All available tests are used but the result is inconclusive. Leading hypotheses: {alts}.",
            "Escalate to a cross-functional team: the cause may be outside the candidate list.",
            "Add new candidate causes or tests and re-run, or treat the top two as parallel corrective actions."]


def build_capa_prompt(defect_text, state, scenario, context):
    cause, p = top_cause(state)
    ev = "\n".join(f"- [E{i}] {l}" for i, l in enumerate(evidence_lines(state, scenario), 1)) or "- (no tests recorded)"
    status = {"conclusive": "HIGH MODEL BELIEF (not independently verified)",
              "stop_voi": "MOST LIKELY (testing stopped: further tests not cost-justified)"}.get(
        stage(state, scenario), "MOST LIKELY (not yet conclusive)")
    return (
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"NEW DEFECT REPORT [CASE] (reported, not independently verified):\n{defect_text}\n\n"
        f"DIAGNOSTIC EVIDENCE GATHERED:\n{ev}\n\n"
        f"LEADING HYPOTHESIS ({status}, {p*100:.0f}% model belief): {_label(scenario, cause)}\n\n"
        "Cite case observations with [CASE] or [E1], [E2], etc. Cite reference facts with the supplied [REF:passage-id]. "
        "Never invent citation IDs. References support general facts, not proof of this case cause. "
        "Write a provisional CAPA. Separate observations, reference facts, hypotheses, "
        "proposed actions and verification criteria. Cite recorded evidence for observations. "
        "Do not force a 5-Why chain or invent causal links, dates, limits or procedural failures. "
        "Stop the causal chain where evidence ends and state what remains unresolved. "
        "Model belief is not verified causation. Consider competing hypotheses. "
        "Mark deadlines and thresholds as proposals requiring engineer approval. After section 5, add section "
        "'6. Action Plan' as a markdown table with EXACT columns: What | Where | When | Who | Required Results. "
        "Prefix each action with its type: Containment, Corrective, Preventive, Verification. "
        "When must be Pending approval unless a deadline is supplied in the case. Who is a suggested role, not a named assignee. "
        "Required Results must describe evidence to collect; never invent a measured result or numerical acceptance limit."
    )
