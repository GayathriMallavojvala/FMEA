"""Sequential diagnosis baselines using authored probabilities.
Information gain per cost and monetary VOI are distinct established rules.
Default runs are deterministic demonstrations, not industrial validation.
"""
import math
import zlib

CONFIDENCE_THRESHOLD = 0.85


def entropy(belief):
    """Shannon entropy of a probability distribution (dict of prob values)."""
    return -sum(p * math.log2(p) for p in belief.values() if p > 0)


def cost_score(cost_inr, time_minutes, money_weight=1.0, time_weight_inr_per_min=20.0):
    """
    Combines money and time into a single cost score, in effective rupees.
    time_weight_inr_per_min is the assumed "cost" of an extra minute of
    diagnostic downtime -- exposed as a parameter, not hardcoded silently,
    since this assumption materially affects which tests look cheap. Default
    (₹20/min ≈ ₹1200/hour) is a placeholder representative of lost
    production-line time; a real deployment should set this from the actual
    plant's downtime cost.
    """
    return money_weight * cost_inr + time_weight_inr_per_min * time_minutes


def normalize(d):
    total = sum(d.values())
    if total == 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: v / total for k, v in d.items()}


def bayes_update(belief, test, outcome):
    """Posterior belief after observing `outcome` for `test`, via Bayes' rule."""
    likelihoods = test["outcome_likelihoods"]
    unnormalized = {
        cause: belief[cause] * likelihoods[cause].get(outcome, 1e-9)
        for cause in belief
    }
    return normalize(unnormalized)


def expected_info_gain(belief, test):
    """Expected reduction in entropy from running `test`, before knowing the
    actual outcome (averaging over the possible outcomes weighted by their
    predicted probability under the current belief)."""
    prior_entropy = entropy(belief)
    likelihoods = test["outcome_likelihoods"]
    outcomes = test["outcomes"]

    outcome_probs = {}
    for outcome in outcomes:
        outcome_probs[outcome] = sum(belief[c] * likelihoods[c].get(outcome, 1e-9) for c in belief)

    expected_posterior_entropy = 0.0
    for outcome, p_outcome in outcome_probs.items():
        if p_outcome <= 0:
            continue
        posterior = bayes_update(belief, test, outcome)
        expected_posterior_entropy += p_outcome * entropy(posterior)

    return prior_entropy - expected_posterior_entropy


def choose_next_test_cost_aware(belief, available_tests, **cost_kwargs):
    """The engine's core decision rule: highest (info_gain / cost) among
    remaining tests. Returns (test, info_gain, cost, value) or None if no
    tests remain."""
    best = None
    for test in available_tests:
        info_gain = expected_info_gain(belief, test)
        cost = cost_score(test["cost_inr"], test["time_minutes"], **cost_kwargs)
        value = info_gain / cost if cost > 0 else info_gain
        if best is None or value > best[3]:
            best = (test, info_gain, cost, value)
    return best


def expected_max_posterior(belief, test):
    """E over outcomes of the max posterior belief = expected probability of a CORRECT
    diagnosis if we stopped right after this test."""
    likelihoods = test["outcome_likelihoods"]
    total = 0.0
    for outcome in test["outcomes"]:
        p_out = sum(belief[c] * likelihoods[c].get(outcome, 1e-9) for c in belief)
        if p_out > 0:
            total += p_out * max(bayes_update(belief, test, outcome).values())
    return total


def rank_tests_voi(belief, available_tests, loss_inr, **cost_kwargs):
    """Shared monetary VOI ranking for guided UI and evaluation."""
    if not math.isfinite(loss_inr) or loss_inr < 0:
        raise ValueError("loss_inr must be finite and nonnegative")
    rows = []
    for test in available_tests:
        cost = cost_score(test["cost_inr"], test["time_minutes"], **cost_kwargs)
        benefit = loss_inr * (expected_max_posterior(belief, test) - max(belief.values()))
        rows.append(dict(test=test, info_gain=expected_info_gain(belief, test),
                         cost_score=cost, benefit=benefit, net=benefit-cost))
    return sorted(rows, key=lambda row: -row["net"])


def choose_next_test_voi(belief, available_tests, loss_inr, **cost_kwargs):
    """Value-of-information rule. A wrong diagnosis costs `loss_inr`. Running a test is
    worth it iff  loss * (E[max posterior] - max prior)  >  its cost.  Returns
    (test, net_value_inr) for the best test, or None when NO test is worth running
    (so the investigation should stop and act on the current belief).
    Myopic (one-step) by design -- see README/roadmap for the lookahead limitation."""
    rows = rank_tests_voi(belief, available_tests, loss_inr, **cost_kwargs)
    return (rows[0]["test"], rows[0]["net"]) if rows and rows[0]["net"] > 0 else None


def choose_next_test_greedy_info(belief, available_tests):
    """Baseline: ignores cost entirely, always picks the most informative
    test regardless of price -- the strategy a purely accuracy-driven (not
    cost-aware) system would use."""
    best = None
    for test in available_tests:
        info_gain = expected_info_gain(belief, test)
        if best is None or info_gain > best[1]:
            best = (test, info_gain)
    return best[0] if best else None


def run_diagnosis(scenario, strategy="cost_aware", true_outcome_fn=None,
                   confidence_threshold=CONFIDENCE_THRESHOLD, cost_kwargs=None, rng=None, prior=None, loss_inr=None, inconclusive_outcome=None):
    """
    Runs one full diagnostic episode on a scenario dict (see
    diagnosis_library.json) under the given strategy, and returns a trace:
    {
        final_diagnosis, correct (bool), tests_used (list of ids),
        total_cost_inr, total_time_minutes, steps (list of per-step detail)
    }

    Inconclusive outcomes (explicit opt-in sentinel) incur cost and consume the test
    without changing belief. no_test is a prior-only decision baseline.
    strategy: "cost_aware" | "cost_aware_voi" (needs loss_inr) | "greedy_info" | "cheapest_first" | "fixed_order" | "random"
    true_outcome_fn: function(test) -> outcome string. If None, uses the
        deterministic "most likely outcome under the true cause" (see `deterministic_outcome`), which is what the harness
        uses for reproducible evaluation.
    """
    if strategy == "cost_aware_voi" and loss_inr is None:
        raise ValueError("cost_aware_voi requires loss_inr")
    strategies = {"cost_aware", "cost_aware_voi", "greedy_info", "cheapest_first", "fixed_order", "random", "no_test"}
    if strategy not in strategies:
        raise ValueError(f"unknown strategy {strategy}")
    if not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    cost_kwargs = cost_kwargs or {}
    if prior is None:
        prior = scenario.get("prior")
    causes = [c["id"] for c in scenario["candidate_causes"]]
    belief = normalize({c: (prior or {}).get(c, 1.0) if prior else 1.0 for c in causes})
    remaining_tests = list(scenario["tests"])
    used_tests = []
    total_cost, total_time = 0.0, 0.0
    steps = []

    import random
    rnd = rng if rng is not None else random.Random(zlib.crc32(scenario["id"].encode()))

    stop_reason = "no_test" if strategy == "no_test" else "exhausted"
    while remaining_tests and strategy != "no_test":
        top_cause, top_belief = max(belief.items(), key=lambda kv: kv[1])
        if top_belief >= confidence_threshold:
            stop_reason = "confidence_threshold"
            break

        if strategy == "cost_aware_voi":
            choice = choose_next_test_voi(belief, remaining_tests, loss_inr, **cost_kwargs)
            if choice is None:
                stop_reason = "stop_voi"
                break  # Myopic economic stopping, not causal confirmation.
            test = choice[0]
        elif strategy == "cost_aware":
            choice = choose_next_test_cost_aware(belief, remaining_tests, **cost_kwargs)
            test = choice[0] if choice else None
        elif strategy == "greedy_info":
            test = choose_next_test_greedy_info(belief, remaining_tests)
        elif strategy == "fixed_order":
            test = remaining_tests[0]
        elif strategy == "cheapest_first":
            test = min(remaining_tests, key=lambda t: cost_score(t["cost_inr"], t["time_minutes"], **cost_kwargs))
        elif strategy == "random":
            test = rnd.choice(remaining_tests)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        if test is None:
            break

        outcome = true_outcome_fn(test) if true_outcome_fn else deterministic_outcome(scenario, test)
        prior_belief_snapshot = dict(belief)
        if inconclusive_outcome is not None and outcome == inconclusive_outcome:
            pass  # Paid, consumed test; no evidence update under this missing-result model.
        elif outcome not in test["outcomes"]:
            raise ValueError(f"Invalid outcome {outcome!r} for {test['id']}")
        else:
            belief = bayes_update(belief, test, outcome)

        total_cost += test["cost_inr"]
        total_time += test["time_minutes"]
        used_tests.append(test["id"])
        remaining_tests.remove(test)
        steps.append({
            "test_id": test["id"],
            "outcome": outcome,
            "belief_before": prior_belief_snapshot,
            "belief_after": dict(belief),
        })

    final_cause, final_belief = max(belief.items(), key=lambda kv: kv[1])
    if strategy != "no_test" and final_belief >= confidence_threshold:
        stop_reason = "confidence_threshold"
    return {
        "final_belief": dict(belief),
        "stop_reason": stop_reason,
        "final_diagnosis": final_cause,
        "final_confidence": final_belief,
        "correct": final_cause == scenario["true_cause"],
        "tests_used": used_tests,
        "num_tests_used": len(used_tests),
        "total_cost_inr": total_cost,
        "total_time_minutes": total_time,
        "steps": steps,
    }


def deterministic_outcome(scenario, test):
    """Default simulated test result: the MOST LIKELY outcome under the
    scenario's true cause. Deterministic (not sampled) so evaluation results
    are exactly reproducible run to run. This is not a stochastic benchmark."""
    true_cause = scenario["true_cause"]
    likelihoods = test["outcome_likelihoods"][true_cause]
    return max(likelihoods.items(), key=lambda kv: kv[1])[0]
