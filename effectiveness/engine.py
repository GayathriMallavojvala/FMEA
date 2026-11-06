"""Bayesian design and verification of CAPA effectiveness (Gamma-Poisson, dispersion-aware).

Question answered: after a corrective action, did the weekly defect rate fall by at least the
promised target, with enough evidence to close the CAPA? Pure numpy/scipy, no LLM involved.
"""
import numpy as np
from scipy import stats

A0 = 0.5                      # weak Jeffreys-style Gamma prior shape (rate ~ 0)
MAX_PHI = 5.0


def dispersion(pre):
    pre = np.asarray(pre, float)
    if len(pre) < 3 or pre.mean() == 0:
        return 1.0
    return float(np.clip(pre.var(ddof=1) / pre.mean(), 1.0, MAX_PHI))


def prob_reduction(pre, post, target, phi=None):
    """P(post rate <= (1-target) x pre rate). Exact via the beta-prime identity."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    phi = dispersion(pre) if phi is None else phi
    a0, b0 = A0 + pre.sum() / phi, len(pre) / phi
    a1, b1 = A0 + post.sum() / phi, max(len(post), 1e-9) / phi
    y = (1 - target) * b1 / b0
    return float(stats.beta.cdf(y / (1 + y), a1, a0))


def prob_vec(pre_sum, pre_n, post_sum, post_n, target, phi):
    """Vectorised version used by the simulator (arrays of sums)."""
    a0, b0 = A0 + pre_sum / phi, pre_n / phi
    a1, b1 = A0 + post_sum / phi, post_n / phi
    y = (1 - target) * b1 / b0
    return stats.beta.cdf(y / (1 + y), a1, a0)


def verify(pre, post, target=0.5, conf=0.90, min_weeks=4, max_weeks=26, futility=0.05, seed=1):
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    phi = dispersion(pre)
    p = prob_reduction(pre, post, target, phi)
    if len(post) < min_weeks:
        dec, why = "MONITOR", f"Only {len(post)} of the minimum {min_weeks} weeks observed."
    elif p >= conf:
        dec, why = "CLOSE", f"{p:.0%} probability the reduction reaches {target:.0%} (needs {conf:.0%})."
    elif p <= futility:
        dec, why = "REOPEN", f"Only {p:.0%} probability the target is met; the action is not working."
    elif len(post) >= max_weeks:
        dec, why = "ESCALATE", "Window exhausted without a clear answer; review the action."
    else:
        dec, why = "EXTEND", f"Evidence is not conclusive yet ({p:.0%}); keep monitoring."
    rng = np.random.default_rng(seed)
    l0 = rng.gamma(A0 + pre.sum() / phi, phi / max(len(pre), 1), 20000)
    l1 = rng.gamma(A0 + post.sum() / phi, phi / max(len(post), 1), 20000)
    red = 1 - l1 / l0
    lo, hi = np.percentile(red, [5, 95])
    return dict(decision=dec, reason=why, prob=p, phi=phi, pre_rate=pre.mean(), post_rate=post.mean(),
                reduction_mean=float(np.median(red)), ci_low=float(lo), ci_high=float(hi))


def draw_counts(rng, mean, weeks, cv, size):
    """Weekly defect counts with extra-Poisson variation (gamma frailty, coefficient of variation cv)."""
    k = 1 / max(cv, 1e-3) ** 2
    return rng.poisson(mean * rng.gamma(k, 1 / k, (size, weeks)))


def design(base_rate, target=0.5, expected=0.7, conf=0.90, cv=0.35, pre_weeks=12,
           reps=1500, max_weeks=26, seed=7):
    """Smallest monitoring window n that closes a truly effective CAPA >=80% of the time
    while falsely closing an ineffective one <=5% of the time."""
    rng = np.random.default_rng(seed)
    pre_e = draw_counts(rng, base_rate, pre_weeks, cv, reps)
    pre_n = draw_counts(rng, base_rate, pre_weeks, cv, reps)
    phi_e = np.clip(pre_e.var(1, ddof=1) / np.maximum(pre_e.mean(1), 1e-9), 1, MAX_PHI)
    phi_n = np.clip(pre_n.var(1, ddof=1) / np.maximum(pre_n.mean(1), 1e-9), 1, MAX_PHI)
    post_e = np.cumsum(draw_counts(rng, base_rate * (1 - expected), max_weeks, cv, reps), 1)
    post_n = np.cumsum(draw_counts(rng, base_rate, max_weeks, cv, reps), 1)
    rows, best = [], None
    for n in range(2, max_weeks + 1):
        pw = float((prob_vec(pre_e.sum(1), pre_weeks, post_e[:, n - 1], n, target, phi_e) >= conf).mean())
        fc = float((prob_vec(pre_n.sum(1), pre_weeks, post_n[:, n - 1], n, target, phi_n) >= conf).mean())
        rows.append(dict(weeks=n, power=pw, false_close=fc))
        if best is None and pw >= 0.8 and fc <= 0.05:
            best = n
    return best, rows
