"""Simulation benchmark: Bayesian sequential verification vs. fixed-window plant practice."""
import numpy as np
import pandas as pd
from scipy import stats
from .engine import draw_counts, prob_vec, MAX_PHI

SCENARIOS = {"Strong effect (70%)": 0.70, "Borderline (55%)": 0.55, "Weak effect (30%)": 0.30, "No effect (0%)": 0.0}
METHODS = ["Fixed 30 days: point estimate", "Fixed 60 days: point estimate", "Fixed 90 days: point estimate",
           "Fixed 90 days: significance test", "Bayesian sequential (proposed)"]
FIX = {METHODS[0]: 4, METHODS[1]: 8, METHODS[2]: 13}


def run(base_rate=6.0, target=0.5, conf=0.90, cv=0.35, pre_weeks=12, reps=1000, max_weeks=26, min_weeks=4, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for sname, eff in SCENARIOS.items():
        truth = eff >= target - 1e-9
        pre = draw_counts(rng, base_rate, pre_weeks, cv, reps)
        post = draw_counts(rng, base_rate * (1 - eff), max_weeks, cv, reps)
        phi = np.clip(pre.var(1, ddof=1) / np.maximum(pre.mean(1), 1e-9), 1, MAX_PHI)
        cum = np.cumsum(post, 1)
        pm = pre.mean(1)
        res = {}
        for m, w in FIX.items():                       # plant practice: close if observed drop >= target
            res[m] = (post[:, :w].mean(1) <= (1 - target) * pm, np.full(reps, w))
        w = 13                                          # exact conditional test against the SAME target as the proposed method
        s0, s1 = pre.sum(1), cum[:, w - 1]
        q = w * (1 - target)
        p = stats.binom.cdf(s1, s0 + s1, q / (q + pre_weeks))   # H0: post rate = (1-target) x pre rate
        res[METHODS[3]] = (p < 0.05, np.full(reps, w))
        closed, when = np.zeros(reps, bool), np.full(reps, max_weeks)
        reopened = np.zeros(reps, bool)
        for n in range(min_weeks, max_weeks + 1):       # proposed: weekly Bayesian look, stop when decided
            pr = prob_vec(pre.sum(1), pre_weeks, cum[:, n - 1], n, target, phi)
            live = ~closed & ~reopened
            c, r = live & (pr >= conf), live & (pr <= 0.05)
            closed |= c
            reopened |= r
            when[c | r] = n
        res[METHODS[4]] = (closed, when)
        for m, (cl, wk) in res.items():
            rows.append(dict(method=m, scenario=sname, truly_effective=truth, closed=float(cl.mean()),
                             weeks=float(wk.mean())))
    df = pd.DataFrame(rows)
    summ = []
    for m in METHODS:
        d = df[df.method == m]
        summ.append(dict(method=m,
                         false_closure=d[~d.truly_effective].closed.mean(),
                         missed_effect=1 - d[d.truly_effective].closed.mean(),
                         mean_weeks=d.weeks.mean()))
    return df, pd.DataFrame(summ)
