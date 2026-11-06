"""Separate outcome-generating model and immutable, paired episode evidence."""
from copy import deepcopy
import hashlib
import json
import math
import random

UNKNOWN = '__unknown_cause__'
INCONCLUSIVE = '__inconclusive__'
PROFILES = {
    'matched': {},
    'prior_shift': {'rotate_prior': True},
    'likelihood_shift': {'likelihood_mix': 0.30},
    'correlated': {'shared_uniform_probability': 0.60},
    'inconclusive': {'missing_probability': 0.20},
    'unknown_cause': {'unknown_prevalence': 0.20},
    'combined': {'rotate_prior': True, 'likelihood_mix': 0.30,
                 'shared_uniform_probability': 0.60, 'missing_probability': 0.20,
                 'unknown_prevalence': 0.20},
}


def stable_seed(*parts):
    raw = json.dumps(parts, separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def validate_scenario(sc):
    causes = [c['id'] for c in sc['candidate_causes']]
    if not causes or len(causes) != len(set(causes)):
        raise ValueError('Candidate IDs must be nonempty and unique')
    prior = sc.get('prior')
    if not prior or set(prior) != set(causes):
        raise ValueError('Benchmark requires a complete explicit prior')
    check_distribution(prior)
    ids = [t['id'] for t in sc['tests']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate test IDs')
    for test in sc['tests']:
        outcomes = test['outcomes']
        if not outcomes or len(outcomes) != len(set(outcomes)):
            raise ValueError('Outcome IDs must be nonempty and unique')
        if INCONCLUSIVE in outcomes:
            raise ValueError('Reserved outcome ID')
        if set(test['outcome_likelihoods']) != set(causes):
            raise ValueError('Missing cause likelihoods')
        for probabilities in test['outcome_likelihoods'].values():
            if set(probabilities) != set(outcomes):
                raise ValueError('Missing outcome likelihoods')
            check_distribution(probabilities)
        for field in ['cost_inr', 'time_minutes']:
            if not math.isfinite(test[field]) or test[field] < 0:
                raise ValueError('Invalid test cost or duration')


def check_distribution(values):
    if not values or not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError('Invalid probability')
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-8):
        raise ValueError('Probabilities must sum to one')


def models(sc, profile):
    """Policy retains authored likelihoods; environment may have different ones.
    Rotated priors are a fixed sensitivity intervention, not fitted estimates.
    """
    settings = PROFILES[profile]
    policy = deepcopy(sc)
    policy.pop('true_cause', None)
    causes = [c['id'] for c in sc['candidate_causes']]
    if settings.get('rotate_prior'):
        probabilities = [sc['prior'][c] for c in causes]
        policy['prior'] = dict(zip(causes, probabilities[-1:] + probabilities[:-1]))
    prevalence = settings.get('unknown_prevalence', 0)
    prior = {c: (1-prevalence)*sc['prior'][c] for c in causes}
    if prevalence:
        prior[UNKNOWN] = prevalence
    likelihoods = {}
    mix = settings.get('likelihood_mix', 0)
    for test in sc['tests']:
        outcomes = test['outcomes']
        likelihoods[test['id']] = {
            c: {o: (1-mix)*test['outcome_likelihoods'][c][o] + mix/len(outcomes)
                for o in outcomes} for c in causes}
        if prevalence:
            likelihoods[test['id']][UNKNOWN] = {o: 1/len(outcomes) for o in outcomes}
    environment = dict(scenario_id=sc['id'], profile=profile, prior=prior,
                       likelihoods=likelihoods, settings=deepcopy(settings))
    return policy, environment


def draw(probabilities, u):
    cumulative = 0
    for outcome, probability in probabilities.items():
        cumulative += probability
        if u < cumulative:
            return outcome
    return next(reversed(probabilities))


def sample_episode(environment, cause, replicate, seed, replicates):
    """Stratified draws cover every cause. Weight reconstructs environment prior.
    Common random numbers across profiles where cause/test IDs match.
    Each policy receives only the outcomes it requests through a callback.
    """
    sid = environment['scenario_id']
    base = (seed, sid, cause, replicate)
    shared = random.Random(stable_seed(*base, 'shared')).random()
    settings = environment['settings']
    outcomes = {}
    for tid, likelihoods in environment['likelihoods'].items():
        rng = random.Random(stable_seed(*base, tid))
        independent, use_shared, missing = rng.random(), rng.random(), rng.random()
        u = shared if use_shared < settings.get('shared_uniform_probability', 0) else independent
        outcomes[tid] = (INCONCLUSIVE if missing < settings.get('missing_probability', 0)
                         else draw(likelihoods[cause], u))
    eid = f"{environment['profile']}:{sid}:{cause}:{replicate}"
    return dict(episode_id=eid, scenario_id=sid, profile=environment['profile'],
                true_cause=cause, replicate=replicate,
                scenario_weight=environment['prior'][cause]/replicates,
                outcomes=outcomes)
