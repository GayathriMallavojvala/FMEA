"""Policy adapter and metrics. Simulator truth is never passed to selectors."""
from copy import deepcopy
import random
from active_diagnosis import engine
from benchmark.simulation import INCONCLUSIVE, UNKNOWN, stable_seed

STRATEGIES = ['cost_aware_voi', 'cost_aware', 'greedy_info', 'cheapest_first',
              'fixed_order', 'random', 'no_test']


def score_brier(belief, true_cause):
    # Unnormalized multiclass Brier score, range [0, 2]. An unmodeled true class
    # has predicted probability zero, contributing an extra 1 to the score.
    return sum((p-float(c == true_cause))**2 for c, p in belief.items()) + float(true_cause not in belief)


def run_episode(policy, episode, strategy, seed=20261029, loss_inr=20000,
                time_weight=20, threshold=0.85):
    model = deepcopy(policy)
    model['true_cause'] = '__hidden_from_policy__'
    result = engine.run_diagnosis(model, strategy=strategy,
        true_outcome_fn=lambda test: episode['outcomes'][test['id']],
        confidence_threshold=threshold, prior=model['prior'], loss_inr=loss_inr,
        cost_kwargs={'time_weight_inr_per_min': time_weight},
        rng=random.Random(stable_seed(seed, episode['episode_id'], strategy)),
        inconclusive_outcome=INCONCLUSIVE)
    correct = result['final_diagnosis'] == episode['true_cause']
    confident = result['final_confidence'] >= threshold
    effective = result['total_cost_inr'] + time_weight*result['total_time_minutes']
    row = {k: episode[k] for k in ['episode_id', 'scenario_id', 'profile', 'true_cause', 'scenario_weight']}
    row.update(strategy=strategy, correct=int(correct), model_belief=result['final_confidence'],
               confident=int(confident), confident_error=int(confident and not correct),
               known_cause=int(episode['true_cause'] != UNKNOWN),
               brier=score_brier(result['final_belief'], episode['true_cause']),
               tests=result['num_tests_used'], direct_cost_inr=result['total_cost_inr'],
               time_minutes=result['total_time_minutes'], effective_test_cost_inr=effective,
               decision_loss_inr=loss_inr*(not correct), total_loss_inr=effective + loss_inr*(not correct),
               inconclusive_tests=sum(step['outcome'] == INCONCLUSIVE for step in result['steps']),
               stop_reason=result['stop_reason'], final_diagnosis=result['final_diagnosis'])
    return row, result
