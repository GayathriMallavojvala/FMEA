import json
import unittest
from copy import deepcopy
from pathlib import Path

from active_diagnosis import engine
from benchmark.simulation import (PROFILES, UNKNOWN, INCONCLUSIVE, models,
                                  sample_episode, validate_scenario)
from benchmark.evaluation import STRATEGIES, run_episode, score_brier
from benchmark.statistics import summarize, paired_comparisons

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = json.loads((ROOT/'active_diagnosis/diagnosis_library.json').read_text())['scenarios']


class SimulationTests(unittest.TestCase):
    def test_library_validation(self):
        for sc in SCENARIOS:
            validate_scenario(sc)
        bad = deepcopy(SCENARIOS[0]); bad['prior'][next(iter(bad['prior']))] = -1
        with self.assertRaises(ValueError):
            validate_scenario(bad)

    def test_models_independent_and_normalized(self):
        original = deepcopy(SCENARIOS[0])
        for profile in PROFILES:
            policy, env = models(original, profile)
            self.assertNotIn('true_cause', policy)
            self.assertAlmostEqual(sum(env['prior'].values()), 1)
            for test in env['likelihoods'].values():
                for distribution in test.values():
                    self.assertAlmostEqual(sum(distribution.values()), 1)
            self.assertEqual(original, SCENARIOS[0])
        policy['tests'][0]['cost_inr'] = 999999
        self.assertEqual(original, SCENARIOS[0])

    def test_all_causes_and_weights(self):
        for profile in PROFILES:
            _, env = models(SCENARIOS[0], profile)
            episodes = [sample_episode(env, c, r, 33, 3) for c in env['prior'] for r in range(3)]
            self.assertEqual({e['true_cause'] for e in episodes}, set(env['prior']))
            self.assertAlmostEqual(sum(e['scenario_weight'] for e in episodes), 1)
            self.assertEqual(len({e['episode_id'] for e in episodes}), len(episodes))

    def test_repeatability_and_strategy_pairing(self):
        policy, env = models(SCENARIOS[0], 'matched')
        cause = next(iter(env['prior']))
        episode = sample_episode(env, cause, 0, 33, 1)
        self.assertEqual(episode, sample_episode(env, cause, 0, 33, 1))
        before = deepcopy(episode)
        for strategy in STRATEGIES:
            row, trace = run_episode(policy, episode, strategy)
            self.assertEqual((row, trace), run_episode(policy, episode, strategy))
            for step in trace['steps']:
                self.assertEqual(step['outcome'], episode['outcomes'][step['test_id']])
            self.assertEqual(row['total_loss_inr'], row['effective_test_cost_inr'] + row['decision_loss_inr'])
        self.assertEqual(before, episode)

    def test_stochastic_marginals(self):
        _, env = models(SCENARIOS[0], 'matched')
        cause = next(iter(env['prior'])); tid = next(iter(env['likelihoods']))
        distribution = env['likelihoods'][tid][cause]
        counts = dict.fromkeys(distribution, 0)
        for r in range(4000):
            counts[sample_episode(env, cause, r, 12, 4000)['outcomes'][tid]] += 1
        for outcome, probability in distribution.items():
            self.assertAlmostEqual(counts[outcome]/4000, probability, delta=0.035)

    def test_shared_uniform_extreme(self):
        _, env = models(SCENARIOS[0], 'matched')
        cause = next(iter(env['prior']))
        env['likelihoods'] = {t: {cause: {'yes': 0.5, 'no': 0.5}} for t in ['t1', 't2']}
        env['settings']['shared_uniform_probability'] = 1.0
        for r in range(100):
            outcome = sample_episode(env, cause, r, 12, 100)['outcomes']
            self.assertEqual(outcome['t1'], outcome['t2'])

    def test_inconclusive_costs_without_update(self):
        policy, env = models(SCENARIOS[0], 'matched')
        cause = next(iter(env['prior']))
        env['settings']['missing_probability'] = 1
        episode = sample_episode(env, cause, 0, 12, 1)
        self.assertTrue(all(o == INCONCLUSIVE for o in episode['outcomes'].values()))
        row, trace = run_episode(policy, episode, 'fixed_order', threshold=1)
        self.assertEqual(row['tests'], len(policy['tests']))
        self.assertEqual(trace['final_belief'], engine.normalize(policy['prior']))
        self.assertEqual(row['direct_cost_inr'], sum(t['cost_inr'] for t in policy['tests']))
        self.assertEqual(row['inconclusive_tests'], row['tests'])

    def test_unknown_truth_not_in_policy(self):
        policy, env = models(SCENARIOS[0], 'unknown_cause')
        episode = sample_episode(env, UNKNOWN, 0, 33, 1)
        row, trace = run_episode(policy, episode, 'cost_aware_voi')
        self.assertEqual(row['correct'], 0)
        self.assertEqual(row['known_cause'], 0)
        self.assertNotIn(UNKNOWN, trace['final_belief'])
        self.assertGreaterEqual(row['brier'], 1)

    def test_truth_does_not_change_decisions(self):
        policy, env = models(SCENARIOS[0], 'matched')
        episode = sample_episode(env, next(iter(env['prior'])), 0, 33, 1)
        changed = deepcopy(episode); changed['true_cause'] = UNKNOWN
        for strategy in STRATEGIES:
            _, first = run_episode(policy, episode, strategy)
            _, second = run_episode(policy, changed, strategy)
            self.assertEqual(first, second)

    def test_prior_only_and_brier(self):
        self.assertEqual(score_brier({'a': 1, 'b': 0}, 'a'), 0)
        self.assertEqual(score_brier({'a': 1, 'b': 0}, 'b'), 2)
        self.assertEqual(score_brier({'a': 0.5, 'b': 0.5}, 'a'), 0.5)
        self.assertEqual(score_brier({'a': 0.5, 'b': 0.5}, UNKNOWN), 1.5)
        policy, env = models(SCENARIOS[0], 'matched')
        ep = sample_episode(env, next(iter(env['prior'])), 0, 33, 1)
        row, trace = run_episode(policy, ep, 'no_test')
        self.assertEqual(row['tests'], 0)
        self.assertEqual(row['effective_test_cost_inr'], 0)
        self.assertEqual(trace['stop_reason'], 'no_test')

    def test_weighted_statistics_and_paired_interval(self):
        rows = []
        for sc in SCENARIOS[:2]:
            policy, env = models(sc, 'matched')
            for cause in env['prior']:
                ep = sample_episode(env, cause, 0, 33, 1)
                row, _ = run_episode(policy, ep, 'no_test')
                rows += [{**row, 'strategy': strategy} for strategy in ['cost_aware_voi', 'cost_aware']]
        summaries, calibration = summarize(rows, 33, 50)
        expected = sum(max(sc['prior'].values()) for sc in SCENARIOS[:2])/2
        self.assertAlmostEqual(summaries[0]['correct'], expected)
        comparisons = paired_comparisons(rows, 33, 50)
        self.assertTrue(all(r['difference'] == r['ci_low'] == r['ci_high'] == 0 for r in comparisons))
        self.assertAlmostEqual(sum(r['weight_mass'] for r in calibration if r['strategy'] == 'cost_aware'), 1)
        with self.assertRaises(ValueError):
            paired_comparisons(rows+[rows[0]], 33, 50)

if __name__ == '__main__':
    unittest.main()
