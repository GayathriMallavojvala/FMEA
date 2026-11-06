import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from active_diagnosis import engine
from workflow import session as wf
import store

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = json.loads((ROOT/'active_diagnosis/diagnosis_library.json').read_text())['scenarios']


class BaselineTests(unittest.TestCase):
    def test_guided_and_runner_match_every_scenario(self):
        for sc in SCENARIOS:
            with self.subTest(sc=sc['id']):
                state = wf.new_case(sc)
                while wf.stage(state, sc) == 'continue':
                    test = wf.recommend(state, sc)[0]['test']
                    state = wf.record_result(state, sc, test['id'], engine.deterministic_outcome(sc, test))
                result = engine.run_diagnosis(sc, 'cost_aware_voi', loss_inr=wf.DEFAULT_LOSS_INR)
                self.assertEqual([h['test_id'] for h in state['history']], result['tests_used'])
                self.assertEqual(wf.top_cause(state)[0], result['final_diagnosis'])
                self.assertAlmostEqual(wf.top_cause(state)[1], result['final_confidence'])

    def test_state_preserved_and_not_mutated(self):
        sc = SCENARIOS[0]
        state = wf.new_case(sc, 37000)
        state.update(case_id=9, extra={'nested': [1]})
        t = sc['tests'][0]
        updated = wf.record_result(state, sc, t['id'], t['outcomes'][0])
        self.assertEqual(updated['loss_inr'], 37000)
        self.assertEqual(updated['case_id'], 9)
        updated['extra']['nested'].append(2)
        self.assertEqual(state['extra'], {'nested': [1]})
        self.assertEqual(state['history'], [])
        with self.assertRaises(ValueError):
            wf.record_result(updated, sc, t['id'], t['outcomes'][0])

    def test_probability_distributions(self):
        for sc in SCENARIOS:
            self.assertAlmostEqual(sum(wf.new_case(sc)['belief'].values()), 1)
            for t in sc['tests']:
                for probs in t['outcome_likelihoods'].values():
                    self.assertEqual(set(probs), set(t['outcomes']))
                    self.assertAlmostEqual(sum(probs.values()), 1)
                    self.assertTrue(all(0 <= p <= 1 for p in probs.values()))

    def test_high_belief_not_confirmation(self):
        sc = SCENARIOS[0]; state = wf.new_case(sc)
        first = next(iter(state['belief']))
        state['belief'] = {c: float(c == first) for c in state['belief']}
        prompt = wf.build_capa_prompt('defect', state, sc, 'context')
        self.assertNotIn('CONFIRMED', prompt)
        self.assertNotIn('leads to exactly', prompt)
        self.assertIn('not independently verified', prompt)
        self.assertNotIn('Root cause established', ' '.join(wf.next_steps(state, sc)))

    def test_action_regeneration_and_reopening(self):
        con = store.connect(':memory:')
        cid = store.create_case(con, 'defect', wf.new_case(SCENARIOS[0]))
        self.assertEqual(store.add_actions(con, cid, [{'description': 'Inspect'}]), 1)
        action = store.list_actions(con, cid)[0]
        store.update_action(con, action['id'], status='Done')
        self.assertEqual(store.add_actions(con, cid, [{'description': 'Reworded inspection'}]), 0)
        store.save_capa(con, cid, 'new report')
        self.assertEqual(store.get_case(con, cid)['status'], 'Actions complete - verify effectiveness')
        self.assertEqual(len(store.list_actions(con, cid)), 1)
        store.update_action(con, action['id'], status='Open')
        self.assertEqual(store.get_case(con, cid)['status'], 'Actions in progress')
        con.close()

    def test_invalid_loss(self):
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                engine.rank_tests_voi({'a': 1}, [], value)


class UIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.app = app

    def test_unknown_match_abstains_without_saving(self):
        with patch.object(store, 'create_case', side_effect=AssertionError('must not save')):
            result = self.app.wf_start('xyzzy qwerty', 20000, 'tester')
        self.assertIsNone(result[0])
        self.assertIn('No diagnosis started', result[1])

    def test_loss_change_recomputes_and_persists(self):
        with tempfile.TemporaryDirectory() as d, patch.object(store, 'DB_PATH', str(Path(d)/'test.db')):
            sc = SCENARIOS[0]
            state, *_ = self.app.wf_start(sc['presenting_symptom'], 20000, 'tester')
            self.assertIsNotNone(state)
            updated, md, *_ = self.app.wf_change_loss(90000, 'tester', state)
            self.assertEqual(updated['loss_inr'], 90000)
            con = store.connect()
            self.assertEqual(store.get_case(con, state['case_id'])['state']['loss_inr'], 90000)
            self.assertEqual(store.events(con, state['case_id'])[-1]['event'], 'loss_changed')
            con.close()

    def test_generation_uses_case_evidence_and_keeps_plan(self):
        from types import SimpleNamespace
        report = '| Action | Owner (role) | Due (days from today) | Done when |\n|---|---|---|---|\n| Inspect | Engineer | 2 | Reviewed |'
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=report))]))))
        with tempfile.TemporaryDirectory() as d, patch.object(store, 'DB_PATH', str(Path(d)/'test.db')), patch.object(self.app, 'get_groq_client', return_value=client), patch.object(self.app, '_faithfulness_block', return_value='checked') as verifier:
            sc = SCENARIOS[0]
            state, *_ = self.app.wf_start(sc['presenting_symptom'], 20000, 'tester')
            test = sc['tests'][0]
            state = wf.record_result(state, sc, test['id'], test['outcomes'][0])
            for _ in range(2):
                result, _ = self.app.wf_generate('edited unrelated description', 'tester', state)
                self.assertNotIn('Error:', result)
            evidence = verifier.call_args.args[1]
            self.assertIn(sc['presenting_symptom'], evidence)
            self.assertIn(test['label'], evidence)
            self.assertNotIn('edited unrelated description', evidence)
            con = store.connect()
            self.assertEqual(len(store.list_actions(con, state['case_id'])), 1)
            con.close()

    def test_comparison_contains_guided_baseline(self):
        output = self.app.run_diagnosis_comparison(SCENARIOS[0]['id'])
        self.assertIn('Monetary VOI', output)
        self.assertIn('Cheapest first', output)
        self.assertIn('Deterministic demonstration', output)

if __name__ == '__main__':
    unittest.main()
