import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import app
import store
from workflow import session as wf

class RestorationTests(unittest.TestCase):
    def test_multiple_inspections_saved_and_threshold_does_not_hide_checks(self):
        sc = next(iter(app.DIAG_SCENARIOS.values()))
        with tempfile.TemporaryDirectory() as folder, patch.object(store, 'DB_PATH', str(Path(folder)/'cases.db')):
            state, *_ = app.wf_start(sc['presenting_symptom'], 20000, 'reviewer')
            for test in sc['tests'][:2]:
                state, *_ = app.wf_record(test['id'], test['outcomes'][0], 20000, 'reviewer', state)
            self.assertEqual(len(state['history']), 2)
            con = store.connect()
            self.assertEqual(len(store.get_case(con, state['case_id'])['state']['history']), 2)
            con.close()
            cause = next(iter(state['belief']))
            state['belief'] = {c: float(c == cause) for c in state['belief']}
            _, test_update, _, button_update = app._panel(state)
            self.assertTrue(test_update['visible'])
            self.assertTrue(button_update['visible'])
            before = list(state['history'])
            test = sc['tests'][0]
            updated, message, *_ = app.wf_record(test['id'], test['outcomes'][0], 20000, 'reviewer', state)
            self.assertEqual(updated['history'], before)
            self.assertIn('already recorded', message)
