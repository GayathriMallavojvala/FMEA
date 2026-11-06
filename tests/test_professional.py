import json
import unittest
from pathlib import Path
from unittest.mock import patch
import app
import store
import professional_report as pr
from workflow import session as wf

class ProfessionalTests(unittest.TestCase):
    def test_escaped_report_and_professional_tables(self):
        sc=next(iter(app.DIAG_SCENARIOS.values()))
        state=wf.new_case(sc)
        state['defect_text']='<script>alert(1)</script>'
        report='6. Action Plan\n| What | Where | When | Who | Required Results |\n|---|---|---|---|---|\n| Check | Line | Pending approval | Engineer | Record finding |'
        view=pr.render(report,state,sc)
        self.assertNotIn('<script>',view)
        for heading in ['Process FMEA','Recorded inspection','Diagnostic ranking','Required Results','Review, effectiveness']:
            self.assertIn(heading,view)
        actions=store.parse_action_table(report)
        self.assertEqual(len(actions),1)
        self.assertEqual(actions[0]['owner'],'Engineer')
        self.assertEqual(actions[0]['due_date'],'')

    def test_fast_verification_never_calls_llm(self):
        with patch.dict('os.environ',{'CAPA_DEEP_VERIFY':'0'}), patch.object(app,'get_groq_client',side_effect=AssertionError('Extra LLM call')):
            result=app._faithfulness_block('A scratch was reported on the steel coil.','A scratch was reported on the steel coil.')
            self.assertIn('not factual validation',result)

    def test_preview_error_has_no_download(self):
        with patch.object(app,'wf_generate',return_value=('Error: provider unavailable','')):
            result=app.wf_generate_professional('scratch','reviewer',{})
            self.assertIsNone(result[-1])
