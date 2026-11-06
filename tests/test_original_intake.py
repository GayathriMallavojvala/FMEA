import unittest
from unittest.mock import Mock
from pathlib import Path
import intake

class IntakeTests(unittest.TestCase):
    def test_common_misspellings_preserve_family_without_llm(self):
        for word in ['scratches','sctraches','scraches','scrathes','scracthes','scrathces','sratches','scartches']:
            with self.subTest(word=word):
                fn=Mock(side_effect=AssertionError('Unneeded model call'))
                self.assertEqual(intake.interpret('I found '+word+' on coil surfsce',fn)['family'],'Scratches')
    def test_crack_patterns_and_negation(self):
        fn=Mock(side_effect=AssertionError('Unneeded model call'))
        self.assertEqual(intake.interpret('while rollng the coil I got craks on surfsce',fn)['clarify'],'cracks')
        self.assertEqual(intake.interpret('Fine cracks connected like a web on the surface',fn)['family'],'Crazing')
        for text in ['No scratches','Inspect for cracks','Maybe crazing','scratches and pits']:
            self.assertIn('clarify',intake.interpret(text,fn))
    def test_semantic_path_requires_quote_and_family(self):
        text='The sheet has narrow channels dragged along its length'
        semantic=Mock(return_value={'family':'Scratches','evidence_quote':'narrow channels dragged along its length','question':''})
        self.assertEqual(intake.interpret(text,semantic)['family'],'Scratches')
        self.assertEqual(semantic.call_args.args[0],text)
        self.assertEqual(len(semantic.call_args.args[1]),10)
        for result in [{'family':'Unknown','evidence_quote':text,'question':''},{'family':'Scratches','evidence_quote':'invented quote','question':''},{'family':None,'evidence_quote':'','question':'Where is the mark?'}]:
            self.assertIn('clarify',intake.interpret(text,lambda *_:result))
    def test_original_stylesheet_and_entry_labels(self):
        html=(Path(__file__).resolve().parents[1]/'web/index.html').read_text()
        for label in ['Observed defect','Generate draft →','Try scratched-coil example']:
            self.assertIn(label,html)
        self.assertNotIn('Reported by',html)
        self.assertNotIn('id="family"',html)
        self.assertNotIn('CAPA Effectiveness Lab',html)
