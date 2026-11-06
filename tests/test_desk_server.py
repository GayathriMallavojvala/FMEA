import json
import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import desk_server as ds
import store


def fake_draft(state,sc,hits):
    return {'issue':'Reported surface defect; review required.','investigation':'Candidates remain unconfirmed.',
      'fmea':[dict(cause_id=c['id'],effect='Assess applicable specification',prevention='Pending engineer review',detection='Pending engineer review') for c in sc['candidate_causes']],
      'actions':[dict(type=t,what='Proposed '+t+' action',where='Pending engineer review',when='Pending approval',who='Quality engineer (suggested)',required_results='Record evidence for review') for t in ['containment','corrective','preventive','verification']],
      'verification':'Approve acceptance limits and monitoring window.','unresolved':['Cause requires verification'],'reference_ids':[]},'test-provider-model'

class DeskTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.patch=patch.object(store,'DB_PATH',self.folder.name+'/cases.db');self.patch.start()
        self.client=TestClient(ds.app)
    def tearDown(self):self.patch.stop();self.folder.cleanup()
    def start(self):
        r=self.client.post('/api/cases',json={'description':'Longitudinal scratches on a cold rolled coil.','family':'Scratches'})
        self.assertEqual(r.status_code,200,r.text)
        d=r.json();return d['case'],{'X-Case-Token':d['token']}
    def test_simple_page_and_crack_clarification(self):
        r=self.client.get('/');self.assertEqual(r.status_code,200)
        self.assertNotIn('CAPA Effectiveness Lab',r.text)
        r=self.client.post('/api/cases',json={'description':'While rolling coil I got cracks on surfsce'})
        self.assertEqual(r.json()['clarify'],'cracks')
    def test_sequential_evidence_reports_cache_and_persistence(self):
        case,headers=self.start();cid=case['id']
        with patch.object(ds,'request_draft',side_effect=fake_draft) as gen:
            draft=self.client.post(f'/api/cases/{cid}/generate',json={'revision':0},headers=headers)
            self.assertEqual(draft.status_code,200,draft.text)
            self.assertEqual(len(draft.json()['draft']['fmea']),len(case['ranking']))
            self.client.post(f'/api/cases/{cid}/generate',json={'revision':0},headers=headers)
            self.assertEqual(gen.call_count,1)
            for i in range(2):
                test=case['tests'][0]
                r=self.client.post(f'/api/cases/{cid}/findings',json={'revision':i,'test_id':test['id'],'outcome':test['outcomes'][0],'source':f'Inspection log {i+1}'},headers=headers)
                self.assertEqual(r.status_code,200,r.text);case=r.json()
                r=self.client.post(f'/api/cases/{cid}/generate',json={'revision':i+1},headers=headers)
                self.assertEqual(r.status_code,200,r.text)
            saved=self.client.get(f'/api/cases/{cid}',headers=headers).json()
            self.assertEqual(len(saved['history']),2)
            self.assertEqual(saved['draft_revision'],2)
            self.assertAlmostEqual(sum(r['current'] for r in saved['ranking']),1)
            self.assertEqual(gen.call_count,3)
    def test_invalid_and_stale_findings_do_not_change_case(self):
        case,h=self.start();cid=case['id'];test=case['tests'][0]
        for rev in [99,0]:
            r=self.client.post(f'/api/cases/{cid}/findings',json={'revision':rev,'test_id':test['id'],'outcome':'invalid','source':'Record'},headers=h)
            self.assertIn(r.status_code,[409,422])
        self.assertEqual(len(self.client.get(f'/api/cases/{cid}',headers=h).json()['history']),0)
        self.assertEqual(self.client.get(f'/api/cases/{cid}').status_code,404)
    def test_missing_key_clear_error_no_fabricated_report(self):
        case,h=self.start()
        with patch.dict(os.environ,{},clear=True):
            r=self.client.post(f'/api/cases/{case["id"]}/generate',json={'revision':0},headers=h)
        self.assertEqual(r.status_code,503)
        self.assertIn('configured',r.text)
        self.assertIsNone(self.client.get(f'/api/cases/{case["id"]}',headers=h).json()['draft'])
    def test_provider_404_and_malformed_response(self):
        case,h=self.start();sc=next(s for s in ds.SCENARIOS.values() if s['defect_type']=='Scratches')
        state=ds.wf.new_case(sc);state['defect_text']='scratches'
        with patch.dict(os.environ,{'GROQ_API_KEY':'test-key'}),patch.object(ds.httpx.Client,'post',return_value=ds.httpx.Response(404)):
            with self.assertRaises(ds.HTTPException) as e:ds.request_draft(state,sc,[])
            self.assertIn('LLM_MODEL',e.exception.detail)
        with patch.dict(os.environ,{'GROQ_API_KEY':'test-key'}),patch.object(ds.httpx.Client,'post',return_value=ds.httpx.Response(200,json={'choices':[{'message':{'content':'{}'}}]})):
            with self.assertRaises(ds.HTTPException) as e:ds.request_draft(state,sc,[])
            self.assertIn('not published',e.exception.detail)
