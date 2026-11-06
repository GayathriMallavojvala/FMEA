import hashlib
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from evidence.registry import build_registry, validate_registry, eligibility, canonical_url, field
from evidence.splits import leakage_groups, make_split, validate_split, training_passages
from evidence.retrieval import EvidenceRetriever, format_context, citation_audit, benchmark_retriever, retrieve_incident

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    sources=[dict(source_id=f'S{i}',title=f'Source {i}',url=f'https://example.org/{i}',verification='primary_page_accessed') for i in range(5)]
    texts=['Heldoutsecret failure cause answer.', 'Steel cracking due to documented thermal stress.',
           'Validationsecret observation.', 'Steel cracking due to documented thermal stress.',
           'Thermal fatigue steel rolling background.']
    passages=[dict(passage_id=f'P{i}',source_id=f'S{i}',text=text,locator='Section 1',kind='reference',domain='steel',
        incident_ids=['I1'] if i==0 else [],review_state='source_checked',representation='test_fixture',
        text_sha256=hashlib.sha256(text.encode()).hexdigest()) for i,text in enumerate(texts)]
    incident=dict(incident_id='I1',source_ids=['S0'],origin='published_incident',duplicate_group=None,domain='steel',
        initial_information=field('Steel cracking and thermal stress','observed',['S0'],'initial observations'),
        established_cause=field('Heldoutsecret','observed',['S0'],'conclusion'),
        acceptable_actions=field(),costs=field(),
        tests=[dict(test_id='T1',label='Inspect',result=field('positive','observed',['S0'],'table 1'),cost=field())],
        domain_review=dict(status='approved',reviewer='fixture reviewer',date='2026-10-29'))
    registry=dict(schema_version=1,sources=sources,passages=passages,incidents=[incident])
    validate_registry(registry)
    split=make_split(registry)
    for pid,record in split['passages'].items():
        partition='test' if pid=='P0' else 'validation' if pid=='P2' else 'train'
        record['partition']=partition
        split['group_partitions'][record['group_id']]=partition
    split['incidents']['I1']['partition']='test'
    validate_split(registry,split)
    return registry,split


class EvidenceTests(unittest.TestCase):
    def test_legacy_audit_no_promotions(self):
        r=build_registry(ROOT)
        self.assertEqual(len(r['passages']),125)
        self.assertEqual(len(r['incidents']),94)
        self.assertTrue(all(not eligibility(i)['eligible'] for i in r['incidents']))
        self.assertEqual(len(EvidenceRetriever(r).passages),1)
        self.assertEqual(sum(p['review_state']=='unverified' for p in r['passages']),124)

    def test_canonical_aliases(self):
        self.assertEqual(canonical_url('http://www.ncbi.nlm.nih.gov/pmc/articles/PMC8588222/?utm_source=x#foo'),
                         canonical_url('https://pmc.ncbi.nlm.nih.gov/articles/PMC8588222/'))

    def test_hash_and_relationship_validation(self):
        r,_=fixture()
        r['passages'][0]['text']='silently changed'
        with self.assertRaises(ValueError):validate_registry(r)
        r,_=fixture();r['incidents'][0]['initial_information']['source_ids']=[]
        with self.assertRaises(ValueError):validate_registry(r)

    def test_source_incident_and_duplicate_groups(self):
        r,s=fixture();groups=leakage_groups(r)
        self.assertEqual(groups['i:I1'],groups['p:P0'])
        self.assertEqual(groups['p:P1'],groups['p:P3'])
        r['incidents'].append({**deepcopy(r['incidents'][0]),'incident_id':'I2','source_ids':['S4'],'duplicate_group':'same'})
        r['incidents'][0]['duplicate_group']='same'
        groups=leakage_groups(r)
        self.assertEqual(groups['i:I1'],groups['i:I2'])

    def test_split_tampering_rejected(self):
        r,s=fixture();s['passages']['P3']['partition']='test'
        with self.assertRaises(ValueError):validate_split(r,s)
        r,s=fixture();s['passages'].pop('P4')
        with self.assertRaises(ValueError):validate_split(r,s)

    def test_train_only_vocabulary_and_self_exclusion(self):
        r,s=fixture();retriever=benchmark_retriever(r,s,'I1')
        self.assertNotIn('heldoutsecret',retriever.vectorizer.vocabulary_)
        self.assertNotIn('validationsecret',retriever.vectorizer.vocabulary_)
        self.assertNotIn('P0',[p['passage_id'] for p in retriever.passages])
        # Even a training incident cannot retrieve its own source group.
        s['group_partitions'][s['incidents']['I1']['group_id']]='train'
        s['incidents']['I1']['partition']=s['passages']['P0']['partition']='train'
        self.assertNotIn('P0',[p['passage_id'] for p in training_passages(r,s,'I1')])
        with self.assertRaises(ValueError):training_passages(r,s,'missing')

    def test_initial_information_query_only(self):
        r,s=fixture()
        with patch.object(EvidenceRetriever,'search',return_value=[]) as search:
            retrieve_incident(r,s,'I1')
        self.assertEqual(search.call_args.args[0],r['incidents'][0]['initial_information']['value'])
        self.assertNotIn('Heldoutsecret',search.call_args.args[0])

    def test_unverified_and_cross_domain_excluded(self):
        r,s=fixture()
        r['passages'][1]['review_state']='unverified'
        r['passages'][3]['domain']='pharmaceutical'
        ids={p['passage_id'] for p in EvidenceRetriever(r).passages}
        self.assertNotIn('P1',ids);self.assertNotIn('P3',ids)
        r['incidents'][0]['domain_review']['status']='not_reviewed'
        with self.assertRaises(ValueError):benchmark_retriever(r,s,'I1')

    def test_relevance_abstention_and_reference_metadata(self):
        r,s=fixture();retriever=benchmark_retriever(r,s,'I1')
        self.assertEqual(retriever.search('xyzzy qqqq'),[])
        hits=retriever.search('thermal steel cracking')
        self.assertTrue(hits)
        context=format_context(hits)
        self.assertIn('[REF:',context);self.assertIn('https://example.org/',context)
        self.assertIn('Section 1',context)
        self.assertIn('not evidence',context)
        self.assertIn('No eligible reference',format_context([]))

    def test_citation_id_validity_not_entailment(self):
        r,s=fixture();hits=benchmark_retriever(r,s,'I1').search('steel cracking')
        valid=hits[0]['passage_id']
        check=citation_audit(f'Claim [REF:{valid}] [E1] [CASE] [REF:FAKE] [E9]',hits,['E1'])
        self.assertEqual(check['invalid_ids'],['E9','REF:FAKE'])
        self.assertIn('ID validity only',check['scope'])

    def test_legacy_capa_whole_source_stays_together(self):
        r=build_registry(ROOT);s=make_split(r)
        self.assertEqual(len({rec['group_id'] for rec in s['incidents'].values()}),1)
        self.assertEqual(make_split(r),s)
        for incident in r['incidents']:
            self.assertTrue(all(p['kind']!='legacy_example' for p in training_passages(r,s,incident['incident_id'])))


if __name__=='__main__':unittest.main()
