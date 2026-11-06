"""Import legacy content without laundering uncertain provenance into facts."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


FIELD_STATUSES = {'observed', 'expert_estimated', 'simulated', 'unverified', 'missing'}


def canonical_url(url):
    parts = urlsplit(url.strip())
    host = (parts.hostname or '').lower()
    if parts.scheme not in {'http', 'https'} or not host:
        raise ValueError('Source URL must be http(s)')
    path = parts.path.rstrip('/')
    if host in {'www.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov'} and path.startswith('/pmc/articles/'):
        host, path = 'pmc.ncbi.nlm.nih.gov', path[len('/pmc'):]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith('utm_')))
    return urlunsplit(('https', host, path, query, ''))


def source_id(url):
    return 'SRC-'+hashlib.sha256(canonical_url(url).encode()).hexdigest()[:16]


def field(value=None, status='missing', source_ids=None, locator=None):
    return dict(value=value, status=status, source_ids=source_ids or [], locator=locator)


def build_registry(root):
    root = Path(root)
    kb = root/'knowledge_base'
    load = lambda name: json.loads((kb/name).read_text())
    review_data = json.loads((root/'evidence/source_reviews.json').read_text())
    reviews = {canonical_url(r['source_url']): r for r in review_data['reviews']}
    sources, passages, incidents = {}, [], []

    def local_source(name, title):
        sid = 'LOCAL-'+name.removesuffix('.json').upper().replace('_','-')
        sources[sid] = dict(source_id=sid, title=title, url=None,
                            local_path='knowledge_base/'+name, verification='unverified',
                            content_sha256=hashlib.sha256((kb/name).read_bytes()).hexdigest())
        return sid

    def external_source(url, title):
        sid = source_id(url)
        if sid not in sources:
            review = reviews.get(canonical_url(url), {})
            sources[sid] = dict(source_id=sid, title=title, url=canonical_url(url),
                verification=review.get('status', 'not_checked'), review=review)
        return sid

    def add(pid, source, text, locator, kind, domain='unknown', incident_ids=None,
            review_state='unverified', representation='legacy_paraphrase'):
        passages.append(dict(passage_id=pid, source_id=source, text=text, locator=locator,
            kind=kind, domain=domain, incident_ids=incident_ids or [],
            review_state=review_state, representation=representation,
            text_sha256=hashlib.sha256(text.encode()).hexdigest()))

    sid = local_source('capa_dataset.json', 'Legacy CAPA examples: origin not established')
    for index, rec in enumerate(load('capa_dataset.json')):
        iid = 'INC-'+rec['id']
        locator = f"$[{index}] (id={rec['id']})"
        text = (f"Defect type: {rec['defect_type']}. Process stage: {rec['process_stage']}. "
                f"{rec['non_conformance_description']} Root cause: {rec['root_cause']} "
                f"Corrective action: {rec['corrective_action']} Preventive action: {rec['preventive_action']}")
        add('CAPA-'+rec['id'], sid, text, locator, 'legacy_example', 'steel', [iid])
        incidents.append(dict(incident_id=iid, source_ids=[sid], origin='unverified_legacy',
            duplicate_group='legacy-capa-origin-unresolved', domain='steel',
            initial_information=field(rec['non_conformance_description'], 'unverified', [sid], locator+'.non_conformance_description'),
            tests=[], established_cause=field(rec['root_cause'], 'unverified', [sid], locator+'.root_cause'),
            acceptable_actions=field([rec['corrective_action'],rec['preventive_action']], 'unverified', [sid], locator),
            costs=field(), domain_review=dict(status='not_reviewed', reviewer=None, date=None)))
    for index, rec in enumerate(load('real_document_chunks.json')):
        sid = external_source(rec['source_url'], rec['source_name'])
        domain = 'pharmaceutical' if rec['source_type']=='FDA_483' else ('aerospace' if rec['source_type']=='NASA_NTRS' else 'metallurgy_or_steel_unverified')
        add(rec['chunk_id'], sid, rec['paraphrased_summary'],
            f"knowledge_base/real_document_chunks.json $[{index}] (original passage location missing)",
            'reference', domain)
    sid = local_source('defect_taxonomy.json', 'Legacy defect taxonomy: mechanisms unverified')
    for index, rec in enumerate(load('defect_taxonomy.json')['defect_families']):
        add(f'TAX-{index+1:02}',sid,f"{rec['defect_type']}. {rec['mechanism_description']}",
            f'$.defect_families[{index}]', 'taxonomy', 'steel')
    sid = local_source('standards_reference.json', 'Legacy standards summaries: not authoritative standard text')
    for index, rec in enumerate(load('standards_reference.json')['clauses']):
        add(f'STD-{index+1:02}',sid,f"{rec['standard']} clause {rec['clause']} {rec['title']}. {rec['summary']}",
            f'$.clauses[{index}]', 'standard_summary', 'quality_management')
    for passage in review_data['reviewed_passages']:
        sid = external_source(passage['source_url'], passage['title'])
        add(passage['passage_id'],sid,passage['text'],passage['locator'], 'reference',
            passage['domain'],review_state=passage['review_state'],representation=passage['representation'])
    registry = dict(schema_version=1, sources=list(sources.values()), passages=passages, incidents=incidents,
        notes=['Source access does not verify every derived summary.',
               'No legacy record is automatically promoted to observed plant evidence.'])
    validate_registry(registry)
    return registry


def validate_registry(registry):
    def unique(records, key):
        keys=[r[key] for r in records]
        if len(keys)!=len(set(keys)):
            raise ValueError(f'Duplicate {key}')
        return set(keys)
    sources=unique(registry['sources'],'source_id')
    passages=unique(registry['passages'],'passage_id')
    incidents=unique(registry['incidents'],'incident_id')
    for p in registry['passages']:
        if p['source_id'] not in sources or not set(p['incident_ids'])<=incidents:
            raise ValueError('Unresolved passage relationship')
        if not p['locator'] or not p['text']:
            raise ValueError('Passage must have text and local/source locator')
        if hashlib.sha256(p['text'].encode()).hexdigest()!=p['text_sha256']:
            raise ValueError('Passage content hash mismatch')
    for inc in registry['incidents']:
        if not inc['source_ids'] or not set(inc['source_ids'])<=sources:
            raise ValueError('Unresolved incident source')
        for f in [inc['initial_information'],inc['established_cause'],inc['acceptable_actions'],inc['costs']]+[v for t in inc['tests'] for v in [t['result'],t['cost']]]:
            if f['status'] not in FIELD_STATUSES or not set(f['source_ids'])<=sources:
                raise ValueError('Invalid field provenance')
            if f['status']=='missing' and f['value'] is not None:
                raise ValueError('Missing fields must be null')
            if f['status']=='observed' and (f['value'] is None or not f['source_ids'] or not f['locator']):
                raise ValueError('Observed fields require a source and location')
    return True


def eligibility(incident):
    """Conservative gate for incident-level diagnosis evaluation, not human authentication."""
    reasons=[]
    if incident['origin'] not in {'published_incident','plant_record'}:
        reasons.append('incident origin is not established')
    for name in ['initial_information','established_cause']:
        if incident[name]['status']!='observed':
            reasons.append(name+' lacks observed evidence')
    if not incident['tests']:
        reasons.append('no documented requestable test outcomes')
    elif any(t['result']['status']!='observed' for t in incident['tests']):
        reasons.append('some test outcomes are not observed')
    review=incident['domain_review']
    if review['status']!='approved' or not review.get('reviewer') or not review.get('date'):
        reasons.append('qualified review is missing')
    return dict(eligible=not reasons, reasons=reasons)


def load_registry(root, path=None):
    if path:
        registry=json.loads(Path(path).read_text())
        validate_registry(registry)
        return registry
    return build_registry(root)
