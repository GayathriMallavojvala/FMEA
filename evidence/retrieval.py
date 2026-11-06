"""Source-aware retrieval with strict default eligibility and training-only fitting."""
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from evidence.registry import validate_registry, eligibility
from evidence.splits import training_passages


class EvidenceRetriever:
    def __init__(self, registry, passages=None, allow_unverified=False):
        validate_registry(registry)
        self.sources={s['source_id']:s for s in registry['sources']}
        candidates=registry['passages'] if passages is None else passages
        self.passages=[p for p in candidates
            if p['domain']=='steel' and p['kind']=='reference'
            and (p['review_state']=='source_checked' or allow_unverified)]
        # Unverified mixed-domain material is excluded even when requested.
        self.vectorizer=None
        self.matrix=None
        if self.passages:
            self.vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
            try:
                self.matrix=self.vectorizer.fit_transform([p['text'] for p in self.passages])
            except ValueError as e:
                if 'empty vocabulary' not in str(e):
                    raise
                self.passages=[]
                self.vectorizer=None

    def search(self, queries, k=5, minimum_similarity=0.10):
        if not 0<=minimum_similarity<=1 or not math.isfinite(minimum_similarity):
            raise ValueError('Invalid minimum similarity')
        if k<1:
            raise ValueError('k must be positive')
        if isinstance(queries,str):
            queries=[queries]
        if self.matrix is None or not queries:
            return []
        sims=cosine_similarity(self.vectorizer.transform(queries),self.matrix)
        scores=sims.max(axis=0)
        ranked=sorted(range(len(scores)),key=lambda i:(-scores[i],self.passages[i]['passage_id']))
        return [{**self.passages[i], 'score':float(scores[i]),
                 'source':self.sources[self.passages[i]['source_id']]}
                for i in ranked if scores[i]>0 and scores[i]>=minimum_similarity][:k]


def benchmark_retriever(registry, split, incident_id):
    incident=next((i for i in registry['incidents'] if i['incident_id']==incident_id),None)
    if incident is None:
        raise ValueError('Unknown incident')
    gate=eligibility(incident)
    if not gate['eligible']:
        raise ValueError('Incident not eligible: '+'; '.join(gate['reasons']))
    return EvidenceRetriever(registry,training_passages(registry,split,incident_id))


def format_context(hits):
    if not hits:
        return 'No eligible reference passage met the relevance threshold. Do not invent reference support.'
    blocks=[]
    for hit in hits:
        source=hit['source']
        blocks.append(f"[REF:{hit['passage_id']}]\nSource: {source['title']}\n"
            f"URL: {source.get('url') or 'No external source established'}\n"
            f"Location: {hit['locator']}\nProvenance: {hit['representation']}; {hit['review_state']}\n"
            f"Scope: background reference, not evidence that this case has this cause.\n{hit['text']}")
    return '\n\n---\n\n'.join(blocks)


def citation_audit(report, hits, evidence_ids=()):
    """Reference-ID validity only; no semantic entailment or contradiction claim."""
    cited=set(re.findall(r'\[(REF:[A-Za-z0-9_.:-]+|E\d+|CASE)\]',report))
    allowed={f"REF:{h['passage_id']}" for h in hits}|set(evidence_ids)|{'CASE'}
    invalid=sorted(cited-allowed)
    refs=sorted(c for c in cited if c.startswith('REF:'))
    return dict(cited_ids=sorted(cited),invalid_ids=invalid,reference_citation_count=len(refs),
                scope='ID validity only; citation correctness and factual support require review')


def retrieve_incident(registry, split, incident_id, k=5):
    """Benchmark query contains initial information only, never final-cause labels."""
    retriever=benchmark_retriever(registry,split,incident_id)
    incident=next(i for i in registry['incidents'] if i['incident_id']==incident_id)
    return retriever.search(incident['initial_information']['value'],k=k)
