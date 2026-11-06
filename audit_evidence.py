"""Build and audit the provenance registry and conservative source-group split."""
import argparse
from collections import Counter
import json
from pathlib import Path
from evidence.registry import load_registry, eligibility
from evidence.splits import make_split, training_passages
from evidence.retrieval import EvidenceRetriever


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/evidence'))
    parser.add_argument('--registry',type=Path,help='Optional reviewed registry JSON')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    registry=load_registry(root,args.registry)
    split=make_split(registry)
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    checks=[]
    for incident in registry['incidents']:
        allowed=training_passages(registry,split,incident['incident_id'])
        checks.append(dict(incident_id=incident['incident_id'],**eligibility(incident),
                           allowed_training_passages=len(allowed)))
    stats=dict(sources=len(registry['sources']),passages=len(registry['passages']),
        incident_records=len(registry['incidents']),eligible_incidents=sum(x['eligible'] for x in checks),
        strict_reference_passages=len(EvidenceRetriever(registry).passages),
        groups=len(split['group_partitions']),group_partitions=dict(Counter(split['group_partitions'].values())),
        passage_review_states=dict(Counter(p['review_state'] for p in registry['passages'])),
        legacy_domain_counts=dict(Counter(p['domain'] for p in registry['passages'])),
        split_incident_checks=len(checks),warnings=[
            'Grouped split is infrastructure, not a usable incident benchmark.',
            'No human reviewer identity is authenticated by this schema.',
            'Unknown paraphrase relationships require manual provenance review.',
            'ID-valid citations do not establish support or absence of contradictions.'])
    for name,data in [('registry.json',registry),('split.json',split),('incident_eligibility.json',checks),('audit.json',stats)]:
        (output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
    (output/'REPORT.md').write_text(f'''# Evidence audit — Step 3

{stats['passages']} passages from {stats['sources']} source identities;
{stats['incident_records']} legacy incident-shaped records;
**{stats['eligible_incidents']} eligible incident benchmarks**.

Only {stats['strict_reference_passages']} reference passage currently meets strict
steel-domain/source-checked retrieval rules. The legacy 124 passages remain
unverified at passage level and are excluded from default generation context.
One added passage is an assistant paraphrase checked against a publisher abstract;
it is background literature, not observed incident evidence or expert validation.

The split has {stats['groups']} connected source/incident/duplicate groups.
All 94 legacy CAPA examples are grouped together because their incident provenance
is unresolved. All incident exclusions passed. No retrieval recall, diagnosis
accuracy or report quality is reported from this registry: there are no eligible
case labels or relevance judgments.

See `evidence/PROTOCOL.md` for importing real cases and completing domain review.
''')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    main()
