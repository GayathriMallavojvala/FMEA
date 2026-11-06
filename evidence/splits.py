"""Conservative connected-component splits: sources, incidents and duplicates stay together."""
import hashlib


def leakage_groups(registry):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            parent[max(ra,rb)]=min(ra,rb)
    for s in registry['sources']:
        find('s:'+s['source_id'])
    for inc in registry['incidents']:
        node='i:'+inc['incident_id']
        for sid in inc['source_ids']:
            union(node,'s:'+sid)
        if inc.get('duplicate_group'):
            union(node,'d:'+inc['duplicate_group'])
    for p in registry['passages']:
        node='p:'+p['passage_id']
        union(node,'s:'+p['source_id'])
        for iid in p['incident_ids']:
            union(node,'i:'+iid)
        normalized=' '.join(p['text'].lower().split())
        union(node,'t:'+hashlib.sha256(normalized.encode()).hexdigest())
    components={}
    for node in sorted(parent):
        components.setdefault(find(node),[]).append(node)
    groups={}
    for nodes in components.values():
        gid='GROUP-'+hashlib.sha256('\n'.join(nodes).encode()).hexdigest()[:16]
        for node in nodes:
            groups[node]=gid
    return groups


def make_split(registry, seed=20261029):
    groups=leakage_groups(registry)
    distinct=sorted(set(groups.values()),key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(distinct)<3:
        raise ValueError('At least three independent source groups needed')
    n_test=max(1,len(distinct)//5)
    n_valid=max(1,len(distinct)//5)
    assignment={g:('test' if i<n_test else 'validation' if i<n_test+n_valid else 'train') for i,g in enumerate(distinct)}
    result=dict(schema_version=1,seed=seed, grouping='source+incident+declared_duplicate+exact_normalized_text',
        group_partitions=assignment,
        passages={p['passage_id']:dict(group_id=groups['p:'+p['passage_id']],partition=assignment[groups['p:'+p['passage_id']]]) for p in registry['passages']},
        incidents={inc['incident_id']:dict(group_id=groups['i:'+inc['incident_id']],partition=assignment[groups['i:'+inc['incident_id']]]) for inc in registry['incidents']})
    validate_split(registry,result)
    return result


def validate_split(registry, split):
    groups=leakage_groups(registry)
    if set(split['passages'])!={p['passage_id'] for p in registry['passages']}:
        raise ValueError('Split does not match registry passage IDs')
    if set(split['incidents'])!={i['incident_id'] for i in registry['incidents']}:
        raise ValueError('Split does not match registry incident IDs')
    partitions={}
    for key,prefix in [('passages','p:'),('incidents','i:')]:
        for ident,record in split[key].items():
            gid=groups[prefix+ident]
            if record['group_id']!=gid or record['partition'] not in {'train','validation','test'}:
                raise ValueError('Invalid group or partition')
            if split['group_partitions'].get(gid)!=record['partition']:
                raise ValueError('Inconsistent group partition')
            if gid in partitions and partitions[gid]!=record['partition']:
                raise ValueError('Leakage: group occurs in multiple partitions')
            partitions[gid]=record['partition']
    return True


def training_passages(registry, split, query_incident_id):
    """Explicit query ID is mandatory. No held-out passage enters the fitted index."""
    validate_split(registry,split)
    if query_incident_id not in split['incidents']:
        raise ValueError('Unknown query incident; cannot enforce incident exclusion')
    query_group=split['incidents'][query_incident_id]['group_id']
    return [p for p in registry['passages']
            if split['passages'][p['passage_id']]['partition']=='train'
            and split['passages'][p['passage_id']]['group_id']!=query_group]
