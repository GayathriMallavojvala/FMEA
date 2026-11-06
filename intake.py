"""Typo-tolerant lexical retrieval plus constrained semantic intake.
Never overwrite the original observation or force unsupported defects into a family.
"""
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFINITIONS={
 'Scratches':'Long narrow grooves, scoring, drag marks or lengthwise lines caused by surface contact.',
 'Crazing':'Many fine surface cracks forming an interconnected web or network. Not every crack is crazing.',
 'Inclusion':'Foreign or non-metallic particles trapped or embedded in steel.',
 'Patches':'Irregular blotches or patchy discoloured areas on the steel surface.',
 'Pitted_Surface':'Small recessed pits, holes or cavities visible on the surface.',
 'Rolled-in_Scale':'Oxide or scale pressed into the steel surface during rolling.',
 'Crescent_Gap':'Crescent-shaped or semicircular missing section or notch.',
 'Waist_Folding':'Folded or overlapping material at a waist region; folding defects.',
 'Weld_Line_Defect':'Failure or imperfection at a welded seam or coil joint, such as splitting or incomplete fusion.',
 'Water_Oil_Spot':'Water or oil spots, stains, oily residue or droplet marks.'}
ALIASES={
 'Scratches':['scratches','scratch','scratched','scoring','score marks','drag marks'],
 'Crazing':['crazing'], 'Inclusion':['inclusion','inclusions'],
 'Patches':['patches','patchy discoloration','blotches'],
 'Pitted_Surface':['pitting','pitted surface','pitted','pits','tiny cavities'],
 'Rolled-in_Scale':['rolled in scale','embedded oxide','pressed in scale'],
 'Crescent_Gap':['crescent gap','semicircular notch'],
 'Waist_Folding':['waist folding','folding','folded material'],
 'Weld_Line_Defect':['weld line defect','weld defect','split weld','weld crack','weld cracks','seam split'],
 'Water_Oil_Spot':['oil spot','oil spots','water spot','water spots','oil stains','oily stains']}
SPELLINGS={'surfsce':'surface','surfce':'surface','surfac':'surface','craks':'cracks','cracs':'cracks','sctraches':'scratches','scraches':'scratches','scrathes':'scratches','scracthes':'scratches','scrathces':'scratches','sratches':'scratches','scartches':'scratches','crazng':'crazing','pittting':'pitting','inclusons':'inclusions','rollng':'rolling'}
VOCAB={'scratches','scratch','scratched','crazing','inclusion','inclusions','patches','pitted','pitting','folding','surface','cracks','rolling','stains'}

def normalize(text):
    words=re.findall(r"[a-z]+|[^a-z]+",text.lower().replace('_',' ').replace('-',' '))
    out=[]
    for w in words:
        if w in SPELLINGS:w=SPELLINGS[w]
        elif w.isalpha() and len(w)>=5 and w not in VOCAB:
            near=difflib.get_close_matches(w,VOCAB,n=1,cutoff=.86)
            if near:w=near[0]
        out.append(w)
    return ''.join(out)


def retrieve_families(text):
    """Character TF-IDF tolerates partial words and spelling variations; ranking only."""
    ids=list(DEFINITIONS)
    docs=[f.replace('_',' ')+' '+DEFINITIONS[f]+' '+' '.join(ALIASES[f]) for f in ids]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit(docs)
    scores=cosine_similarity(vec.transform([normalize(text)]),vec.transform(docs))[0]
    return sorted([(f,float(v)) for f,v in zip(ids,scores)],key=lambda x:-x[1])


def interpret(text,semantic):
    cleaned=normalize(text)
    clauses=re.split(r'[.;!?\n]|\b(?:but|however)\b',cleaned)
    # Do not treat a negated or merely requested inspection as an observed failure.
    excluded=r'\b(?:no|not|without|possible|possibly|maybe|suspected|check for|inspect for|look for|training|historical|previous|yesterday)\b'
    observed=[c for c in clauses if c.strip() and not re.search(excluded,c)]
    matches=set()
    for f,aliases in ALIASES.items():
        if any(re.search(r'\b'+re.escape(a)+r'\b',c) for c in observed for a in aliases):matches.add(f)
    crack=any(re.search(r'\bcracks?\b',c) for c in observed)
    network=any(re.search(r'network|web|mesh|interconnected|intersecting',c) and re.search(r'cracks?',c) for c in observed)
    if crack and (not matches or matches=={'Crazing'}):
        if network and '?' not in text:return {'family':'Crazing','basis':'Reported interconnected crack pattern'}
        return {'clarify':'cracks','message':'Does the surface have many fine cracks joined together like a web or network?'}
    if crack and matches and not matches <= {'Crazing','Weld_Line_Defect'}:
        return {'clarify':'family','message':'Cracks and another defect pattern are described. Confirm the observed family or investigate them separately.'}
    if len(matches)==1 and '?' not in text:
        return {'family':next(iter(matches)),'basis':'Typo-normalized observation match'}
    if len(matches)>1:return {'clarify':'family','message':'More than one defect is described. Investigate one observed defect at a time, or confirm its family.'}
    if not observed:return {'clarify':'family','message':'Please describe a defect that was actually observed, including its shape and location.'}
    # Unfamiliar everyday language is interpreted by the LLM, with exact-quote evidence.
    result=semantic(text,retrieve_families(text))
    family=result.get('family');quote=result.get('evidence_quote','')
    if family not in DEFINITIONS or not isinstance(quote,str) or not quote.strip() or quote.casefold() not in text.casefold():
        return {'clarify':'family','message':result.get('question') or 'Please add the defect shape, pattern and location, or confirm its family.'}
    if result.get('question'):
        return {'clarify':'family','message':result['question']}
    return {'family':family,'basis':'LLM interpretation grounded in reported text','evidence_quote':quote}
