"""Auditable structural and lexical measurements (heuristics are not gold labels)."""
import hashlib
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .loaders import CONFIG
from .preprocessing import terms


def features(documents, tokenizer):
    rows = []
    for d in documents:
        text = d['content']
        tokens = terms(text)
        fences = re.findall(r'```[^\n]*\n(.*?)```', text, re.S)
        headings = re.findall(r'^#{1,6}\s+(.+)$', text, re.M)
        rows.append(dict(document_id=d['document_id'], source_type=d['source_type'],
             characters=len(text), words=len(text.split()), tokens=len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']),
             lexical_tokens=len(tokens), type_token_ratio=len(set(t.lower() for t in tokens))/max(1,len(tokens)),
             technical_density=sum(bool(re.search(r'[\d_./:+-]|[A-Z]{2,}', t)) for t in tokens)/max(1,len(tokens)),
             code_fraction=sum(len(c) for c in fences)/max(1,len(text)), headings=len(headings),
             heading_characters=np.mean([len(h) for h in headings]) if headings else 0,
             links=len(re.findall(r'https?://[^\s<>\])]+',text)),
             commands=len(re.findall(r'\b(?:kubectl|sudo|psql|redis-cli|curl|systemctl|gitlab-ctl)\b',text)),
             version_candidates=len(re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b',text)),
             encoding_replacement=text.count('\ufffd'), html_artifacts=len(re.findall(r'<[A-Za-z][^>]+>',text)),
             hugo_artifacts=text.count('{{'), non_ascii_fraction=sum(ord(c)>127 for c in text)/len(text),
             timeline=bool(re.search(r'timeline|\b\d{2}:\d{2}\b',text,re.I)),
             root_cause=bool(re.search(r'root cause|caus(?:e|ed)|contributing',text,re.I)),
             corrective_actions=bool(re.search(r'action items|prevent|remediat|follow.up|what.*improv',text,re.I)),
             symptoms=bool(re.search(r'symptom|error|failure|unavailable|latency',text,re.I)),
             conditions=bool(re.search(r'\bif\b|\bwhen\b|prerequisite',text,re.I)),
             expected_result=bool(re.search(r'expected|should (?:see|return)|verify|confirm',text,re.I)),
             stale_candidate=bool(re.search(r'deprecated|outdated|obsolete|no longer',text,re.I)),
             language=d['language']))
    return pd.DataFrame(rows)


def describe(frame, group='source_type', columns=('characters','words','tokens')):
    return frame.groupby(group)[list(columns)].describe(percentiles=[.25,.50,.75,.90,.95,.99]).stack(0, future_stack=True).reset_index().rename(columns={'level_1':'unit'})


def duplicates(documents):
    texts = [d['content'] for d in documents]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=30000)
    matrix = vectorizer.fit_transform(texts)
    sim = (matrix @ matrix.T).toarray()
    rows = []
    for i,j in zip(*np.where(np.triu(sim >= CONFIG['near_duplicate_cosine'], k=1))):
        rows.append(dict(left_id=documents[i]['document_id'], right_id=documents[j]['document_id'],
                         exact=texts[i]==texts[j], cosine=float(sim[i,j]),
                         left_source=documents[i]['source_type'], right_source=documents[j]['source_type']))
    return pd.DataFrame(rows, columns=['left_id','right_id','exact','cosine','left_source','right_source'])


def vocabulary(documents):
    rng = np.random.default_rng(CONFIG['seed'])
    sources = sorted({d['source_type'] for d in documents})
    counters = {}
    for source in sources:
        tokens = [t.lower() for d in documents if d['source_type']==source for t in terms(d['content'])]
        # Equal token budget, seeded sampling: a sensitivity view, not population inference.
        counters[source] = Counter(rng.choice(tokens, min(10000,len(tokens)), replace=False))
    overlap = pd.DataFrame(index=sources, columns=sources, dtype=float)
    for a in sources:
        for b in sources:
            aa,bb = set(counters[a]),set(counters[b])
            overlap.loc[a,b] = len(aa&bb)/max(1,len(aa|bb))
    lexical = []
    for source,counter in counters.items():
        lexical.append(dict(source_type=source, sampled_tokens=sum(counter.values()), vocabulary=len(counter),
                            hapax=sum(n==1 for n in counter.values()), top_terms=counter.most_common(20),
                            top_technical=[(t,n) for t,n in counter.most_common() if re.search(r'[\d_./:+-]',t)][:20]))
    return overlap, lexical
