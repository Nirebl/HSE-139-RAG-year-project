"""Execute real local-snapshot EDA; export measurements and review materials."""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.decomposition import PCA

from . import visualization as viz
from .augmentation import make_augmentations
from .embeddings import chunks, encode, load_model
from .loaders import CONFIG, ROOT, dump
from .preprocessing import prepare, terms
from .statistics import describe, duplicates, features, vocabulary

QUERIES = [
 ('c01','PostgreSQL replication lag increases and database connections fail. What should I check?'),
 ('c02','Kubernetes pods restart repeatedly after deployment. How do I diagnose the failure?'),
 ('c03','DNS lookups return SERVFAIL and services cannot resolve hostnames. What could be wrong?'),
 ('c04','Redis memory usage grows and clients experience high latency. What can I inspect?'),
 ('c05','A PostgreSQL primary was accidentally deleted. How can backups and replication help recovery?'),
 ('c06','Kubernetes service endpoints are unavailable and requests time out. How do I debug networking?'),
 ('c07','TLS certificate expiration causes connection failures. How can I diagnose and renew certificates?'),
 ('c08','Database connections are exhausted during a traffic spike. What diagnostics are useful?'),
 ('c09','A configuration rollout causes widespread HTTP 500 errors. How can I investigate and recover?'),
 ('c10','Disk space is exhausted on a database server. Which diagnostics and mitigations are documented?')
]


def csv(frame, name):
    path=ROOT / 'reports' / (name+'.csv')
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)


def aggregate(vectors, fragments, documents):
    positions={}
    for i,c in enumerate(fragments):
        positions.setdefault(c['document_id'],[]).append(i)
    rows=np.stack([vectors[positions[d['document_id']]].mean(axis=0) for d in documents])
    return rows / np.maximum(np.linalg.norm(rows,axis=1,keepdims=True),1e-12)


def run():
    import torch
    torch.set_num_threads(4)
    np.random.seed(CONFIG['seed'])
    documents,qa=prepare()
    model=load_model()
    tokenizer=model.tokenizer
    stats=features(documents,tokenizer)
    csv(stats,'document_features')
    csv(describe(stats),'length_statistics')
    totals=stats.groupby('source_type')[['characters','words','tokens']].sum().reset_index()
    csv(totals,'volume_totals')
    missing=[]
    for source in sorted(stats.source_type.unique()):
        subset=[d for d in documents if d['source_type']==source]
        for field in documents[0]:
            n=sum(d[field] is None or d[field]=='' for d in subset)
            missing.append(dict(source_type=source,field=field,missing=n,total=len(subset),fraction=n/len(subset)))
    missing=pd.DataFrame(missing)
    csv(missing,'missingness')
    near=duplicates(documents)
    csv(near,'duplicate_pairs')
    overlap,lexical=vocabulary(documents)
    overlap.to_csv(ROOT / 'reports/vocabulary_overlap.csv')
    dump(ROOT / 'reports/lexical_profiles.json',lexical)
    bigrams=[]
    for source in sorted(stats.source_type.unique()):
        counts=Counter()
        for d in documents:
            if d['source_type']==source:
                ts=[t.lower() for t in terms(d['content'])]
                counts.update(zip(ts,ts[1:]))
        bigrams.extend(dict(source_type=source,bigram=' '.join(t),count=n) for t,n in counts.most_common(20))
    csv(pd.DataFrame(bigrams),'top_bigrams')
    # Questions and answers are separate populations from supporting contexts.
    qa_stats=[]
    for q in qa:
        for kind in ['question','answer']:
            text=q[kind] or ''
            unit='answer_placeholder' if kind=='answer' and text.strip().lower() in {'','-','n/a','none'} else kind
            qa_stats.append(dict(query_id=q['query_id'],source_type=unit,split=q['split'],
                characters=len(text),words=len(text.split()),tokens=len(tokenizer(text,add_special_tokens=False,truncation=False)['input_ids'])))
    csv(pd.DataFrame(qa_stats),'qa_lengths')
    csv(describe(pd.DataFrame(qa_stats)),'qa_length_statistics')
    prefixes=Counter(q['original_id_prefix'] for q in qa)
    question_groups={}
    for q in qa:
        question_groups.setdefault(' '.join(q['question'].lower().split()),[]).append(q['query_id'])
    qdup=[v for v in question_groups.values() if len(v)>1]
    placeholders={'','-','n/a','none'}
    audit=dict(rows=len(qa),splits=dict(Counter(q['split'] for q in qa)),id_prefixes=dict(prefixes),
          answerability=dict(Counter(str(q['is_impossible']) for q in qa)),
          empty_answers=sum(not (q['answer'] or '').strip() for q in qa),empty_questions=sum(not q['question'].strip() for q in qa),
          placeholder_answers=sum((q['answer'] or '').strip().lower() in placeholders for q in qa),
          answerable_placeholder_answers=sum(not q['is_impossible'] and (q['answer'] or '').strip().lower() in placeholders for q in qa),
          contexts_histogram=dict(Counter(len(q['context_ids']) for q in qa)),duplicate_question_groups=qdup,
          cross_prefix_duplicate_questions=[v for v in qdup if len({x.split('_')[0] for x in v})>1],
          context_shared_across_prefixes=sum(len({q['original_id_prefix'] for q in qa if d['document_id'] in q['context_ids']})>1 for d in documents if d['source_type']=='qa_context'),
          context_not_answer_span=sum(not q['is_impossible'] and (q['answer'] or '').strip().lower() not in placeholders and not any(q['answer'] in d['content'] for d in documents if d['document_id'] in q['context_ids']) for q in qa),
          split_warning='Published train split contains TRAIN/DEV ID prefixes; prefixes are retained as provenance, not invented dataset splits. No test file published in this NVIDIA snapshot.')
    dump(ROOT / 'reports/qa_audit.json',audit)
    question_docs=[dict(document_id=q['query_id'],source_type='question',content=q['question']) for q in qa]
    csv(duplicates(question_docs),'question_duplicate_pairs')
    # Include ALL English infrastructure docs and 60 seeded IBM contexts.
    rng=np.random.default_rng(CONFIG['seed'])
    contexts=[d for d in documents if d['source_type']=='qa_context' and d['language']=='en']
    chosen_indices=sorted(rng.choice(len(contexts),min(CONFIG['embedding_documents_per_source'],len(contexts)),replace=False))
    selected=sorted([d for d in documents if d['source_type']!='qa_context' and d['language']=='en']+[contexts[i] for i in chosen_indices],key=lambda x:x['document_id'])
    fragments=chunks(selected,tokenizer)
    dump(ROOT / 'data/processed/embedding_chunks.json',fragments)
    vectors=encode(model,[c['content'] for c in fragments],'chunks')
    doc_vectors=aggregate(vectors,fragments,selected)
    np.save(ROOT / 'data/processed/document_embeddings.npy',doc_vectors)
    dump(ROOT / 'data/processed/embedding_document_ids.json',[d['document_id'] for d in selected])
    dump(ROOT / 'reports/embedding_protocol.json',dict(model=CONFIG['model'],
        revision=json.loads((ROOT/'data/source_manifest.json').read_text(encoding='utf-8'))['model_revision'],
        tokenizer=type(tokenizer).__name__,max_length=model.max_seq_length,chunk_tokens=CONFIG['chunk_tokens'],
        overlap=CONFIG['overlap_tokens'],pooling=str(model[1]),normalization=True,device='cpu',batch_size=CONFIG['batch_size'],
        documents=len(selected),chunks=len(fragments),aggregation='unweighted mean of overlapping chunk embeddings, L2 normalized; document is similarity unit',
        selection='all English non-QA documents + seeded 60 English QA contexts',
        fraction_over_limit=float((stats.tokens+2>model.max_seq_length).mean()),
        non_english_excluded=[d['document_id'] for d in documents if d['language']!='en']))
    sim=doc_vectors@doc_vectors.T
    pair_groups={}
    for i in range(len(selected)):
        for j in range(i+1,len(selected)):
            if selected[i]['content']==selected[j]['content']:
                continue
            key=tuple(sorted([selected[i]['source_type'],selected[j]['source_type']]))
            pair_groups.setdefault(key,[]).append((i,j))
    pairs=[]
    for (a,b),pool in sorted(pair_groups.items()):
        for ix in rng.choice(len(pool),min(CONFIG['pair_samples'],len(pool)),replace=False):
            i,j=pool[ix]
            pairs.append(dict(source_a=a,source_b=b,group='within' if a==b else 'cross',
                              left_id=selected[i]['document_id'],right_id=selected[j]['document_id'],cosine=float(sim[i,j])))
    pairs=pd.DataFrame(pairs); csv(pairs,'semantic_pairs')
    csv(pairs.groupby(['source_a','source_b']).cosine.agg(['count','mean','median','std']).reset_index(),'semantic_pair_summary')
    pca=PCA(n_components=2,random_state=42)
    coords=pca.fit_transform(doc_vectors)
    dump(ROOT / 'reports/pca.json',dict(explained_variance=pca.explained_variance_ratio_.tolist()))
    viz.overview(stats)
    viz.heatmap(overlap,'vocabulary_overlap','Vocabulary Jaccard (equal 10k-token budget)')
    viz.heatmap(missing.pivot(index='source_type',columns='field',values='fraction'),'missingness','Missing values (fraction per source)')
    viz.semantic_plot(coords,[d['source_type'] for d in selected],pairs)
    # Retrieval candidates over a saved common English infrastructure corpus; no answer leakage.
    infra_ix=[i for i,d in enumerate(selected) if d['source_type']!='qa_context']
    infra=[selected[i] for i in infra_ix]
    bm25=BM25Okapi([[t.lower() for t in terms(d['content'])] for d in infra])
    qvec=encode(model,[q for _,q in QUERIES],'coverage_queries')
    review=[]
    candidate_sets=[]
    for qi,(qid,text) in enumerate(QUERIES):
        dense=qvec[qi]@doc_vectors[infra_ix].T
        sparse=bm25.get_scores([t.lower() for t in terms(text)])
        for source in ['documentation','runbook','incident']:
            indexes=[i for i,d in enumerate(infra) if d['source_type']==source]
            top=sorted(indexes,key=lambda i:(-dense[i],infra[i]['document_id']))[:3]
            sparse_top=sorted(indexes,key=lambda i:(-sparse[i],infra[i]['document_id']))[:3]
            candidate_sets.append(dict(query_id=qid,source_type=source,dense_top_ids=[infra[i]['document_id'] for i in top],bm25_top_ids=[infra[i]['document_id'] for i in sparse_top],overlap_at3=len(set(top)&set(sparse_top))/max(1,len(top))))
            for rank,i in enumerate(top,1):
                d=infra[i]
                cix=[j for j,c in enumerate(fragments) if c['document_id']==d['document_id']]
                best=max(cix,key=lambda j:float(qvec[qi]@vectors[j]))
                review.append(dict(query_id=qid,query=text,document_id=d['document_id'],source_type=source,
                     url=d['source_url'],rank=rank,cosine=float(dense[i]),evidence=fragments[best]['content'],
                     preliminary_label='unverified_candidate',human_label='',reviewer='',reviewed_at='',comment=''))
    review_path=ROOT / 'reports/coverage_review.csv'
    # Preserve human work across reruns by exact query/document/evidence identity.
    if review_path.exists():
        old=pd.read_csv(review_path,keep_default_na=False)
        previous={(r['query_id'],r['document_id'],r['evidence']):r for r in old.to_dict('records')}
        for row in review:
            prior=previous.get((row['query_id'],row['document_id'],row['evidence']),{})
            for field in ['human_label','reviewer','reviewed_at','comment']:
                row[field]=prior.get(field,'')
    csv(pd.DataFrame(review),'coverage_review')
    csv(pd.DataFrame(candidate_sets),'retrieval_candidates')
    dump(ROOT / 'reports/retrieval_protocol.json',dict(candidate_document_ids=[d['document_id'] for d in infra],
        queries=[dict(query_id=k,text=v,provenance='synthetic_agent_diagnostic_scenario') for k,v in QUERIES],qrels=[],
        evaluation='pending human review; no Recall/MRR/nDCG without qrels; IBM QA not mixed into infrastructure benchmark'))
    aug=make_augmentations(qa)
    avec=encode(model,[r['original'] for r in aug],'augmentation_originals')
    bvec=encode(model,[r['transformed'] for r in aug],'augmentation_variants')
    for i,row in enumerate(aug):
        row['cosine']=float(avec[i]@bvec[i])
    csv(pd.DataFrame(aug),'augmentations')
    for method in ['paraphrase','typo']:
        subset=[r for r in aug if r['method']==method]
        print(method,'entity mismatches:',[(r['original_id'],r['missing_entities']) for r in subset if not r['entity_subset_preserved']])
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ['paraphrase','typo']:
        vals=[r['cosine'] for r in aug if r['method']==method]
        axes[0].scatter([method]*len(vals),vals,alpha=.6)
        vals=[r['length_delta'] for r in aug if r['method']==method]
        axes[1].scatter([method]*len(vals),vals,alpha=.6)
    axes[0].set(ylabel='Cosine to original',title='20 original questions, two variants each')
    axes[1].set(ylabel='Character count delta',title='Query length changes')
    viz.save('augmentations')
    structure=stats.groupby('source_type')[['headings','technical_density','code_fraction','commands','timeline','root_cause','corrective_actions','symptoms','conditions','expected_result','stale_candidate']].mean()
    structure.to_csv(ROOT / 'reports/structure_summary.csv')
    # QA-to-doc domain shift: compare original QA prompts with all four source types.
    proximity=[]
    for i,row in enumerate(aug):
        if row['method']!='paraphrase':
            continue
        for source in sorted(stats.source_type.unique()):
            ix=[j for j,d in enumerate(selected) if d['source_type']==source]
            j=max(ix,key=lambda j:float(avec[i]@doc_vectors[j]))
            proximity.append(dict(query_id=row['original_id'],source_type=source,document_id=selected[j]['document_id'],cosine=float(avec[i]@doc_vectors[j]),title=selected[j]['title']))
    csv(pd.DataFrame(proximity),'qa_source_proximity')
    result=dict(counts=stats.source_type.value_counts().to_dict(),qa=audit,near_pairs=len(near),exact_pairs=int(near.exact.sum()) if len(near) else 0,
                encoding_affected=int((stats.encoding_replacement>0).sum()),truncation_fraction=float((stats.tokens+2>model.max_seq_length).mean()),
                augmentations=pd.DataFrame(aug).groupby('method')[['cosine','length_delta','entity_subset_preserved']].mean().to_dict(),
                coverage_rows=len(review),human_reviewed=sum(str(r['human_label']) in ['0','1','2'] for r in review),
                embedding_documents=len(selected),chunks=len(fragments),languages=stats.language.value_counts().to_dict())
    dump(ROOT/'reports/results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    run()
