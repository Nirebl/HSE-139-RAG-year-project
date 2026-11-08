"""Deterministic identity and explicit missing-value semantics."""
import hashlib
from urllib.parse import urlparse

TYPES = {'documentation', 'runbook', 'incident', 'qa_context'}
OPTIONAL = ['product', 'version', 'component', 'language', 'created_at', 'updated_at']


def stable_id(source, identity):
    return source + '_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]


def document(source_type, source_name, identity, content, title=None, source_url=None, **kwargs):
    row = dict(document_id=stable_id(source_type, identity), source_type=source_type,
               source_name=source_name, source_url=source_url, title=title, content=content,
               **{k: kwargs.get(k) for k in OPTIONAL}, metadata=kwargs.get('metadata', {}))
    row['metadata']['field_status'] = {
        k: 'observed' if row[k] is not None else 'not_published'
        for k in ['title', 'source_url'] + OPTIONAL}
    if row['created_at'] is None and source_type == 'qa_context':
        row['metadata']['field_status']['created_at'] = 'unknown'
    return row


def validate(documents, qa):
    ids = [d['document_id'] for d in documents]
    assert len(ids) == len(set(ids)), 'Duplicate document identity'
    assert len({q['query_id'] for q in qa}) == len(qa)
    for d in documents:
        assert d['source_type'] in TYPES
        assert all(isinstance(d[k],str) and d[k] for k in ['document_id','source_type','source_name'])
        assert d['title'] is None or isinstance(d['title'],str)
        assert isinstance(d['content'], str) and d['content'].strip()
        assert isinstance(d['metadata'], dict)
        assert set(d['metadata']['field_status'].values()) <= {'observed','not_applicable','not_published','lost_in_parsing','unknown'}
        assert all(d[k] is None or isinstance(d[k], str) for k in OPTIONAL)
        assert d['source_url'] is None or urlparse(d['source_url']).scheme in {'http', 'https'}
    for q in qa:
        assert isinstance(q['question'],str) and q['question'].strip()
        assert q['answer'] is None or isinstance(q['answer'],str)
        assert q['is_impossible'] is None or isinstance(q['is_impossible'],bool)
        assert set(q['context_ids']) <= set(ids), 'Dangling QA context reference'
    return True
