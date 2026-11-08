"""Conservative normalization: no lowercasing or punctuation stripping of content."""
import json
import re
from collections import Counter

from bs4 import BeautifulSoup

from .loaders import ROOT, dump
from .schema import document, stable_id, validate

TERM_RE = re.compile(r'(?:--?[A-Za-z][\w-]*|[A-Za-z0-9_]+(?:[./:+-][A-Za-z0-9_]+)+|[A-Za-z][A-Za-z0-9_]*|\d+)', re.UNICODE)


def terms(text):
    return TERM_RE.findall(text)


def clean(text):
    return '\n'.join(line.rstrip() for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')).strip()


def parse_body(body, html=False):
    if not html:
        return clean(body), {'parser': 'markdown_preserve', 'frontmatter_present': body.startswith('---')}
    soup = BeautifulSoup(body, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    candidates = soup.select('.entry-content, .post-content, .article-body, article, main')
    node = max(candidates, key=lambda n: len(n.get_text()), default=soup.body or soup)
    for code in node.find_all('pre'):
        code.replace_with('\n```\n' + code.get_text() + '\n```\n')
    for heading in node.find_all(re.compile('^h[1-6]$')):
        heading.replace_with('\n' + '#' * int(heading.name[1]) + ' ' + heading.get_text(' ', strip=True) + '\n')
    title = soup.title.get_text(' ', strip=True) if soup.title else None
    return clean(node.get_text('\n', strip=True)), {'parser': 'bs4_longest_article_main_or_content', 'html_title': title}


def prepare():
    records = json.loads((ROOT / 'data/parsed/source_records.json').read_text(encoding='utf-8'))
    docs, excluded, parsed = {}, [], []
    for row in records:
        text, metadata = parse_body(row['body'], row['source_type'] == 'incident')
        parsed.append(dict(url=row['url'], content=text, metadata=metadata))
        if len(terms(text)) < 40:
            excluded.append(dict(url=row['url'], reason='fewer than 40 lexical tokens; inspect parse or index page', stage='prepare'))
            continue
        title_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$|^#\s+(.+)$', text, re.M)
        title = next((g for g in title_match.groups() if g), None) if title_match else metadata.get('html_title')
        if row['source_type']=='incident':
            title=metadata.get('html_title') or title
        metadata.update(revision=row['revision'], path=row.get('path'), rights=row['terms'], language_method='English source selection; not automatic language detection')
        hangul = len(re.findall(r'[\uac00-\ud7af]', text)) / max(1, len(text))
        language = 'ko' if hangul > 0.05 else 'en'
        metadata['language_method'] = 'Hangul char share >5% => ko, otherwise English source selection; heuristic'
        d = document(row['source_type'], row['source_name'], row.get('path', row['url']), text,
                     title=title, source_url=row['url'], product=row['product'], language=language, metadata=metadata)
        docs[d['document_id']] = d
    original = json.loads((ROOT / 'data/parsed/qa_original.json').read_text(encoding='utf-8'))
    qa = []
    conflicts = []
    for row in original:
        context_ids = []
        for c in row.get('contexts', []):
            text = clean(c['text'])
            if not text:
                excluded.append(dict(query_id=row['id'], reason='empty QA context', stage='prepare'))
                continue
            # Include content hash in identity: same filename with changed text is not silently overwritten.
            identity = c['filename'] + '\n' + text
            d = document('qa_context', 'NVIDIA TechQA (IBM provenance)', identity, text,
                         title=text.split('\n')[0].removeprefix('Title: '), product='IBM (mixed products)', language='en',
                         metadata=dict(filename=c['filename'], revision=row['source_revision'], language_method='English dataset card', rights='Apache-2.0 declared by NVIDIA card; IBM upstream terms must also be considered'))
            docs[d['document_id']] = d
            context_ids.append(d['document_id'])
        qa.append(dict(query_id=row['id'], split=row['source_split'],
                       original_id_prefix=row['id'].split('_')[0], question=row['question'], answer=row.get('answer'),
                       is_impossible=row.get('is_impossible'), context_ids=context_ids))
    docs = sorted(docs.values(), key=lambda d: d['document_id'])
    validate(docs, qa)
    dump(ROOT / 'data/parsed/documents.json', parsed)
    dump(ROOT / 'data/cleaned/documents.json', docs)
    dump(ROOT / 'data/processed/documents.json', docs)
    dump(ROOT / 'data/processed/qa.json', qa)
    dump(ROOT / 'reports/exclusions.json', excluded)
    dump(ROOT / 'reports/preparation_counts.json', dict(selected=len(records), processed=len(docs), excluded=len(excluded), qa=len(qa), source_counts=dict(Counter(d['source_type'] for d in docs)),
        selected_by_type=dict(Counter(r['source_type'] for r in records)),
        successful_by_type=dict(Counter(d['source_type'] for d in docs)),qa_context_occurrences=sum(len(q['context_ids']) for q in qa)))
    return docs, qa


if __name__ == '__main__':
    documents, qa = prepare()
    print(len(documents), 'unified documents;', len(qa), 'QA')
