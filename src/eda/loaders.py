"""Bounded online acquisition. Original responses are immutable local snapshots."""
import concurrent.futures
import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / 'config/eda.json').read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class Snapshot:
    def __init__(self):
        self.entries, self.errors = [], []

    def get(self, url, source, revision=None, terms='unknown; local research only'):
        key = hashlib.sha256(url.encode()).hexdigest()
        path = ROOT / 'data/raw' / key
        sidecar = path.with_suffix('.json')
        try:
            if path.exists() and sidecar.exists():
                info = json.loads(sidecar.read_text(encoding='utf-8'))
                body = path.read_bytes()
                if hashlib.sha256(body).hexdigest() != info['sha256']:
                    raise ValueError('Snapshot checksum mismatch')
            else:
                for attempt in range(3):
                    response = requests.get(url, timeout=CONFIG['timeout_seconds'], stream=True,
                                            headers={'User-Agent': 'IncidentCopilot-EDA/1.0 (bounded academic study)'})
                    if response.status_code in (429, 502, 503) and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    response.raise_for_status()
                    break
                parts, size = [], 0
                for part in response.iter_content(65536):
                    size += len(part)
                    if size > CONFIG['max_download_bytes']:
                        raise ValueError('Download size limit exceeded')
                    parts.append(part)
                body = b''.join(parts)
                info = dict(url=url, final_url=response.url, source=source, revision=revision,
                            retrieved_at=datetime.now(timezone.utc).isoformat(),
                            sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                            terms=terms, terms_check='local_only' if 'unknown' in terms else 'declared_license_reviewed',
                            raw_path=str(path.relative_to(ROOT)))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                dump(sidecar, info)
            self.entries.append(info)
            return body.decode('utf-8-sig')
        except Exception as exc:
            self.errors.append(dict(url=url, source=source, error=str(exc), stage='download'))
            return None

    def json(self, *args, **kwargs):
        body = self.get(*args, **kwargs)
        return json.loads(body) if body is not None else None


def acquire():
    snap = Snapshot()
    selected = []
    audits = {}
    # NVIDIA packages one train.json; inspect repo files, not guessed split names.
    hf = 'https://huggingface.co'
    meta = snap.json(f'{hf}/api/datasets/nvidia/TechQA-RAG-Eval', 'techqa')
    rev = meta['sha']
    card = snap.get(f'{hf}/datasets/nvidia/TechQA-RAG-Eval/raw/{rev}/README.md', 'techqa', rev, 'Apache-2.0 declared by dataset card')
    files = [f['rfilename'] for f in meta['siblings'] if f['rfilename'].endswith('.json')]
    qa = []
    for file in files:
        rows = snap.json(f'{hf}/datasets/nvidia/TechQA-RAG-Eval/resolve/{rev}/{file}', 'techqa', rev, 'Apache-2.0 declared by dataset card; upstream provenance retained')
        if rows is None:
            continue
        if isinstance(rows, dict):
            raise ValueError(f'Unexpected QA container: {list(rows)}')
        audits[file] = dict(count=len(rows), fields={k: sorted({type(r.get(k)).__name__ for r in rows}) for k in sorted(set().union(*(r.keys() for r in rows)))})
        for row in rows:
            qa.append(dict(row, source_split=Path(file).stem, source_revision=rev))
    rng = random.Random(CONFIG['seed'])
    qa = sorted(qa, key=lambda x: x['id'])
    if len(qa) > CONFIG['limits']['qa']:
        qa = rng.sample(qa, CONFIG['limits']['qa'])
    dump(ROOT / 'data/parsed/qa_original.json', qa)
    # IBM is provenance/schema documentation; do not download 800k technotes.
    ibm = snap.json('https://api.github.com/repos/IBM/techqa/commits/master', 'ibm')
    ibm_rev = ibm['sha']
    for file in ['README.md', 'LICENSE']:
        snap.get(f'https://raw.githubusercontent.com/IBM/techqa/{ibm_rev}/{file}', 'ibm', ibm_rev)
    # Stratify by top-level section and file-size quartile, then round robin.
    tree = snap.json('https://api.github.com/repos/kubernetes/website/git/trees/main?recursive=1', 'kubernetes')
    if tree.get('truncated'):
        raise ValueError('Kubernetes Git tree truncated')
    krev = tree['sha']
    snap.get(f'https://raw.githubusercontent.com/kubernetes/website/{krev}/LICENSE', 'kubernetes', krev, 'CC-BY-4.0 documentation; license snapshot retained')
    pool = [x for x in tree['tree'] if x['path'].startswith('content/en/docs/') and x['path'].endswith('.md') and not x['path'].endswith('_index.md')]
    groups = {}
    ordered = sorted(pool, key=lambda x: (x.get('size', 0), x['path']))
    for i, row in enumerate(ordered):
        key = (row['path'].split('/')[3], min(3, i * 4 // len(ordered)))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while groups and len(chosen) < CONFIG['limits']['documentation']:
        for key in sorted(list(groups)):
            chosen.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(chosen) == CONFIG['limits']['documentation']:
                break
    for row in chosen:
        selected.append(dict(source_type='documentation', source_name='Kubernetes', path=row['path'],
                             revision=krev, url=f'https://raw.githubusercontent.com/kubernetes/website/{krev}/{row["path"]}',
                             terms='CC-BY-4.0 documentation', product='Kubernetes'))
    # GitLab repository archive: one pinned revision, bounded compressed transfer.
    base = 'https://gitlab.com/api/v4/projects/gitlab-com%2Frunbooks'
    commit = snap.json(base + '/repository/commits/main', 'gitlab')
    if commit is None:
        commit = snap.json(base + '/repository/commits/master', 'gitlab')
    grev = commit['id']
    license_text = snap.get(base + f'/repository/files/LICENSE/raw?ref={grev}', 'gitlab', grev)
    terms = 'MIT; repository LICENSE snapshot' if license_text and 'Permission is hereby granted, free of charge' in license_text else 'unknown; local research only'
    paths, page = [], 1
    while page <= 40:
        rows = snap.json(base + f'/repository/tree?path=docs&recursive=true&per_page=100&page={page}&ref={grev}', 'gitlab', grev, terms)
        if not rows:
            break
        paths.extend(x['path'] for x in rows if x['type'] == 'blob' and x['path'].endswith('.md'))
        if len(rows) < 100:
            break
        page += 1
    # Include common infrastructure; half seeded general sample to expose domain shift.
    common = [p for p in sorted(paths) if re.search(r'postgres|redis|kube|patroni|dns', p, re.I)]
    other = sorted(set(paths) - set(common))
    n = CONFIG['limits']['runbook']
    gp = rng.sample(common, min(n // 2, len(common)))
    gp += rng.sample(other, min(n-len(gp), len(other)))
    for path in gp:
        selected.append(dict(source_type='runbook', source_name='GitLab Runbooks', path=path, revision=grev,
                             url=base + f'/repository/files/{quote(path, safe="")}/raw?ref={grev}', terms=terms, product='GitLab'))
    catalog_meta = snap.json('https://api.github.com/repos/danluu/post-mortems/commits/master', 'postmortem_catalog')
    drev = catalog_meta['sha']
    catalog = snap.get(f'https://raw.githubusercontent.com/danluu/post-mortems/{drev}/README.md', 'postmortem_catalog', drev)
    # Diverse organizations, original publisher pages. Only select links in catalog.
    links = list(dict.fromkeys(re.findall(r'\]\((https?://[^\s)]+)\)', catalog)))
    domains = ['about.gitlab.com', 'github.blog', 'blog.cloudflare.com', 'slack.engineering', 'engineering.atspotify.com', 'allegro.tech', 'aws.amazon.com']
    incident_urls = []
    for domain in domains:
        candidates = [u for u in links if domain in u and 'web.archive.org' not in u]
        incident_urls.extend(candidates[:2])
    for url in incident_urls[:CONFIG['limits']['incident']]:
        selected.append(dict(source_type='incident', source_name=url.split('/')[2], url=url,
                             revision='site snapshot; sha256 in manifest', terms='unknown; local research only', product=None))
    def fetch(row):
        body = snap.get(row['url'], row['source_name'], row['revision'], row['terms'])
        return dict(row, body=body) if body else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        records = [r for r in executor.map(fetch, selected) if r]
    model = snap.json(f'{hf}/api/models/{CONFIG["model"]}', 'embedding_model')
    mrev = model['sha']
    snap.get(f'{hf}/{CONFIG["model"]}/raw/{mrev}/README.md', 'embedding_model', mrev, 'Apache-2.0 model card')
    dump(ROOT / 'data/parsed/source_records.json', records)
    dump(ROOT / 'data/source_manifest.json', dict(entries=sorted(snap.entries, key=lambda x:x['url']), selected=selected,
         qa_schema=audits, model_revision=mrev, ibm_revision=ibm_rev,
         selection=dict(seed=CONFIG['seed'], kubernetes_pool=len(pool), runbook_pool=len(paths), incident_catalog_revision=drev,
                        rule='Kubernetes section x global size quartile round-robin; GitLab half common infrastructure + half random; incidents first two per chosen publisher in catalog order')))
    dump(ROOT / 'reports/download_errors.json', snap.errors)
    print('Fetched', len(records), 'documents;', len(qa), 'QA;', len(snap.errors), 'download errors', flush=True)


def restore():
    """Restore selected pinned source payloads from a saved manifest, never resample."""
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text(encoding='utf-8'))
    expected={e['url']:e for e in manifest['entries']}
    snap=Snapshot()
    def checked(url, source, revision, terms):
        body=snap.get(url,source,revision,terms)
        if body is None:
            return None
        if snap.entries[-1]['sha256']!=expected[url]['sha256']:
            raise ValueError(f'Publisher content changed: {url}. Retain previous snapshot; do not silently mix revisions.')
        return body
    records=[]
    for row in manifest['selected']:
        body=checked(row['url'],row['source_name'],row['revision'],row['terms'])
        if body is not None:
            records.append(dict(row,body=body))
    qa=[]
    for entry in manifest['entries']:
        if entry['source']=='techqa' and '/resolve/' in entry['url'] and entry['url'].endswith('.json'):
            body=checked(entry['url'],'techqa',entry['revision'],entry['terms'])
            if body:
                qa.extend(dict(row,source_split=Path(entry['url']).stem,source_revision=entry['revision']) for row in json.loads(body))
    if snap.errors:
        dump(ROOT/'reports/restore_errors.json',snap.errors)
        raise RuntimeError('Snapshot restore incomplete; see reports/restore_errors.json')
    qa=sorted(qa,key=lambda q:q['id'])
    if len(qa)>CONFIG['limits']['qa']:
        qa=random.Random(CONFIG['seed']).sample(qa,CONFIG['limits']['qa'])
    dump(ROOT/'data/parsed/source_records.json',records)
    dump(ROOT/'data/parsed/qa_original.json',qa)
    dump(ROOT/'reports/restore_log.json',snap.entries)
    print('Restored pinned snapshot:',len(records),'source records;',len(qa),'QA')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--new-snapshot',action='store_true',help='Explicitly resample from current upstream metadata (cached responses remain stable)')
    args=parser.parse_args()
    if (ROOT/'data/source_manifest.json').exists() and not args.new_snapshot:
        restore()
    else:
        acquire()
