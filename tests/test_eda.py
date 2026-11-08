import json
import unittest
import importlib.util
from pathlib import Path

from src.eda.loaders import ROOT
from src.eda.preprocessing import clean, prepare, terms
from src.eda.schema import stable_id, validate


class ResearchIntegrity(unittest.TestCase):
    def test_technical_symbols_survive(self):
        text='CVE-2017-3156\r\n--max-connections=100  v1.28.2\nCrashLoopBackOff /var/log/postgresql.log 503'
        normalized=clean(text)
        for token in ['CVE-2017-3156','--max-connections=100','v1.28.2','CrashLoopBackOff','/var/log/postgresql.log','503']:
            self.assertIn(token,normalized)
        self.assertIn('CVE-2017-3156',terms(normalized))

    def test_snapshot_schema_and_links(self):
        docs,qa=prepare()
        self.assertTrue(validate(docs,qa))
        self.assertEqual({d['source_type'] for d in docs},{'documentation','runbook','incident','qa_context'})
        self.assertEqual(len({q['query_id'] for q in qa}),len(qa))
        self.assertEqual(stable_id('incident','https://example.org/a'),stable_id('incident','https://example.org/a'))
        before=[(d['document_id'],d['content']) for d in docs]
        again,_=prepare()
        self.assertEqual(before,[(d['document_id'],d['content']) for d in again])

    def test_chunk_offsets_and_no_truncation(self):
        from src.eda.embeddings import load_model
        has_transformers=importlib.util.find_spec("sentence_transformers") is not None
        model=load_model() if has_transformers else None
        docs={d['document_id']:d for d in json.loads((ROOT/'data/processed/documents.json').read_text(encoding='utf-8'))}
        chunks=json.loads((ROOT/'data/processed/embedding_chunks.json').read_text(encoding='utf-8'))
        end={}
        for c in chunks:
            original=docs[c['document_id']]['content']
            self.assertEqual(c['content'],original[c['char_start']:c['char_end']])
            if model is not None:
                self.assertLessEqual(len(model.tokenizer(c['content'])['input_ids']),model.max_seq_length)
            self.assertLessEqual(c['token_start'],end.get(c['document_id'],0))
            end[c['document_id']]=c['token_end']
        for did,last in end.items():
            if model is not None:
                self.assertEqual(last,len(model.tokenizer(docs[did]['content'],add_special_tokens=False,truncation=False)['input_ids']))

    def test_raw_checksums(self):
        import hashlib
        manifest=json.loads((ROOT/'data/source_manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['entries']:
            self.assertEqual(hashlib.sha256((ROOT/Path(entry['raw_path'].replace('\\','/'))).read_bytes()).hexdigest(),entry['sha256'])


if __name__=='__main__':
    unittest.main()
