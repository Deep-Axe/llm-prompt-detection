"""Bounded-memory readers and disk-backed tokens for the large experiment."""
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from prepare_large_corpus import file_hash

PARTITIONS = ('train', 'val', 'test', 'official_eval')
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batches(path, size):
    batch = []
    for r in rows(path):
        batch.append(r)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_manifest(data):
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['partitions']['train']['rows'] < 100000:
        raise ValueError('Fewer than 100,000 training prompts')
    for name in PARTITIONS:
        if file_hash(data / f'{name}.jsonl') != manifest['partitions'][name]['sha256']:
            raise ValueError(f'Partition checksum mismatch: {name}')
    return manifest


def word_tokens(text, limit):
    # finditer stops at the limit without materializing arbitrarily long prompts.
    result = []
    for m in TOKEN.finditer(text.lower()):
        result.append(m.group())
        if len(result) == limit:
            break
    return result or ['[UNK]']


def encode_corpus(data, cache, manifest, max_tokens=256, transformer=False, encoder=None):
    cache.mkdir(parents=True, exist_ok=True)
    if transformer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
        vocabulary = None
    else:
        counts = Counter()
        for r in rows(data / 'train.jsonl'):
            counts.update(word_tokens(r['text'], max_tokens))
        vocabulary = {t: i + 2 for i, (t, _) in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:20000])}
        (cache / 'vocabulary.json').write_text(json.dumps(vocabulary) + '\n')
    encoded = {}
    for split in PARTITIONS:
        n = manifest['partitions'][split]['rows']
        ids = np.lib.format.open_memmap(cache / f'{split}_ids.npy', mode='w+', dtype='int32', shape=(n, max_tokens))
        lengths = np.lib.format.open_memmap(cache / f'{split}_lengths.npy', mode='w+', dtype='int32', shape=(n,))
        labels = np.lib.format.open_memmap(cache / f'{split}_labels.npy', mode='w+', dtype='float32', shape=(n,))
        offset = 0
        for batch in batches(data / f'{split}.jsonl', 256):
            if transformer:
                result = tokenizer([r['text'] for r in batch], padding='max_length', truncation=True,
                                   max_length=max_tokens, return_tensors='np')
                ids[offset:offset+len(batch)] = result['input_ids']
                lengths[offset:offset+len(batch)] = result['attention_mask'].sum(1)
            else:
                ids[offset:offset+len(batch)] = 0
                for i, r in enumerate(batch):
                    tokens = word_tokens(r['text'], max_tokens)
                    values = [vocabulary.get(t, 1) for t in tokens]
                    ids[offset+i, :len(values)] = values
                    lengths[offset+i] = len(values)
            labels[offset:offset+len(batch)] = [r['label'] for r in batch]
            offset += len(batch)
        if offset != n:
            raise ValueError(f'Row count mismatch in {split}')
        ids.flush(); lengths.flush(); labels.flush()
        encoded[split] = (ids, lengths, labels)
        print(f'Encoded {split}: {n} rows', flush=True)
    return encoded, vocabulary
