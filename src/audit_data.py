"""Audit the imported corpus without changing its rows or splits."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from prepare_corpus import norm

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    data = ROOT / 'data/processed'
    rows = read(data / 'corpus.jsonl')
    hard = read(data / 'hard_negatives.jsonl')
    stats = json.loads((data / 'stats.json').read_text())
    hashes = {name: hashlib.sha256((ROOT / info['path']).read_bytes()).hexdigest() == info['sha256']
              for name, info in stats['files'].items()}
    assert all(hashes.values()), 'Raw files differ from the imported manifest'
    keys = [norm(r['text']) for r in rows + hard]
    assert len(keys) == len(set(keys)), 'Exact duplicate text found'
    assert len({r['id'] for r in rows + hard}) == len(rows + hard)
    train = [r for r in rows if r['split'] == 'train']
    held = [r for r in rows if r['split'] == 'heldout']
    assert not {r['family'] for r in held} & {r['family'] for r in rows if r['split'] != 'heldout'}
    lengths = {}
    for source in sorted({r['source'] for r in rows}):
        values = [len(r['text']) for r in rows if r['source'] == source]
        lengths[source] = {'n': len(values), 'median': float(np.median(values)),
                           'p90': float(np.percentile(values, 90)), 'max': max(values)}
    # Similarity is a diagnostic, not a semantic duplicate label or a split change.
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=20000, min_df=2)
    xtrain = vec.fit_transform([r['text'] for r in train])
    similarities = {}
    for split in ['val', 'test', 'heldout']:
        query = [r for r in rows if r['split'] == split and r['source'] == 'dan']
        maximum = linear_kernel(vec.transform([r['text'] for r in query]), xtrain).max(axis=1)
        similarities[split] = {'dan_rows': len(query), 'nearest_train_cosine_ge_0.9': int((maximum >= .9).sum()),
                               'nearest_train_cosine_ge_0.95': int((maximum >= .95).sum())}
    communities = {}
    for split in ['train', 'val', 'test', 'heldout']:
        communities[split] = {r['family'] for r in rows if r['split'] == split and r['source'] == 'dan' and r['family'] != 'dan:unknown'}
    report = {'raw_hashes_match': hashes, 'exact_duplicates': 0,
              'processed_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'corpus.jsonl', data/'hard_negatives.jsonl']},
              'counts': dict(Counter(f"{r['split']}|{r['label']}|{r['source']}" for r in rows)),
              'lengths': lengths, 'dan_similarity_to_training': similarities,
              'known_dan_communities_shared_train_test': len(communities['train'] & communities['test']),
              'alpaca_label_status': 'assumed benign; overlap checks are not safety review',
              'similarity_method': 'train-fitted character TF-IDF 3-5 grams, 20000 features, cosine; diagnostic only'}
    (ROOT/'results/data_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
