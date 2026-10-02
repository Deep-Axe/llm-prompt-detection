"""Stream WildJailbreak into deduplicated, request-grouped partitions.

Download the pinned upstream files with hf before running this script.
Labels describe prompt harmfulness, not whether a model's answer is safe.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REVISION = '5ddc12a7894f842b0619b8e1c7ee496b198af009'
LABELS = {'vanilla_benign': 0, 'adversarial_benign': 0,
          'vanilla_harmful': 1, 'adversarial_harmful': 1}


def digest(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_for_group(group, seed=4442):
    # Related adversarial versions and their vanilla request always travel together.
    fraction = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if fraction < .70 else ('val' if fraction < .85 else 'test')


def upstream_rows(path):
    csv.field_size_limit(100_000_000)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {'adversarial', 'data_type'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Unexpected upstream schema: {reader.fieldnames}')
        for r in reader:
            kind = r['data_type']
            if kind not in LABELS:
                raise ValueError(f'Unknown data_type {kind!r}')
            text = r['adversarial'] if kind.startswith('adversarial') else r['vanilla']
            if not text.strip():
                continue
            yield {'id': digest(text), 'group': digest(r.get('vanilla') or text),
                   'text': text, 'label': LABELS[kind], 'family': kind,
                   'source': 'wildjailbreak'}


def prepare(raw, output, train_rows=100000, seed=4442):
    if train_rows < 100000:
        raise ValueError('The large comparison requires at least 100,000 training rows')
    output.mkdir(parents=True, exist_ok=True)
    dbpath = output / 'dedup.sqlite'
    dbpath.unlink(missing_ok=True)
    db = sqlite3.connect(dbpath)
    db.executescript('''PRAGMA cache_size=-16000; PRAGMA temp_store=FILE;
        CREATE TABLE rows (id TEXT PRIMARY KEY, grp TEXT, payload TEXT, label INTEGER,
                           split TEXT, conflict INTEGER DEFAULT 0);
        CREATE TABLE excluded_groups (grp TEXT PRIMARY KEY);''')
    counts = Counter()
    # Import official evaluation first: it takes precedence over training variants.
    for upstream, split in [('eval/eval.tsv', 'official_eval'), ('train/train.tsv', None)]:
        for r in upstream_rows(raw / upstream):
            counts[f'raw_{upstream.split("/")[0]}'] += 1
            if split:
                db.execute('INSERT OR IGNORE INTO excluded_groups VALUES (?)', (r['group'],))
            old = db.execute('SELECT label, grp FROM rows WHERE id=?', (r['id'],)).fetchone()
            if old:
                counts['duplicate_texts'] += 1
                if old[0] != r['label']:
                    db.execute('UPDATE rows SET conflict=1 WHERE id=?', (r['id'],))
                    counts['conflicting_labels'] += 1
                # If the same text connects two requests, quarantine the later group.
                if old[1] != r['group']:
                    db.execute('INSERT OR IGNORE INTO excluded_groups VALUES (?)', (r['group'],))
                    counts['duplicate_bridge_groups'] += 1
                continue
            assigned = split or split_for_group(r['group'], seed)
            db.execute('INSERT INTO rows(id,grp,payload,label,split) VALUES (?,?,?,?,?)',
                       (r['id'], r['group'], json.dumps(r, ensure_ascii=False), r['label'], assigned))
            if counts[f'raw_{upstream.split("/")[0]}'] % 10000 == 0:
                db.commit()
        db.commit()
    counts['excluded_train_variants'] = db.execute('''SELECT COUNT(*) FROM rows
        WHERE split!='official_eval' AND grp IN (SELECT grp FROM excluded_groups)''').fetchone()[0]
    db.execute("DELETE FROM rows WHERE conflict=1 OR (split!='official_eval' AND grp IN (SELECT grp FROM excluded_groups))")
    db.commit()
    available = db.execute("SELECT COUNT(*) FROM rows WHERE split='train'").fetchone()[0]
    if available < train_rows:
        raise ValueError(f'Only {available} unique, non-overlapping training prompts available')
    manifest = {'dataset': 'allenai/wildjailbreak', 'revision': REVISION, 'license': 'ODC-BY',
                'task': 'binary prompt harmfulness', 'seed': seed,
                'label_mapping': LABELS, 'raw_counts': dict(counts),
                'official_eval_limit': 'Upstream evaluation omits vanilla requests; exact text overlap removed, underlying-request overlap cannot be checked.',
                'train_pool_before_cap': available, 'split_method': '70/15/15 hash of normalized vanilla request; train capped by text hash',
                'raw_file_sha256': {p: file_hash(raw / p) for p in ['train/train.tsv', 'eval/eval.tsv']},
                'partitions': {}}
    for split in ['train', 'val', 'test', 'official_eval']:
        query = 'SELECT payload FROM rows WHERE split=? ORDER BY id'
        params = (split,)
        if split == 'train':
            query += ' LIMIT ?'
            params += (train_rows,)
        totals = Counter()
        groups = set()
        path = output / f'{split}.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for (payload,) in db.execute(query, params):
                r = json.loads(payload)
                r['split'] = split
                totals[f'label_{r["label"]}'] += 1
                totals[r['family']] += 1
                groups.add(r['group'])
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        manifest['partitions'][split] = {'rows': totals['label_0'] + totals['label_1'],
                                       'groups': len(groups), 'counts': dict(totals), 'sha256': file_hash(path)}
    db.close()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    public = ROOT / 'results/large'
    public.mkdir(parents=True, exist_ok=True)
    (public / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT / 'data/raw/large/wildjailbreak')
    p.add_argument('--output', type=Path, default=ROOT / 'data/processed/large')
    p.add_argument('--train-rows', type=int, default=100000)
    args = p.parse_args()
    prepare(args.raw, args.output, args.train_rows)
