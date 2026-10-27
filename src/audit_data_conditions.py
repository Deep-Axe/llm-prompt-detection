"""Audit nested WildJailbreak training pools and WildGuardMix prompt labels.

The frozen 100,000-row partitions are not modified. Prompt text is not written
to the result file. WildGuardMix is assessed as a separate source.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from prepare_large_corpus import (
    LABELS, REVISION, ROOT, digest, file_hash, split_for_group, upstream_rows,
)

POOLS = (100_000, 150_000, 183_005)


def id_digest(ids):
    h = hashlib.sha256()
    for item in ids:
        h.update(item.encode())
        h.update(b'\n')
    return h.hexdigest()


def load_ids(path):
    ids = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)['id'])
    return ids


def build_rows(raw):
    db = sqlite3.connect(':memory:')
    db.executescript('''
        CREATE TABLE rows (id TEXT PRIMARY KEY, grp TEXT, label INTEGER,
                           family TEXT, split TEXT, conflict INTEGER DEFAULT 0);
        CREATE TABLE excluded_groups (grp TEXT PRIMARY KEY);''')
    counts = Counter()
    for upstream, fixed in [('eval/eval.tsv', 'official_eval'), ('train/train.tsv', None)]:
        for row in upstream_rows(raw / upstream):
            counts[f'raw_{upstream.split("/")[0]}'] += 1
            if fixed:
                db.execute('INSERT OR IGNORE INTO excluded_groups VALUES (?)', (row['group'],))
            old = db.execute('SELECT label, grp FROM rows WHERE id=?', (row['id'],)).fetchone()
            if old:
                counts['duplicate_texts'] += 1
                if old[0] != row['label']:
                    db.execute('UPDATE rows SET conflict=1 WHERE id=?', (row['id'],))
                    counts['conflicting_labels'] += 1
                if old[1] != row['group']:
                    db.execute('INSERT OR IGNORE INTO excluded_groups VALUES (?)', (row['group'],))
                    counts['duplicate_bridge_groups'] += 1
                continue
            assigned = fixed or split_for_group(row['group'], 4442)
            db.execute('INSERT INTO rows(id,grp,label,family,split) VALUES (?,?,?,?,?)',
                       (row['id'], row['group'], row['label'], row['family'], assigned))
    counts['excluded_train_variants'] = db.execute('''SELECT COUNT(*) FROM rows
        WHERE split!='official_eval' AND grp IN (SELECT grp FROM excluded_groups)''').fetchone()[0]
    db.execute("DELETE FROM rows WHERE conflict=1 OR (split!='official_eval' AND grp IN (SELECT grp FROM excluded_groups))")
    db.commit()
    return db, counts


def pool_record(rows, groups_outside):
    labels = Counter(row[1] for row in rows)
    families = Counter(row[2] for row in rows)
    groups = {row[3] for row in rows}
    return {
        'rows': len(rows),
        'groups': len(groups),
        'benign': labels[0],
        'harmful': labels[1],
        'families': dict(families),
        'id_sha256': id_digest(row[0] for row in rows),
        'request_overlap_with_val_or_test': len(groups & groups_outside),
    }


def audit_wildjailbreak(db, counts, processed):
    train = db.execute("SELECT id, label, family, grp FROM rows WHERE split='train' ORDER BY id").fetchall()
    outside = {row[0] for row in db.execute("SELECT grp FROM rows WHERE split IN ('val','test')")}
    existing = load_ids(processed / 'train.jsonl')
    if [row[0] for row in train[:100_000]] != existing:
        raise SystemExit('The 100,000-row prefix does not match the frozen training partition')
    if len(train) < 183_005:
        raise SystemExit(f'Eligible training pool has {len(train)} rows, expected at least 183,005')
    pools = {}
    for size in POOLS:
        if size > len(train):
            raise SystemExit(f'Cannot form a {size}-row pool from {len(train)} eligible rows')
        chosen = train[:size]
        record = pool_record(chosen, outside)
        record['contains_frozen_100k'] = [row[0] for row in chosen[:100_000]] == existing
        pools[str(size)] = record
    held = {}
    for split in ('val', 'test', 'official_eval'):
        rows = db.execute('SELECT id, label, family, grp FROM rows WHERE split=? ORDER BY id', (split,)).fetchall()
        held[split] = {'rows': len(rows), 'sha256': file_hash(processed / f'{split}.jsonl')}
    return {
        'dataset': 'allenai/wildjailbreak',
        'revision': REVISION,
        'rule': 'Eligible train rows ordered by normalized-text id. Each larger pool is a prefix of that order and contains the frozen 100,000-row set. Validation and test assignments are unchanged.',
        'label_mapping': LABELS,
        'raw_counts': dict(counts),
        'eligible_train_rows': len(train),
        'pools': pools,
        'unchanged_partitions': held,
        'trained': False,
    }


def audit_wildguard(path, ids_by_split):
    import pyarrow.parquet as pq
    summary = {'dataset': 'allenai/wildguardmix', 'task': 'prompt harmfulness only', 'splits': {}}
    for split, filename in (('train', 'train/wildguard_train.parquet'), ('test', 'test/wildguard_test.parquet')):
        table = pq.read_table(path / filename, columns=['prompt', 'prompt_harm_label', 'adversarial'])
        prompts = table.column('prompt').to_pylist()
        labels = table.column('prompt_harm_label').to_pylist()
        adversarial = table.column('adversarial').to_pylist()
        usable = {}
        blocked = set()
        missing = conflicts = duplicate_rows = 0
        label_counts = Counter()
        adversarial_counts = Counter()
        for prompt, label, flag in zip(prompts, labels, adversarial):
            if label not in {'harmful', 'unharmful'}:
                missing += 1
                continue
            key = digest(prompt or '')
            mapped = 1 if label == 'harmful' else 0
            label_counts[label] += 1
            adversarial_counts[str(bool(flag))] += 1
            if key in blocked:
                continue
            previous = usable.get(key)
            if previous is None:
                usable[key] = mapped
            elif previous != mapped:
                conflicts += 1
                blocked.add(key)
                del usable[key]
            else:
                duplicate_rows += 1
        overlap = {name: sum(1 for key in usable if key in ids) for name, ids in ids_by_split.items()}
        summary['splits'][split] = {
            'rows': len(prompts),
            'missing_prompt_harm_label': missing,
            'duplicate_rows_same_label': duplicate_rows,
            'label_conflicts_removed': conflicts,
            'unique_labeled_prompts': len(usable),
            'harmful': sum(usable.values()),
            'unharmful': len(usable) - sum(usable.values()),
            'raw_label_counts': dict(label_counts),
            'adversarial_flag_counts': dict(adversarial_counts),
            'exact_overlap_with_wildjailbreak': overlap,
        }
    summary['use'] = 'Separate from the WildJailbreak training set. The WildGuard test split stays out of training.'
    summary['trained'] = False
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw/large/wildjailbreak')
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/processed/large')
    parser.add_argument('--wildguard', type=Path, default=ROOT / 'data/raw/large/wildguardmix')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/data_conditions.json')
    args = parser.parse_args()
    db, counts = build_rows(args.raw)
    wildjailbreak = audit_wildjailbreak(db, counts, args.processed)
    ids_by_split = {
        'eligible_train': {row[0] for row in db.execute("SELECT id FROM rows WHERE split='train'")},
        'train_100k': set(load_ids(args.processed / 'train.jsonl')),
        'val': {row[0] for row in db.execute("SELECT id FROM rows WHERE split='val'")},
        'test': {row[0] for row in db.execute("SELECT id FROM rows WHERE split='test'")},
        'official_eval': {row[0] for row in db.execute("SELECT id FROM rows WHERE split='official_eval'")},
    }
    report = {'wildjailbreak_pools': wildjailbreak}
    if (args.wildguard / 'train/wildguard_train.parquet').exists():
        report['wildguardmix'] = audit_wildguard(args.wildguard, ids_by_split)
    else:
        report['wildguardmix'] = {'status': 'not downloaded'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'eligible': wildjailbreak['eligible_train_rows'],
        'pools': {k: v['rows'] for k, v in wildjailbreak['pools'].items()},
        'nested': all(v['contains_frozen_100k'] for v in wildjailbreak['pools'].values()),
        'wildguard': report['wildguardmix'].get('splits', report['wildguardmix'].get('status')),
    }, indent=2))


if __name__ == '__main__':
    main()
