"""Check partition leakage and measure a length-only shortcut on the large corpus."""
import json
from pathlib import Path
import numpy as np
from evaluation import ROOT, metrics
from large_data import PARTITIONS, rows, validate_manifest


def main():
    data = ROOT / 'data/processed/large'
    manifest = validate_manifest(data)
    seen = set(); seen_groups = set(); lengths = {}; labels = {}; summaries = {}
    for split in PARTITIONS:
        group = set(); size = []; gold = []; families = {}
        for r in rows(data / f'{split}.jsonl'):
            if r['id'] in seen:
                raise ValueError('Exact prompt leakage')
            seen.add(r['id']); group.add(r['group'])
            size.append(len(r['text'])); gold.append(r['label'])
            families.setdefault(r['family'], []).append(len(r['text']))
        if split != 'official_eval':
            if group & seen_groups:
                raise ValueError('Underlying request leakage')
            seen_groups.update(group)
        lengths[split] = np.array(size); labels[split] = np.array(gold)
        summaries[split] = {kind: {'n':len(values), 'median_characters':float(np.median(values)),
                                 'p95_characters':float(np.percentile(values,95))}
                            for kind,values in families.items()}
    # Fit only on training labels; search both directions of a length threshold.
    x, y = lengths['train'], labels['train']
    order = np.argsort(x, kind='stable'); sx = x[order]; sy = y[order]
    boundary = np.r_[0, np.flatnonzero(np.diff(sx))+1, len(sx)]
    cumulative = np.r_[0,np.cumsum(sy)]; positives = y.sum()
    best = (-1., None, None)
    for direction in ['short','long']:
        tp = cumulative[boundary] if direction=='short' else positives-cumulative[boundary]
        predicted = boundary if direction=='short' else len(sx)-boundary
        f1 = 2*tp / np.maximum(1, positives+predicted)
        at = int(np.argmax(f1)); index = int(boundary[at])
        threshold = int(sx[index]) if index<len(sx) else int(sx[-1]+1)
        if f1[at] > best[0]:
            best = (float(f1[at]), threshold, direction)
    scores = {}
    for split in PARTITIONS:
        pred = lengths[split] < best[1] if best[2]=='short' else lengths[split] >= best[1]
        scores[split] = metrics([{'label':int(y)} for y in labels[split]], pred.astype(float))
    result = {'data_hashes':{s:manifest['partitions'][s]['sha256'] for s in PARTITIONS},
              'no_exact_prompt_overlap':True, 'main_splits_have_no_underlying_request_overlap':True,
              'official_eval_request_overlap':'unknown; upstream omits vanilla request',
              'length_summaries':summaries,
              'length_only_baseline':{'threshold_characters':best[1], 'harmful_direction':best[2], 'metrics':scores}}
    (ROOT / 'results/large/data_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'rows':{s:len(labels[s]) for s in PARTITIONS}, 'length_only_test_f1':scores['test']['f1']}, indent=2))


if __name__ == '__main__':
    main()
