"""Compare completed runs, checking identical training and evaluation partitions."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluation import ROOT
from large_data import rows


def grouped_bootstrap(names, repeats=1000, seed=4442):
    test = [{k:r[k] for k in ('id','group','label')} for r in rows(ROOT / 'data/processed/large/test.jsonl')]
    identifiers = np.array([r['id'] for r in test])
    labels = np.array([r['label'] for r in test])
    _, groups = np.unique([r['group'] for r in test], return_inverse=True)
    ngroups = groups.max()+1
    group_counts = []
    for name in names:
        with np.load(ROOT / f'artifacts/large/{name}/test_predictions.npz') as saved:
            if not np.array_equal(saved['ids'], identifiers) or not np.array_equal(saved['labels'], labels):
                raise ValueError(f'Prediction IDs/labels do not match test partition: {name}')
            pred = saved['probabilities'] >= .5
        group_counts.append(np.stack([np.bincount(groups, weights=mask, minlength=ngroups)
                                      for mask in [(labels==1)&pred, (labels==0)&pred, (labels==1)&~pred]], axis=1))
    counts = np.stack(group_counts)
    rng = np.random.default_rng(seed); samples = np.empty((repeats,len(names)))
    for i in range(repeats):
        ix = rng.integers(0, ngroups, size=ngroups)
        sums = counts[:,ix,:].sum(axis=1)
        samples[i] = 2*sums[:,0] / np.maximum(1, 2*sums[:,0]+sums[:,1]+sums[:,2])
    summary = {'method':'paired percentile bootstrap over underlying test request groups',
               'repeats':repeats, 'seed':seed, 'test_groups':int(ngroups),
               'scope':'test sampling uncertainty only; excludes training-seed variation and label error',
               'f1_95_percent_intervals':{}, 'paired_f1_difference_95_percent_intervals':{}}
    for i,name in enumerate(names):
        summary['f1_95_percent_intervals'][name] = np.quantile(samples[:,i], [.025,.975]).tolist()
        for j in range(i):
            summary['paired_f1_difference_95_percent_intervals'][f'{name} minus {names[j]}'] = np.quantile(samples[:,i]-samples[:,j], [.025,.975]).tolist()
    return summary


def main():
    folder = ROOT / 'results/large'
    reports = {name:json.loads((folder / f'{name}.json').read_text())
               for name in ['mlp','cnn','bilstm','distilbert']}
    reference = reports['mlp']['data_hashes']
    for name, r in reports.items():
        if r['data_hashes'] != reference or r['train_rows'] < 100000:
            raise ValueError(f'Incompatible experiment: {name}')
    d = reports['distilbert']
    if d.get('encoder_frozen') is not False or not d.get('encoder_gradient_verified') or not d.get('encoder_weight_changed'):
        raise ValueError('DistilBERT encoder fine-tuning was not verified')
    fields = ['model','train_rows','test_accuracy','test_precision','test_recall','test_f1','test_mcc',
              'test_roc_auc','test_average_precision','test_brier_score','official_eval_f1',
              'official_harmful_recall','official_benign_acceptance','chosen_epoch','training_and_validation_seconds']
    table = []
    for name,r in reports.items():
        row = {'model':name,'train_rows':r['train_rows'], 'chosen_epoch':r['chosen_epoch'],
               'training_and_validation_seconds':r['training_and_validation_seconds'],
               'official_eval_f1':r['metrics']['official_eval']['f1'],
               'official_harmful_recall':r['metrics']['official_eval/adversarial_harmful']['recall'],
               'official_benign_acceptance':r['metrics']['official_eval/adversarial_benign']['benign_acceptance']}
        row.update({f'test_{k}':r['metrics']['test'][k] for k in ['accuracy','precision','recall','f1','mcc','roc_auc','average_precision','brier_score']})
        table.append(row)
    with (folder / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); writer.writeheader(); writer.writerows(table)
    (folder / 'comparison.json').write_text(json.dumps(table, indent=2)+'\n')
    uncertainty = grouped_bootstrap(list(reports))
    (folder / 'uncertainty.json').write_text(json.dumps(uncertainty, indent=2)+'\n')
    lines = ['# 100,000-row model comparison', '',
             'WildJailbreak prompt harmfulness classification. All four models use the same 100,000 unique training prompts, 39,502 validation prompts, and 38,911 test prompts. The 2,210 official evaluation prompts form an additional check.', '',
             '| Model | Test accuracy | Test F1 | Test MCC | Official harmful recall | Official benign acceptance |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        lines.append(f'| {r["model"]} | {r["test_accuracy"]:.4f} | {r["test_f1"]:.4f} | {r["test_mcc"]:.4f} | {r["official_harmful_recall"]:.4f} | {r["official_benign_acceptance"]:.4f} |')
    lines += ['', 'Three epochs per architecture; checkpoint selection uses validation F1 only. The decision threshold is 0.5. CNN/BiLSTM use learned embeddings, and all DistilBERT encoder layers are updated. MLP uses hashed word/character TF-IDF to bound RAM; its IDF is fitted only on training data.', '',
              'The main partitions keep each underlying vanilla request and its adversarial variants together. Exact normalized duplicates and conflicting labels are removed. The official evaluation does not expose underlying vanilla requests, so only exact prompt overlap can be checked there. Grouping does not guarantee that different requests never use similar attack tactics.', '',
              'All models have a 256-token context budget, but regex and WordPiece tokenization differ. These are single-seed baseline runs on a largely synthetic corpus, with upstream labels retained. Harmfulness detection is not a complete test of context-dependent prompt injection. The official evaluation is mostly harmful, so benign acceptance must be read alongside F1.', '',
              'The accompanying uncertainty.json gives paired 95% bootstrap intervals over underlying test request groups (1,000 resamples). These measure test sampling uncertainty, not training-seed variation or annotation quality.', '',
              'Hardware differs between local and Kaggle training; training times describe these runs and should not be treated as an architecture speed ranking. Weights, prompt data, and individual predictions remain outside Git. See the JSON reports for metrics, validation histories, code hashes, and partition hashes.']
    (folder / 'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
