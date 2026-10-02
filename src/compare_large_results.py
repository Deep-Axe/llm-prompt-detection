"""Compare completed runs, checking identical training and evaluation partitions."""
import csv
import json
from pathlib import Path
from evaluation import ROOT


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
    lines = ['# 100,000-row model comparison', '',
             'WildJailbreak prompt harmfulness classification. All four models use the same 100,000 unique training prompts, 39,502 validation prompts, and 38,911 test prompts. The 2,210 official evaluation prompts form an additional check.', '',
             '| Model | Test accuracy | Test F1 | Test MCC | Official harmful recall | Official benign acceptance |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        lines.append(f'| {r["model"]} | {r["test_accuracy"]:.4f} | {r["test_f1"]:.4f} | {r["test_mcc"]:.4f} | {r["official_harmful_recall"]:.4f} | {r["official_benign_acceptance"]:.4f} |')
    lines += ['', 'Three epochs per architecture; checkpoint selection uses validation F1 only. The decision threshold is 0.5. CNN/BiLSTM use learned embeddings, and all DistilBERT encoder layers are updated. MLP uses hashed word/character TF-IDF to bound RAM; its IDF is fitted only on training data.', '',
              'The main partitions keep each underlying vanilla request and its adversarial variants together. Exact normalized duplicates and conflicting labels are removed. The official evaluation does not expose underlying vanilla requests, so only exact prompt overlap can be checked there. Grouping does not guarantee that different requests never use similar attack tactics.', '',
              'All models have a 256-token context budget, but regex and WordPiece tokenization differ. These are single-seed baseline runs on a largely synthetic corpus, with upstream labels retained. Harmfulness detection is not a complete test of context-dependent prompt injection. The official evaluation is mostly harmful, so benign acceptance must be read alongside F1.', '',
              'Hardware differs between local and Kaggle training; training times describe these runs and should not be treated as an architecture speed ranking. Weights, prompt data, and individual predictions remain outside Git. See the JSON reports for metrics, validation histories, code hashes, and partition hashes.']
    (folder / 'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
