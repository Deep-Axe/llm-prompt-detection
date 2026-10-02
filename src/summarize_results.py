"""Export recorded model metrics and source slices, without retraining."""
import csv
import json
from evaluation import ROOT


def main():
    records=[]
    for name in ['mlp','cnn','bilstm','distilbert']:
        result=json.loads((ROOT/f'results/{name}.json').read_text())
        m=result['metrics']
        records.append({'model':name,'device':result.get('environment',{}).get('device','cpu'),
            'test_accuracy':m['test']['accuracy'],'test_precision':m['test']['precision'],
            'test_recall':m['test']['recall'],'test_f1':m['test']['f1'],'test_mcc':m['test']['mcc'],
            'test_roc_auc':m['test']['roc_auc'],'test_pr_auc':m['test']['pr_auc'],
            'test_average_precision':m['test']['average_precision'],
            'test_log_loss':m['test']['log_loss'],'test_brier':m['test']['brier_score'],
            'jbb_test_recall':m['test_jbb_harmful']['recall'],'dan_test_recall':m['test_dan']['recall'],
            'heldout_recall':m['heldout']['recall'],'benign_acceptance':m['hard_negative']['benign_acceptance'],
            'total_parameters':result.get('total_parameters',result.get('total_trainable_parameters')),
            'trainable_parameters':result.get('trainable_parameters',result.get('total_trainable_parameters')),
            'feature_seconds':result['feature_fit_seconds'],'training_seconds':result['training_seconds'],
            'batch_ms_per_example':result['batch_ms_per_example_including_features'],
            'single_request_ms':result['single_request_latency_ms_median']})
    hashes=[json.loads((ROOT/f'results/{name}.json').read_text())['data_hashes'] for name in ['mlp','cnn','bilstm','distilbert']]
    if not all(h==hashes[0] for h in hashes): raise ValueError('Models used different datasets')
    output=ROOT/'results/comparison.csv'
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    (ROOT/'results/comparison.json').write_text(json.dumps({'data_hashes':hashes[0],'models':records},indent=2)+'\n')
    lines=['# Recorded preliminary results','',
        'Fixed seed 4442, unchanged corpus partitions, and probability threshold 0.5. No hyperparameter search was performed.', '',
        '| Model | Test F1 | JBB test recall | Held-out recall | Matched benign acceptance |',
        '| --- | ---: | ---: | ---: | ---: |']
    baseline=json.loads((ROOT/'data/processed/length_baseline.json').read_text())['splits']
    lines.append(f"| Length rule | {baseline['test']['f1']:.3f} | {baseline['test_jbb_harmful']['recall']:.3f} | {baseline['heldout']['recall']:.3f} | {baseline['hard_negative']['accuracy']:.3f} |")
    for r in records:
        lines.append(f"| {r['model']} | {r['test_f1']:.3f} | {r['jbb_test_recall']:.3f} | {r['heldout_recall']:.3f} | {r['benign_acceptance']:.3f} |")
    lines += ['', 'The 15 positive JBB test requests form a small diagnostic slice. The held-out slice has only positive examples, so it measures recall rather than overall classification quality. Alpaca is assumed benign; label review has not been completed. The length and template-similarity shortcuts remain.', '',
        'The MLP uses full text and runs on CPU. CNN and BiLSTM use the first 128 regex tokens; DistilBERT uses 128 WordPiece tokens including special tokens, with only its linear head trained. Token budgets have the same numerical cap but different tokenization. CUDA timings are synchronized. Batch timings include text processing and, for DistilBERT, the encoder; cached embeddings are not used for latency measurement.', '',
        'MLP and GPU-model timings are measured on different processors and cannot establish an intrinsic architecture speed ranking. JSON files retain full metrics, parameter counts, training/encoding times, GPU peak allocation, error identifiers, and corpus hashes. Checkpoints and downloaded weights remain local in ignored directories.', '',
        'These are initial single-seed, fixed-budget results. They do not establish tuned performance or robustness to real indirect prompt injection.']
    (ROOT/'results/README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:13]))


if __name__=='__main__': main()
