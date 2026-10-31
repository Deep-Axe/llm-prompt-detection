"""Verify portable CNN checkpoints and evaluate transfer on fixed prompt labels."""
import argparse
from itertools import islice
import json
from pathlib import Path
import numpy as np
import torch
from evaluation import ROOT, metrics
from experiment_state import atomic_json
from large_data import PARTITIONS, rows, validate_manifest
from prepare_large_corpus import file_hash
from predict_cnn import CNNClassifier


def evaluate(device='cuda'):
    torch.set_num_threads(1)
    data=ROOT/'data/processed/large';manifest=validate_manifest(data)
    metadata={s:[{k:r[k] for k in ['id','label','family','group']} for r in rows(data/f'{s}.jsonl')] for s in PARTITIONS}
    samples={s:list(islice(rows(data/f'{s}.jsonl'),128)) for s in PARTITIONS}
    transfer=json.loads((ROOT/'configs/transfer_evaluation.json').read_text())
    external=ROOT/transfer['data_directory']/'test.jsonl'
    if file_hash(external)!=transfer['manifest']['partitions']['test']['sha256']: raise ValueError('Transfer prompts changed')
    external_rows=list(rows(external))
    for length in [256,512]:
        condition=f'wildjailbreak-100k-cnn-wide-{length}'
        for seed in [4442,4443,4444]:
            relative=Path(condition)/'cnn'/f'seed_{seed}'
            result_dir=ROOT/'results/cnn_extensions'/relative
            report_path=result_dir/'result.json'
            if not report_path.exists(): continue
            report=json.loads(report_path.read_text())
            if report['status']!='complete' or report['smoke_only'] or report['train_rows']!=100000 or report['threshold']!=.5:
                raise ValueError('Expected a complete 100000-row CNN experiment')
            artifact=ROOT/'artifacts/cnn_extensions'/relative
            classifier=CNNClassifier(artifact,device)
            differences={}
            for split in PARTITIONS:
                expected=metadata[split]
                with np.load(artifact/f'{split}_predictions.npz') as saved:
                    if not np.array_equal(saved['ids'],[r['id'] for r in expected]) or not np.array_equal(saved['labels'],[r['label'] for r in expected]):
                        raise ValueError('Saved prediction IDs or labels differ')
                    recomputed=metrics(expected,saved['probabilities'])
                    for name,value in recomputed.items():
                        recorded=report['metrics'][split][name]
                        if value is None:
                            if recorded is not None: raise ValueError('Metric availability differs')
                        elif not np.isclose(value,recorded,atol=1e-8,rtol=0): raise ValueError('Saved metrics differ from predictions')
                    values=[]
                    for start in range(0,len(samples[split]),64):
                        values.extend(classifier.probabilities([r['text'] for r in samples[split][start:start+64]]))
                    difference=float(np.max(np.abs(np.asarray(values)-saved['probabilities'][:len(values)])))
                    if difference>1e-6: raise ValueError(f'Portable inference mismatch: {condition}/{seed}/{split}/{difference}')
                    differences[split]=difference
            atomic_json(result_dir/'inference_verification.json',{'status':'verified','samples_per_partition':128,
                'all_saved_metrics_recomputed':True,'max_probability_differences':differences,
                'model_config_sha256':file_hash(artifact/'model_config.json'),
                'checkpoint_sha256':file_hash(artifact/'best.pt'),'vocabulary_sha256':file_hash(artifact/'vocabulary.json'),
                'inference_code_sha256':file_hash(ROOT/'src/predict_cnn.py')})
            probabilities=[]
            for start in range(0,len(external_rows),64):
                probabilities.extend(classifier.probabilities([r['text'] for r in external_rows[start:start+64]]))
            probabilities=np.asarray(probabilities)
            scores={'test':metrics(external_rows,probabilities)}
            for family in sorted({r['family'] for r in external_rows}):
                selected=[i for i,r in enumerate(external_rows) if r['family']==family]
                scores[family]=metrics([external_rows[i] for i in selected],probabilities[selected])
            private=ROOT/'artifacts/cnn_extensions_transfer'/condition/f'seed_{seed}'
            private.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(private/'test_predictions.npz',ids=[r['id'] for r in external_rows],
                labels=[r['label'] for r in external_rows],probabilities=probabilities)
            output=ROOT/'results/cnn_extensions/transfer'/condition/f'seed_{seed}'
            output.mkdir(parents=True,exist_ok=True)
            atomic_json(output/'result.json',{'status':'complete','kind':'transfer_evaluation','condition':condition,
                'seed':seed,'model':'cnn','max_tokens':length,'threshold':.5,'external_training_rows':0,
                'n':len(external_rows),'metrics':scores,'training_result_sha256':file_hash(report_path),
                'evaluation_data_sha256':file_hash(external),'inference_code_sha256':file_hash(ROOT/'src/predict_cnn.py'),
                'evaluation_code_sha256':file_hash(Path(__file__)),'checkpoint_sha256':file_hash(artifact/'best.pt')})
            print(f'Verified and evaluated {condition} seed={seed}',flush=True)
            del classifier
            if device.startswith('cuda'): torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda')
    evaluate(parser.parse_args().device)
