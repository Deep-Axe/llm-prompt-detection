"""Evaluate validation-selected classifiers on a frozen external prompt dataset."""
import argparse
import csv
import gc
import json
from pathlib import Path
import numpy as np
from evaluation import ROOT, metrics
from experiment_state import atomic_json, digest_json
from prepare_large_corpus import file_hash
from predict_large_models import PromptClassifier, MODEL_FILES


def evaluate(protocol_path, device='cpu', conditions=None, models=None):
    protocol=json.loads(protocol_path.read_text())
    import torch
    torch.set_num_threads(1)
    if protocol['threshold']!=0.5 or protocol['fit_on_external_data']:
        raise ValueError('External evaluation cannot change the frozen selection policy')
    data=ROOT / protocol['data_directory'] / 'test.jsonl'
    expected=protocol['manifest']['partitions']['test']
    if file_hash(data)!=expected['sha256']: raise ValueError('External evaluation data changed')
    with data.open() as handle:
        records=[json.loads(line) for line in handle]
    if len(records)!=expected['rows']: raise ValueError('External evaluation row count changed')
    for condition in conditions or protocol['training_conditions']:
        for model in models or protocol['models']:
            for seed in protocol['seeds']:
                relative=Path(condition) / model / f'seed_{seed}'
                training_path=ROOT / 'results/experiments' / relative / 'result.json'
                if not training_path.exists(): continue
                training=json.loads(training_path.read_text())
                if training['status']!='complete' or training['smoke_only'] or training['threshold']!=0.5:
                    raise ValueError('Checkpoint is not an eligible completed training run')
                # The fixed grouped validation partition alone selected this checkpoint.
                artifact=ROOT / 'artifacts/experiments' / relative
                folder=artifact / 'model' if model=='distilbert' else artifact
                checkpoint_hashes={name:file_hash(folder/name) for name in MODEL_FILES[model]}
                provenance={'protocol_sha256':file_hash(protocol_path),
                    'evaluation_sha256':expected['sha256'],'training_result_sha256':file_hash(training_path),
                    'checkpoint_sha256':checkpoint_hashes,'inference_precision':protocol['inference_precision'],
                    'inference_code_sha256':file_hash(ROOT/'src/predict_large_models.py'),
                    'evaluation_code_sha256':file_hash(Path(__file__)),
                    'metrics_code_sha256':file_hash(ROOT/'src/evaluation.py')}
                fingerprint=digest_json(provenance)
                output=ROOT / 'results/transfer' / protocol['condition'] / relative
                destination=output/'result.json'
                if destination.exists():
                    if json.loads(destination.read_text())['fingerprint']!=fingerprint:
                        raise ValueError('Existing transfer result has different provenance')
                    continue
                classifier=PromptClassifier(folder,model,'cpu' if model=='mlp' else device)
                probabilities=[]
                for start in range(0,len(records),32):
                    probabilities.extend(classifier.probabilities([r['text'] for r in records[start:start+32]]))
                probabilities=np.asarray(probabilities,dtype=np.float64)
                scores={'test':metrics(records,probabilities)}
                for family in sorted({r['family'] for r in records}):
                    indices=[i for i,r in enumerate(records) if r['family']==family]
                    scores[family]=metrics([records[i] for i in indices],probabilities[indices])
                prediction_dir=ROOT/'artifacts/transfer'/protocol['condition']/relative
                prediction_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(prediction_dir/'test_predictions.npz',ids=[r['id'] for r in records],
                                    labels=[r['label'] for r in records],probabilities=probabilities)
                output.mkdir(parents=True,exist_ok=True)
                atomic_json(destination,{'status':'complete','evaluation_condition':protocol['condition'],
                    'training_condition':condition,'model':model,'seed':seed,'chosen_epoch':training['chosen_epoch'],
                    'threshold':0.5,'external_training_rows':0,'n':len(records),
                    'fingerprint':fingerprint,'provenance':provenance,'metrics':scores})
                print(f'External evaluation complete: {condition} {model} seed={seed} F1={scores["test"]["f1"]:.6f}',flush=True)
                del classifier; gc.collect()
                if device.startswith('cuda'):
                    import torch
                    torch.cuda.empty_cache()
    return summarize(protocol)


def summarize(protocol):
    folder=ROOT/'results/transfer'/protocol['condition']
    folder.mkdir(parents=True,exist_ok=True)
    reports=[json.loads(p.read_text()) for p in folder.rglob('result.json')]
    summary=[]
    for condition in protocol['training_conditions']:
        for model in protocol['models']:
            runs=[r for r in reports if r['training_condition']==condition and r['model']==model]
            if not runs: continue
            seeds=sorted(r['seed'] for r in runs)
            if len(seeds)!=len(set(seeds)): raise ValueError('Duplicate seed')
            row={'training_condition':condition,'model':model,'n':protocol['manifest']['partitions']['test']['rows'],
                 'seeds':seeds,'complete_seed_set':seeds==protocol['seeds']}
            for metric in ['accuracy','precision','recall','benign_acceptance','f1','mcc','roc_auc']:
                values=np.array([r['metrics']['test'][metric] for r in runs])
                row[metric+'_mean']=float(values.mean())
                row[metric+'_std']=float(values.std(ddof=1)) if len(values)>1 else None
            summary.append(row)
    atomic_json(folder/'summary.json',{'evaluation_condition':protocol['condition'],'rows':summary})
    if summary:
        with (folder/'summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/transfer_evaluation.json')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--conditions',nargs='+')
    parser.add_argument('--models',nargs='+')
    args=parser.parse_args()
    evaluate(args.protocol,args.device,args.conditions,args.models)
