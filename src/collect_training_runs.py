"""Collect and validate completed Kaggle training runs against a source snapshot."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import numpy as np
from evaluation import ROOT, metrics
from experiment_state import atomic_json
from large_data import PARTITIONS, rows, validate_manifest
from prepare_large_corpus import file_hash


def unpack(archive_path,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Invalid checkpoint archive path')
        archive.extractall(destination)


def collect(kernel,source):
    slug=kernel.split('/')[-1]
    download=ROOT / 'kaggle/output' / slug; download.mkdir(parents=True,exist_ok=True)
    subprocess.run(['kaggle','kernels','output',kernel,'-p',str(download),'-o','--file-pattern',
        r'(^|/)(training_results[.]zip|.*_seed_[0-9]+_(model|predictions)[.]zip|.*[.]log)$'],check=True)
    protocol=json.loads((source / 'configs/training_protocol.json').read_text())
    revision=json.loads((source / 'source_revision.json').read_text())
    data=source / 'data/processed/large'; manifest=validate_manifest(data)
    metadata={split:[{k:r[k] for k in ['id','label','family','group']} for r in rows(data / f'{split}.jsonl')] for split in PARTITIONS}
    with zipfile.ZipFile(download / 'training_results.zip') as archive:
        reports=[json.loads(archive.read(name)) for name in archive.namelist() if name.endswith('/result.json')]
    if not reports: raise ValueError('No complete training results were exported')
    imported=[]
    for report in reports:
        model=report['model']; seed=report['seed']
        if report['smoke_only'] or report['status']!='complete' or report['train_rows']!=100000:
            raise ValueError('Expected a complete 100000-row experiment')
        if report['protocol_sha256']!=file_hash(source / 'configs/training_protocol.json'):
            raise ValueError('Training configuration differs')
        if report['config'].get('source_revision')!=revision: raise ValueError('Source revision differs')
        for name,sha in report['code_hashes'].items():
            if file_hash(source / 'src' / name)!=sha: raise ValueError(f'Training code differs: {name}')
        if report['data_hashes']!={s:manifest['partitions'][s]['sha256'] for s in PARTITIONS}:
            raise ValueError('Partitions differ')
        if report['training_rows_seen']!=100000*report['epochs_run']:
            raise ValueError('Training row budget is incomplete')
        if not 1<=report['epochs_run']<=protocol['models'][model]['epoch_cap'] or seed not in protocol['seeds']:
            raise ValueError('Training budget or seed differs')
        if model in ('distilbert','deberta') and not all(report[k] for k in ['encoder_gradient_verified','encoder_weight_changed']):
            raise ValueError('Encoder training is unverified')
        relative=Path(report['condition']) / model / f'seed_{seed}'
        artifact=ROOT / 'artifacts/experiments' / relative
        result_dir=ROOT / 'results/experiments' / relative; result_dir.mkdir(parents=True,exist_ok=True)
        unpack(download / f'{model}_seed_{seed}_predictions.zip',artifact)
        unpack(download / f'{model}_seed_{seed}_model.zip',artifact)
        for split in PARTITIONS:
            with np.load(artifact / f'{split}_predictions.npz') as predictions:
                if not np.array_equal(predictions['ids'],[r['id'] for r in metadata[split]]) or not np.array_equal(predictions['labels'],[r['label'] for r in metadata[split]]):
                    raise ValueError(f'Prediction identifiers differ: {model}/{seed}/{split}')
                recomputed=metrics(metadata[split],predictions['probabilities'])
            for key,value in recomputed.items():
                recorded=report['metrics'][split][key]
                if value is None:
                    if recorded is not None: raise ValueError('Metric availability differs')
                elif not np.isclose(value,recorded,atol=1e-8,rtol=0):
                    raise ValueError(f'Metric mismatch: {model}/{seed}/{split}/{key}')
        atomic_json(result_dir / 'config.json',report['config'])
        atomic_json(result_dir / 'result.json',report)
        atomic_json(result_dir / 'import_verification.json',{'kernel':kernel,'metrics_recomputed':True,
            'source_revision':revision,'model_archive_sha256':file_hash(download / f'{model}_seed_{seed}_model.zip'),
            'prediction_archive_sha256':file_hash(download / f'{model}_seed_{seed}_predictions.zip')})
        # Keep selected weights and unpacked predictions; resumable states remain in Kaggle output.
        (download / f'{model}_seed_{seed}_model.zip').unlink()
        (download / f'{model}_seed_{seed}_predictions.zip').unlink()
        imported.append({'model':model,'seed':seed})
    return imported


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernels',nargs='+',required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    remaining=list(args.kernels)
    record=ROOT / 'artifacts/training_logs/kaggle_collection.json'
    collected=json.loads(record.read_text()) if record.exists() else {}
    while remaining:
        for kernel in list(remaining):
            if kernel in collected: remaining.remove(kernel); continue
            status=subprocess.run(['kaggle','kernels','status',kernel],capture_output=True,text=True,check=True).stdout
            print(status.strip(),flush=True)
            if 'ERROR' in status or 'CANCEL' in status: raise RuntimeError(status)
            if 'COMPLETE' in status:
                imported=collect(kernel,args.source_dir)
                collected[kernel]=imported; atomic_json(record,collected); remaining.remove(kernel)
                subprocess.run([__import__('sys').executable,str(ROOT / 'src/aggregate_experiments.py')],cwd=ROOT,check=True)
        if remaining:
            if not args.watch: return
            time.sleep(30)
    print(json.dumps({'collected':collected}),flush=True)


if __name__=='__main__': main()
