"""Collect fixed CNN extension checkpoints and verify their source and metric records."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from evaluation import ROOT
from experiment_state import atomic_json
from prepare_large_corpus import file_hash
from collect_training_runs import unpack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--kernel',default='deepamahuja/prompt-detection-cnn-width-context')
    args=parser.parse_args()
    while True:
        try:
            status=subprocess.check_output(['kaggle','kernels','status',args.kernel],text=True)
            print(status.strip(),flush=True)
            if 'COMPLETE' in status: break
            if 'ERROR' in status or 'CANCEL' in status: raise RuntimeError(status)
        except subprocess.CalledProcessError as error: print(f'Retrying Kaggle status: {error}',flush=True)
        time.sleep(20)
    download=ROOT/'kaggle/output/prompt-detection-cnn-width-context';download.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        try:
            subprocess.run(['kaggle','kernels','output',args.kernel,'-p',str(download),'-o','--file-pattern',
                r'(^|/)(cnn_training_results[.]zip|cnn_execution_summary[.]json|.*_seed_[0-9]+_(model|predictions)[.]zip|.*[.]log)$'],check=True)
            break
        except subprocess.CalledProcessError:
            if attempt==4: raise
            time.sleep(10)
    revision=json.loads((args.source_dir/'source_revision.json').read_text())
    imported=[]
    with zipfile.ZipFile(download/'cnn_training_results.zip') as archive:
        reports=[json.loads(archive.read(name)) for name in archive.namelist() if name.endswith('/result.json') and '/cnn/seed_' in name]
    for report in reports:
        context=report['config']['max_tokens'];condition=report['condition'];seed=report['seed']
        protocol=args.source_dir/f'configs/cnn_wide_{context}.json'
        if context not in [256,512] or condition!=f'wildjailbreak-100k-cnn-wide-{context}': raise ValueError('Unexpected CNN condition')
        if report['status']!='complete' or report['smoke_only'] or report['train_rows']!=100000: raise ValueError('Expected a complete CNN run')
        if report['training_rows_seen']!=100000*report['epochs_run']: raise ValueError('Incomplete training budget')
        if report['protocol_sha256']!=file_hash(protocol) or report['config']['source_revision']!=revision: raise ValueError('Frozen source differs')
        for name,digest in report['code_hashes'].items():
            if file_hash(args.source_dir/'src'/name)!=digest: raise ValueError('Training code differs')
        relative=Path(condition)/'cnn'/f'seed_{seed}'
        artifact=ROOT/'artifacts/cnn_extensions'/relative
        output=ROOT/'results/cnn_extensions'/relative;output.mkdir(parents=True,exist_ok=True)
        prefix=f'{condition}_seed_{seed}'
        unpack(download/f'{prefix}_model.zip',artifact);unpack(download/f'{prefix}_predictions.zip',artifact)
        atomic_json(output/'result.json',report)
        atomic_json(output/'config.json',report['config'])
        atomic_json(output/'import_verification.json',{'kernel':args.kernel,'source_revision':revision,
            'model_archive_sha256':file_hash(download/f'{prefix}_model.zip'),
            'predictions_archive_sha256':file_hash(download/f'{prefix}_predictions.zip')})
        imported.append({'condition':condition,'seed':seed})
    if not imported: raise ValueError('No completed CNN checkpoints')
    # Inference and metric verification ran on the same Kaggle GPUs as training.
    with zipfile.ZipFile(download/'cnn_training_results.zip') as archive:
        for name in archive.namelist():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts: raise ValueError('Invalid report path')
            if not (name.endswith('/inference_verification.json') or name.startswith('transfer/') and name.endswith('/result.json')): continue
            record=json.loads(archive.read(name))
            if name.endswith('/inference_verification.json'):
                artifact=ROOT/'artifacts/cnn_extensions'/path.parent
                if record['status']!='verified' or not record['all_saved_metrics_recomputed']: raise ValueError('Inference verification failed')
                if record['checkpoint_sha256']!=file_hash(artifact/'best.pt'): raise ValueError('Verified checkpoint differs')
                if record['model_config_sha256']!=file_hash(artifact/'model_config.json'): raise ValueError('Verified model settings differ')
            else:
                if record['status']!='complete' or record['n']!=1695 or record['threshold']!=.5: raise ValueError('Unexpected transfer evaluation')
            target=ROOT/'results/cnn_extensions'/path
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
    subprocess.run([sys.executable,str(ROOT/'src/summarize_cnn_extensions.py')],cwd=ROOT,check=True)
    atomic_json(ROOT/'artifacts/training_logs/cnn_extensions/kaggle_collection.json',{'status':'verified','runs':imported,'source_revision':revision})
    print(json.dumps({'status':'verified','imported_runs':len(imported)}),flush=True)


if __name__=='__main__': main()
