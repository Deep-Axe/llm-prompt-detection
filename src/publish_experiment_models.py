"""Publish selected experiment weights and aggregate metadata, excluding training state and prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluation import ROOT
from experiment_state import atomic_json
from predict_large_models import MODEL_FILES
from prepare_large_corpus import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id',default='DeeAxe/llm-prompt-detection')
    parser.add_argument('--cnn-extensions',action='store_true')
    args=parser.parse_args()
    staging=ROOT/'artifacts/hf_experiment_publication'/('cnn_extensions' if args.cnn_extensions else 'training_conditions')
    staging.mkdir(parents=True,exist_ok=True)
    report_root=ROOT/('results/cnn_extensions' if args.cnn_extensions else 'results/experiments')
    artifact_root=ROOT/('artifacts/cnn_extensions' if args.cnn_extensions else 'artifacts/experiments')
    reports=sorted(report_root.glob('*/cnn/seed_*/result.json') if args.cnn_extensions else report_root.glob('*/*/seed_*/result.json'))
    manifest=[]
    for path in reports:
        r=json.loads(path.read_text())
        if r['status']!='complete' or r['smoke_only']: raise ValueError('Only completed full experiments can be published')
        condition=r['condition'];model=r['model'];seed=r['seed']
        artifact=artifact_root/condition/model/f'seed_{seed}'
        source=artifact/'model' if model=='distilbert' else artifact
        relative=Path(condition)/model/f'seed_{seed}';destination=staging/relative;destination.mkdir(parents=True,exist_ok=True)
        filenames=list(MODEL_FILES[model])
        if model=='cnn' and (artifact/'model_config.json').exists(): filenames.append('model_config.json')
        for name in filenames:
            target=destination/name
            if target.exists():
                if file_hash(target)!=file_hash(source/name): raise ValueError('Publication staging contains different weights')
            else:
                try: os.link(source/name,target)
                except OSError: shutil.copy2(source/name,target)
        shutil.copy2(path,destination/'training_result.json')
        text=f'''# {condition}: {model}, seed {seed}

Binary prompt-harmfulness classifier: 0 = benign, 1 = harmful. The threshold is 0.5. This checkpoint was selected using validation F1.

Training prompts: {r['train_rows']:,}. Input limit: {r['config']['max_tokens']} tokens. Epochs run: {r['epochs_run']}; selected epoch: {r['chosen_epoch']}. Trainable parameters: {r['total_parameters']:,}.

Shared-test F1: {r['metrics']['test']['f1']:.6f}. The shared test was also used by the original baseline. These measurements do not establish performance on arbitrary applications or prompt injection.

Training source: allenai/wildjailbreak, revision 5ddc12a7894f842b0619b8e1c7ee496b198af009, ODC-BY (https://opendatacommons.org/licenses/by/1-0/). DistilBERT, where applicable: distilbert/distilbert-base-uncased, revision 12040accade4e8a0f71eabdb258fecc2e7e948be, Apache-2.0. Code: https://github.com/Deep-Axe/llm-prompt-detection (MIT).

Raw prompts, per-example predictions, and separate recovery-checkpoint files are excluded. Joblib files serialize the recorded scikit-learn estimator. The training result records model settings, data hashes, source revision, and environment.

Load with `src/predict_experiments.py` from the code repository, using condition `{condition}`, model `{model}`, and seed `{seed}`. PyTorch checkpoint weights use float32. Scikit-learn/joblib checkpoints retain their recorded dtypes and require the recorded dependency versions and trusted files.
'''
        (destination/'README.md').write_text(text)
        files={str(p.relative_to(staging)):file_hash(p) for p in destination.iterdir() if p.is_file()}
        manifest.append({'condition':condition,'model':model,'seed':seed,'path':f'experiments/{relative.as_posix()}',
                         'files_sha256':files})
    if not manifest: raise ValueError('No completed models to publish')
    # Dedicated staging contains only explicitly selected portable files and their result cards.
    allowed=set().union(*[set(row['files_sha256']) for row in manifest])
    actual={str(p.relative_to(staging)) for p in staging.rglob('*') if p.is_file()}
    if actual!=allowed: raise ValueError('Unexpected publication files')
    command=[str(ROOT/'.venv/bin/hf'),'upload',args.repo_id,str(staging),'experiments','--repo-type','model',
        '--commit-message','Publish CNN capacity checkpoints' if args.cnn_extensions else 'Publish repeated-seed classifiers for nested training pools']
    subprocess.run(command,check=True)
    from huggingface_hub import HfApi
    info=HfApi().model_info(args.repo_id,files_metadata=True)
    remote={p.rfilename:p for p in info.siblings}
    for row in manifest:
        for relative,digest in row['files_sha256'].items():
            path='experiments/'+relative
            sibling=remote[path]
            local=staging/relative
            if sibling.lfs:
                if sibling.lfs.sha256!=digest: raise ValueError('Uploaded checkpoint checksum differs')
            else:
                payload=local.read_bytes()
                git_digest=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
                if sibling.blob_id!=git_digest: raise ValueError('Uploaded metadata checksum differs')
    destination=ROOT/('results/cnn_extensions/huggingface_publication.json' if args.cnn_extensions else 'results/experiments/huggingface_publication.json')
    atomic_json(destination,{'repository':args.repo_id,'revision':info.sha,'models_published':len(manifest),
        'remote_checksums_verified':True,'models':manifest})
    print(json.dumps({'repository':args.repo_id,'revision':info.sha,'models_published':len(manifest)}),flush=True)


if __name__=='__main__': main()
