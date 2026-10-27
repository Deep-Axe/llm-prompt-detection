"""Kaggle GPU verification of interrupted/resumed Part C runners."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.46.3',
                'huggingface-hub==0.36.2','scikit-learn==1.3.2','numpy==1.26.4'],check=True)
subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchvision'],check=True)
import torch
if not torch.cuda.is_available(): raise RuntimeError('GPU allocation is required for this verification')
print(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}),flush=True)
root=Path('/kaggle/working/experiment')
root.mkdir(parents=True,exist_ok=True)
inputs=Path('/kaggle/input')
base_zips=list(inputs.rglob('experiment.zip'))
if len(base_zips)==1:
    with zipfile.ZipFile(base_zips[0]) as z: z.extractall(root)
else:
    manifests=list(inputs.rglob('data/processed/large/manifest.json'))
    if len(manifests)!=1: raise RuntimeError(f'Expected one baseline input; got {len(manifests)}')
    shutil.copytree(manifests[0].parents[3],root,dirs_exist_ok=True)
overlays=list(inputs.rglob('protocol_runner.zip'))
if len(overlays)==1:
    with zipfile.ZipFile(overlays[0]) as z: z.extractall(root)
else:
    configs=list(inputs.rglob('configs/part_c_protocol.json'))
    if len(configs)!=1: raise RuntimeError('Cannot locate protocol runner input')
    shutil.copytree(configs[0].parents[1],root,dirs_exist_ok=True)
verification=root / 'artifacts/resume_verification'
command=[sys.executable,str(root / 'src/verify_experiment_resume.py'),'--device','cuda','--root',str(verification)]
try:
    subprocess.run(command,cwd=root,check=True)
    shutil.copy2(verification / 'resume_verification.json','/kaggle/working/resume_verification.json')
    subprocess.run([sys.executable,str(root / 'src/aggregate_experiments.py'),'--root',str(verification),
                    '--output','/kaggle/working/smoke_aggregation.json'],cwd=root,check=True)
    with zipfile.ZipFile('/kaggle/working/protocol_smoke_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in verification.rglob('*.json'):
            z.write(path,path.relative_to(verification))
    print('All four save/resume checks passed.',flush=True)
finally:
    # Failed runs retain last/best checkpoints so a failure can be investigated.
    summary=verification / 'resume_verification.json'
    if summary.exists(): shutil.copy2(summary,'/kaggle/working/resume_verification.json')
    if Path('/kaggle/working/resume_verification.json').exists() and json.loads(Path('/kaggle/working/resume_verification.json').read_text()).get('status')=='passed':
        # Keep only recovery/best checkpoints, features, vocabulary, and small outputs.
        # Original baseline inputs and token caches are reproducible from private inputs.
        for cache in verification.rglob('cache'):
            if cache.is_dir(): shutil.rmtree(cache)
        for folder in ['data','src','results','configs']:
            shutil.rmtree(root / folder,ignore_errors=True)
