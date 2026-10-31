"""Evaluate the remaining validation-selected DistilBERT checkpoints on WildGuardMix."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile
subprocess.run([sys.executable,'-m','pip','install','-q','numpy==1.26.4','scikit-learn==1.3.2',
 'transformers==4.46.3','huggingface-hub==0.36.2'],check=True)
subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchvision'],check=True)
root=Path('/kaggle/working/transfer_evaluation');root.mkdir(exist_ok=True)
archives=list(Path('/kaggle/input').rglob('training_source.zip'))
if len(archives)==1:
 with zipfile.ZipFile(archives[0]) as archive: archive.extractall(root)
else:
 scripts=list(Path('/kaggle/input').rglob('src/evaluate_transfer.py'))
 if len(scripts)!=1: raise RuntimeError('Expected one frozen evaluation source')
 shutil.copytree(scripts[0].parents[1],root,dirs_exist_ok=True)
reports=list(Path('/kaggle/input').rglob('training_results.zip'))
if len(reports)!=1: raise RuntimeError('Expected one completed encoder output')
count=0
with zipfile.ZipFile(reports[0]) as archive:
 for name in archive.namelist():
  if not name.endswith('/result.json'): continue
  payload=archive.read(name);r=json.loads(payload)
  if r['model']!='distilbert': raise ValueError('Expected encoder checkpoints')
  relative=Path(r['condition'])/r['model']/f"seed_{r['seed']}"
  output=root/'results/experiments'/relative;output.mkdir(parents=True,exist_ok=True)
  (output/'result.json').write_bytes(payload)
  model_archives=list(Path('/kaggle/input').rglob(f"{r['condition']}_distilbert_seed_{r['seed']}_model.zip"))
  if len(model_archives)!=1: raise RuntimeError('Expected one selected model archive')
  artifact=root/'artifacts/experiments'/relative;artifact.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(model_archives[0]) as models: models.extractall(artifact)
  count+=1
print('Selected encoder checkpoints:',count,flush=True)
subprocess.run([sys.executable,str(root/'src/evaluate_transfer.py'),'--device','cuda','--models','distilbert',
 '--conditions','wildjailbreak-150k','wildjailbreak-183005'],cwd=root,check=True)
import torch
for path in (root/'results/transfer').rglob('result.json'):
 r=json.loads(path.read_text());r['execution_environment']={'device':'cuda:0','gpu':torch.cuda.get_device_name(0),
  'torch':torch.__version__,'precision':'float32','kaggle_kernel':'deepamahuja/prompt-detection-distilbert-transfer'}
 path.write_text(json.dumps(r,indent=2)+'\n')
with zipfile.ZipFile('/kaggle/working/transfer_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
 for path in (root/'results/transfer').rglob('*.json'): archive.write(path,path.relative_to(root/'results/transfer'))
with zipfile.ZipFile('/kaggle/working/transfer_predictions.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
 for path in (root/'artifacts/transfer').rglob('*.npz'): archive.write(path,path.relative_to(root/'artifacts/transfer'))
for name in ['artifacts','data','src','configs','results']: shutil.rmtree(root/name,ignore_errors=True)
print('Transfer evaluation complete.',flush=True)
