"""Kaggle entry point: full DistilBERT fine-tuning, shared 100k training rows."""
import os
from pathlib import Path
import subprocess
import shutil
import sys
import zipfile

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Keep Torch/CUDA provided by Kaggle; pin the tokenizer/model implementation.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.46.3',
                'huggingface-hub==0.36.2', 'scikit-learn==1.3.2', 'numpy==1.26.4'], check=True)
# This text-only job does not need torchvision; incompatible preinstalled builds
# can otherwise break transformers' optional vision import.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision'], check=True)
root = Path('/kaggle/working/experiment'); root.mkdir(parents=True, exist_ok=True)
inputs = list(Path('/kaggle/input').rglob('experiment.zip'))
if len(inputs) == 1:
    with zipfile.ZipFile(inputs[0]) as z:
        z.extractall(root)
else:
    # Kaggle commonly extracts uploaded ZIPs while creating the dataset.
    manifests = list(Path('/kaggle/input').rglob('data/processed/large/manifest.json'))
    if len(manifests) != 1:
        raise RuntimeError(f'Expected one experiment manifest, found {len(manifests)}')
    shutil.copytree(manifests[0].parents[3], root, dirs_exist_ok=True)
subprocess.run([sys.executable, str(root / 'src/train_large_neural.py'), 'distilbert',
                '--batch-size', '32', '--accumulation', '1', '--epochs', '3',
                '--max-tokens', '256', '--device', 'cuda'], cwd=root, check=True)
# Retain deployable weights and test predictions without exporting token caches.
with zipfile.ZipFile('/kaggle/working/distilbert_model.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for path in (root / 'artifacts/large/distilbert/model').rglob('*'):
        if path.is_file():
            z.write(path, path.relative_to(root / 'artifacts/large/distilbert/model'))
for path in (root / 'results/large').glob('*.json'):
    shutil.copy2(path, Path('/kaggle/working') / path.name)
with zipfile.ZipFile('/kaggle/working/distilbert_predictions.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in (root / 'artifacts/large/distilbert').glob('*_predictions.npz'):
        z.write(path, path.name)
shutil.rmtree(root)
