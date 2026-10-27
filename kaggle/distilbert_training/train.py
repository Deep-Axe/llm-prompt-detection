"""Train prompt classifiers from a committed source snapshot."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
archives=list(Path('/kaggle/input').rglob('training_source.zip'))
if len(archives)==1:
    source=Path('/kaggle/working/training_source')
    with zipfile.ZipFile(archives[0]) as archive: archive.extractall(source)
else:
    scripts=list(Path('/kaggle/input').rglob('src/run_kaggle_training.py'))
    if len(scripts)!=1: raise RuntimeError('Cannot locate the training source snapshot')
    source=scripts[0].parents[1]
TASKS = [{'model': 'distilbert', 'seed': 4442, 'device': 'cuda'}, {'model': 'distilbert', 'seed': 4443, 'device': 'cuda'}]
subprocess.run([sys.executable,str(source/'src/run_kaggle_training.py'),
                '--source-dir',str(source),'--tasks',json.dumps(TASKS)],check=True)
