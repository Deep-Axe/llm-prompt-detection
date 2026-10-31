"""Compare CNN width and input context on fixed prompt-harmfulness data."""
from pathlib import Path
import subprocess
import sys
import zipfile
archives=list(Path('/kaggle/input').rglob('training_source.zip'))
if len(archives)==1:
    source=Path('/kaggle/working/training_source')
    with zipfile.ZipFile(archives[0]) as archive: archive.extractall(source)
else:
    scripts=list(Path('/kaggle/input').rglob('src/run_kaggle_cnn_extensions.py'))
    if len(scripts)!=1: raise RuntimeError('Cannot locate the frozen CNN source')
    source=scripts[0].parents[1]
subprocess.run([sys.executable,str(source/'src/run_kaggle_cnn_extensions.py'),
 '--source-dir',str(source),'--deadline','2026-10-31T12:00:00+05:30'],check=True)
