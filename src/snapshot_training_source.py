"""Export committed training code and optionally copy the fixed training dataset."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from evaluation import ROOT
from experiment_state import atomic_json
from large_data import validate_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='HEAD')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--copy-data',action='store_true')
    args=p.parse_args()
    revision=subprocess.check_output(['git','rev-parse',args.revision],cwd=ROOT,text=True).strip()
    archive=subprocess.check_output(['git','archive',revision,'src','configs/training_protocol.json','LICENSE'],cwd=ROOT)
    args.output.mkdir(parents=True,exist_ok=True)
    marker=args.output / 'source_revision.json'
    expected={'commit':revision,'archive_sha256':hashlib.sha256(archive).hexdigest()}
    if marker.exists():
        if json.loads(marker.read_text())!=expected: raise ValueError('Source directory belongs to another revision')
    else:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for entry in tar.getmembers():
                if not (args.output / entry.name).resolve().is_relative_to(args.output.resolve()):
                    raise ValueError('Invalid archive path')
            tar.extractall(args.output)
        atomic_json(marker,expected)
    if args.copy_data:
        destination=args.output / 'data/processed/large'
        destination.mkdir(parents=True,exist_ok=True)
        for name in ['manifest.json','train.jsonl','val.jsonl','test.jsonl','official_eval.jsonl']:
            if not (destination / name).exists(): shutil.copy2(ROOT / 'data/processed/large' / name,destination / name)
        validate_manifest(destination)
    print(args.output)


if __name__=='__main__': main()
