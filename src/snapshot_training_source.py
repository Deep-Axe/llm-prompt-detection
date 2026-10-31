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
from condition_data import condition_directory
from prepare_large_corpus import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='HEAD')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--copy-data',action='store_true')
    p.add_argument('--protocol',default='configs/training_protocol.json')
    p.add_argument('--conditions',nargs='+',default=['wildjailbreak-100k'])
    args=p.parse_args()
    revision=subprocess.check_output(['git','rev-parse',args.revision],cwd=ROOT,text=True).strip()
    archive=subprocess.check_output(['git','archive',revision,'src','configs','LICENSE'],cwd=ROOT)
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
        protocol=json.loads((args.output / args.protocol).read_text())
        for condition in args.conditions:
            source=condition_directory(protocol,condition)
            destination=condition_directory(protocol,condition,root=args.output)
            destination.mkdir(parents=True,exist_ok=True)
            for name in ['manifest.json','train.jsonl','val.jsonl','test.jsonl','official_eval.jsonl']:
                if not (destination / name).exists():
                    try:
                        __import__('os').link(source / name,destination / name)
                    except OSError:
                        shutil.copy2(source / name,destination / name)
            validate_manifest(destination)
        if protocol.get('vocabulary_source'):
            definition=protocol['vocabulary_source']
            source=ROOT/definition['path'];destination=args.output/definition['path']
            if file_hash(source)!=definition['sha256']: raise ValueError('Vocabulary source changed')
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists(): shutil.copy2(source,destination)
            if file_hash(destination)!=definition['sha256']: raise ValueError('Snapshot vocabulary differs')
    print(args.output)


if __name__=='__main__': main()
