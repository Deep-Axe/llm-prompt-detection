"""Check uploaded file hashes and record the published model revision."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi
from evaluation import ROOT
from prepare_large_corpus import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', default='DeeAxe/llm-prompt-detection')
    args = p.parse_args()
    folder = ROOT / 'artifacts/huggingface/llm-prompt-detection'
    info = HfApi().model_info(args.repo_id, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    hashes = {}
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.cache' in path.parts:
            continue
        relative = path.relative_to(folder).as_posix()
        uploaded = remote[relative]
        checksum = file_hash(path)
        if uploaded.lfs is not None:
            assert uploaded.lfs.sha256 == checksum, relative
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert uploaded.blob_id == blob, relative
        hashes[relative] = checksum
    assert not any(f.endswith(('.npz', '.jsonl')) for f in remote), 'Unexpected per-example data'
    publication = {'repo_id': args.repo_id, 'url': f'https://huggingface.co/{args.repo_id}',
                   'revision': info.sha, 'private': info.private,
                   'verification': 'All staged files match remote Git-blob or LFS SHA hashes',
                   'files': hashes}
    (ROOT / 'results/large/huggingface_publication.json').write_text(json.dumps(publication, indent=2)+'\n')
    inventory_path = ROOT / 'results/large/checkpoint_manifest.json'
    inventory = json.loads(inventory_path.read_text())
    for name, record in inventory.items():
        record.update(uploaded=True, repo_id=args.repo_id, revision=info.sha, subfolder=name)
    inventory_path.write_text(json.dumps(inventory, indent=2)+'\n')
    print(json.dumps({'url': publication['url'], 'revision': info.sha, 'verified_files': len(hashes)}))


if __name__ == '__main__':
    main()
