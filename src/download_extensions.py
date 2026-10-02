"""Download a candidate dataset to its own raw directory; never merge it.

Uses a commit-pinned snapshot and writes source and checksum metadata.
Requires network access and the Hugging Face CLI. WildGuard access must
already have been granted by its provider; this script accepts no terms.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'regular': ('TrustAIRLab/in-the-wild-jailbreak-prompts', ['regular_2023_12_25/*', 'README.md']),
    'injection': ('deepset/prompt-injections', ['data/*', 'README.md']),
    'wildguard': ('allenai/wildguardmix', ['*.parquet', 'README.md']),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCES)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    repo, patterns = SOURCES[args.source]
    info = HfApi().dataset_info(repo, revision=args.revision)
    revision = info.sha
    destination = ROOT/'data/raw/extensions'/args.source/revision
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(['hf','download',repo,'--repo-type','dataset','--revision',revision,
                    '--local-dir',str(destination),'--cache-dir',str(ROOT/'data/.hf-cache'),
                    '--include',*patterns],check=True)
    files = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in destination.rglob('*') if p.is_file() and '.cache' not in p.parts}
    if not any(p.endswith('.parquet') for p in files):
        raise SystemExit('No parquet files downloaded; inspect repository layout before proceeding')
    manifest = {'repository':repo,'revision':revision,'files_sha256':files,'status':'downloaded, not merged or label-audited'}
    out = ROOT/'results'/f'{args.source}_acquisition.json'
    out.write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
