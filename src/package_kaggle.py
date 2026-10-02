"""Package a private, checksum-identical Kaggle input and full-finetuning job."""
import argparse
import json
from pathlib import Path
import zipfile
from evaluation import ROOT
from large_data import validate_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--username', default='deepamahuja')
    args = p.parse_args()
    validate_manifest(ROOT / 'data/processed/large')
    folder = ROOT / 'kaggle/input'; folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / 'experiment.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in (ROOT / 'src').glob('*.py'):
            z.write(path, path.relative_to(ROOT))
        for path in (ROOT / 'data/processed/large').glob('*.json*'):
            z.write(path, path.relative_to(ROOT))
        for path in (ROOT / 'data/models/distilbert').iterdir():
            if path.is_file():
                z.write(path, path.relative_to(ROOT))
        z.write(ROOT / 'LICENSE', 'LICENSE')
        z.write(ROOT / 'data/raw/large/wildjailbreak/README.md', 'WILDJAILBREAK_README.md')
    metadata = {'title':'WildJailbreak 100k grouped training input',
                'id':f'{args.username}/wildjailbreak-100k-grouped',
                'licenses':[{'name':'other'}],
                'description':'Private research input. WildJailbreak by Jiang et al. (2024), https://huggingface.co/datasets/allenai/wildjailbreak, ODC-BY-1.0. Training subset: 100000 unique prompts, validation/test grouped by underlying vanilla request. DistilBERT base weights: Apache-2.0, distilbert/distilbert-base-uncased revision 12040accade4e8a0f71eabdb258fecc2e7e948be. Experiment code: MIT. See included upstream card, LICENSE, and source/partition manifests. Labels represent prompt harmfulness; no model completions are used.'}
    (folder / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    kernel = ROOT / 'kaggle/distilbert'; kernel.mkdir(parents=True, exist_ok=True)
    kernel_metadata = {'id':f'{args.username}/prompt-detection-distilbert-100k',
                       'title':'Prompt detection DistilBERT 100k', 'code_file':'train.py',
                       'language':'python', 'kernel_type':'script', 'is_private':True,
                       'enable_gpu':True, 'enable_internet':True,
                       'dataset_sources':[metadata['id']], 'competition_sources':[], 'kernel_sources':[]}
    (kernel / 'kernel-metadata.json').write_text(json.dumps(kernel_metadata, indent=2)+'\n')
    print(f'Packaged {folder / "experiment.zip"}; submit input first, then kernel.')


if __name__ == '__main__':
    main()
