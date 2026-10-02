"""Download immutable DistilBERT weights with the Hugging Face CLI."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]


def main():
    destination=ROOT/'data/models/distilbert'
    destination.mkdir(parents=True,exist_ok=True)
    source=destination/'source.json'
    info=json.loads(source.read_text()) if source.exists() else {
        'repository':'distilbert/distilbert-base-uncased',
        'revision':HfApi().model_info('distilbert/distilbert-base-uncased').sha}
    hf=shutil.which('hf')
    if hf is None: raise SystemExit('Install huggingface-hub to obtain the hf CLI')
    subprocess.run([hf,'download',info['repository'],'--revision',info['revision'],
        '--include','config.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors',
        '--local-dir',str(destination),'--cache-dir',str(ROOT/'data/.hf-cache'),'--max-workers','2'],check=True)
    info['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file() and p.name!='source.json'}
    source.write_text(json.dumps(info,indent=2)+'\n')
    (ROOT/'results/model_source.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__': main()
