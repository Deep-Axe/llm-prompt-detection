"""Collect a finished Kaggle transformer run and verify its partition identity."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import zipfile
from evaluation import ROOT
from large_data import PARTITIONS, validate_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', default='deepamahuja/prompt-detection-distilbert-100k')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    while True:
        status = subprocess.run(['kaggle','kernels','status',args.kernel], capture_output=True, text=True, check=True).stdout
        print(status.strip(), flush=True)
        if 'COMPLETE' in status:
            break
        if 'ERROR' in status or 'CANCEL' in status:
            raise RuntimeError(status.strip())
        if not args.watch:
            return
        time.sleep(30)
    output = ROOT / 'kaggle/output'; output.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle','kernels','output',args.kernel,'-p',str(output),'-o',
                    '--file-pattern',r'(^|/)(distilbert[.]json|distilbert_model[.]zip|distilbert_predictions[.]zip)$'], check=True)
    report_path = output / 'distilbert.json'
    report = json.loads(report_path.read_text())
    manifest = validate_manifest(ROOT / 'data/processed/large')
    if report['data_hashes'] != {s:manifest['partitions'][s]['sha256'] for s in PARTITIONS}:
        raise ValueError('Kaggle used different partitions')
    if report.get('encoder_frozen') is not False or not report.get('encoder_gradient_verified') or not report.get('encoder_weight_changed'):
        raise ValueError('Encoder fine-tuning not verified')
    if report['train_rows'] < 100000 or report['epochs_run'] < 3:
        raise ValueError('Kaggle training budget is incomplete')
    artifact = ROOT / 'artifacts/large/distilbert'
    model = artifact / 'model'; model.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output / 'distilbert_model.zip') as z:
        z.extractall(model)
    with zipfile.ZipFile(output / 'distilbert_predictions.zip') as z:
        z.extractall(artifact)
    # These metadata labels name the existing class indices; weights are unchanged.
    config = json.loads((model / 'config.json').read_text())
    config.update(id2label={'0':'benign','1':'harmful'}, label2id={'benign':0,'harmful':1})
    (model / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    if 'model_source' not in report:
        report['model_source'] = json.loads((ROOT / 'data/models/distilbert/source.json').read_text())
        report['model_source_metadata_added_on_collection'] = True
    report['kaggle_kernel'] = args.kernel
    (ROOT / 'results/large/distilbert.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Collected and verified full DistilBERT training, model, and predictions.', flush=True)


if __name__ == '__main__':
    main()
