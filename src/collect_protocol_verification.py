"""Wait for the Kaggle smoke verification and import its small audit reports."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import zipfile
from evaluation import ROOT
from experiment_state import atomic_json
from prepare_large_corpus import file_hash
from train_experiment import CODE_FILES
from evaluation import code_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',default='deepamahuja/prompt-detection-protocol-resume-check')
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    while True:
        status=subprocess.run(['kaggle','kernels','status',args.kernel],capture_output=True,text=True,check=True).stdout
        print(status.strip(),flush=True)
        if 'COMPLETE' in status: break
        if 'ERROR' in status or 'CANCEL' in status: raise RuntimeError(status)
        if not args.watch: return
        time.sleep(30)
    target=ROOT / 'kaggle/output/protocol_smoke'; target.mkdir(parents=True,exist_ok=True)
    subprocess.run(['kaggle','kernels','output',args.kernel,'-p',str(target),'-o','--file-pattern',
                    r'(^|/)(resume_verification[.]json|smoke_aggregation[.]json|protocol_smoke_results[.]zip|.*[.]log)$'],check=True)
    report=json.loads((target / 'resume_verification.json').read_text())
    protocol=json.loads((ROOT / 'configs/training_protocol.json').read_text())
    if report['protocol_sha256']!=file_hash(ROOT / 'configs/training_protocol.json'): raise ValueError('Protocol hash differs')
    if report['status']!='passed' or set(report['models'])!={'mlp','cnn','bilstm','distilbert'}:
        raise ValueError('Four-model verification is incomplete')
    for model,check in report['models'].items():
        if not check['passed'] or not check['smoke_only'] or not check['resumed_mid_epoch']:
            raise ValueError(f'Incomplete save/resume verification: {model}')
        if check['train_rows']!=protocol['smoke_policy']['train_rows']: raise ValueError('Smoke budget differs')
        if model=='distilbert' and not all(check[k] for k in ['amp_state_verified','encoder_gradient_verified','encoder_weight_changed']):
            raise ValueError('Encoder/AMP verification incomplete')
    local_inventory={name:sha for name,sha in protocol['baseline_inventory'].items() if not name.endswith('.md')}
    if any(file_hash(ROOT / path)!=sha for path,sha in local_inventory.items()):
        raise ValueError('A local baseline file changed')
    actual_code=code_hashes(CODE_FILES)
    environments={}
    with zipfile.ZipFile(target / 'protocol_smoke_results.zip') as archive:
        run_reports=[json.loads(archive.read(name)) for name in archive.namelist() if name.endswith('/result.json')]
    if len(run_reports)!=8: raise ValueError('Expected reference and resumed records for all four models')
    for run in run_reports:
        if run['code_hashes']!=actual_code: raise ValueError('Kaggle verified a different runner version')
        if not run['smoke_only'] or run['status']!='complete': raise ValueError('Unexpected verification result')
        environments[run['model']]=run['environment']
    report.update(kaggle_kernel=args.kernel,local_baseline_files_checked=len(local_inventory),
                  local_baseline_unchanged=True,runner_code_hashes=actual_code,
                  environments=environments)
    output=ROOT / 'results/protocol_verification'; output.mkdir(parents=True,exist_ok=True)
    atomic_json(output / 'kaggle_resume_verification.json',report)
    aggregation=json.loads((target / 'smoke_aggregation.json').read_text())
    if aggregation['summary'] or aggregation['smoke_runs_excluded']!=8: raise ValueError('Smoke runs entered full-run comparison')
    atomic_json(output / 'smoke_aggregation.json',aggregation)
    print('Four-model Kaggle save/resume and unchanged local baseline verified.',flush=True)


if __name__=='__main__': main()
