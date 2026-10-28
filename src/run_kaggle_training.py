"""Train scheduled prompt classifiers using Kaggle GPUs and fixed input snapshots."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--tasks',required=True)
    p.add_argument('--protocol',default='configs/training_protocol.json')
    args=p.parse_args()
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.46.3',
                    'huggingface-hub==0.36.2','scikit-learn==1.3.2','numpy==1.26.4'],check=True)
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchvision'],check=True)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU allocation is required')
    root=Path('/kaggle/working/experiment'); root.mkdir(parents=True,exist_ok=True)
    inputs=Path('/kaggle/input')
    archives=list(inputs.rglob('experiment.zip'))
    if len(archives)==1:
        with zipfile.ZipFile(archives[0]) as archive: archive.extractall(root)
    else:
        manifests=list(inputs.rglob('data/processed/large/manifest.json'))
        if len(manifests)!=1: raise RuntimeError('Expected one prepared training input')
        shutil.copytree(manifests[0].parents[3],root,dirs_exist_ok=True)
    shutil.copytree(args.source_dir,root,dirs_exist_ok=True)
    tasks=json.loads(args.tasks)
    task_path=root / 'training_tasks.json'; task_path.write_text(json.dumps(tasks))
    gpu_ids=[str(i) for i in range(torch.cuda.device_count())]
    print(json.dumps({'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                      'tasks':tasks}),flush=True)
    success=False
    try:
        subprocess.run([sys.executable,str(root / 'src/run_training_sweep.py'),'--tasks',str(task_path),
            '--protocol',str(root / args.protocol),'--gpus',*gpu_ids,'--log-root','/kaggle/working/training_logs'],cwd=root,check=True)
        success=True
    finally:
        results=root / 'results/experiments'
        with zipfile.ZipFile('/kaggle/working/training_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in results.rglob('*.json'):
                archive.write(path,path.relative_to(results))
        for result_path in results.rglob('result.json'):
            result=json.loads(result_path.read_text())
            model=result['model']; seed=result['seed']
            prefix='' if result['condition']=='wildjailbreak-100k' else result['condition']+'_'
            artifact=root / 'artifacts/experiments' / result_path.parent.relative_to(results)
            with zipfile.ZipFile(f'/kaggle/working/{prefix}{model}_seed_{seed}_predictions.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for path in artifact.glob('*_predictions.npz'): archive.write(path,path.name)
            with zipfile.ZipFile(f'/kaggle/working/{prefix}{model}_seed_{seed}_model.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                if model in ('distilbert','deberta'):
                    for path in (artifact / 'model').iterdir():
                        if path.is_file(): archive.write(path,'model/'+path.name)
                else:
                    for filename in ['best.pt','vocabulary.json']: archive.write(artifact / filename,filename)
            print(f'Exported checkpoint and predictions: {model} seed={seed}',flush=True)
        if success:
            # Inputs and preprocessing can be recreated from the attached datasets.
            for cache in (root / 'artifacts').rglob('cache'):
                if cache.is_dir(): shutil.rmtree(cache)
            for name in ['data','src','configs','results']:
                shutil.rmtree(root / name,ignore_errors=True)
    if not success: raise RuntimeError('Scheduled training did not complete')


if __name__=='__main__': main()
