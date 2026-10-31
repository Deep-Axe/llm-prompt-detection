"""Train the fixed CNN context conditions concurrently on Kaggle's two GPUs."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--deadline',required=True)
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None: parser.error('Deadline needs a timezone')
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy==1.26.4','scikit-learn==1.3.2',
                    'transformers==4.46.3','huggingface-hub==0.36.2'],check=True)
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchvision'],check=True)
    import torch
    if torch.cuda.device_count()<2: raise RuntimeError('This schedule requires two GPUs')
    root=Path('/kaggle/working/cnn_experiment');root.mkdir(exist_ok=True)
    archives=list(Path('/kaggle/input').rglob('experiment.zip'))
    if len(archives)==1:
        with zipfile.ZipFile(archives[0]) as archive: archive.extractall(root)
    else:
        manifests=list(Path('/kaggle/input').rglob('data/processed/large/manifest.json'))
        if len(manifests)!=1: raise RuntimeError('Expected one prepared 100000-row training input')
        shutil.copytree(manifests[0].parents[3],root,dirs_exist_ok=True)
    shutil.copytree(args.source_dir,root,dirs_exist_ok=True)
    logs=Path('/kaggle/working/cnn_training_logs');logs.mkdir(exist_ok=True)
    processes=[];handles=[];cutoff=False
    for gpu,context in enumerate([256,512]):
        condition=f'wildjailbreak-100k-cnn-wide-{context}'
        tasks=[{'model':'cnn','condition':condition,'seed':seed,'device':'cuda'} for seed in [4442,4443,4444]]
        path=root/f'tasks_{context}.json';path.write_text(json.dumps(tasks))
        command=[sys.executable,str(root/'src/run_training_sweep.py'),'--tasks',str(path),
            '--protocol',str(root/f'configs/cnn_wide_{context}.json'),
            '--run-root',str(root/'artifacts/cnn_extensions'),'--result-root',str(root/'results/cnn_extensions'),
            '--log-root',str(logs/f'context_{context}')]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        handle=(logs/f'context_{context}_queue.log').open('a');handles.append(handle)
        processes.append(subprocess.Popen(command,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        print(f'Started {condition} on GPU {gpu}',flush=True)
    completed=set()
    while any(p.poll() is None for p in processes):
        for path in (root/'results/cnn_extensions').glob('*/cnn/seed_*/result.json'):
            identity=str(path.relative_to(root/'results/cnn_extensions'))
            if identity not in completed:
                r=json.loads(path.read_text());completed.add(identity)
                print(f"Completed {r['condition']} seed={r['seed']} epochs={r['epochs_run']} validation-selected={r['chosen_epoch']}",flush=True)
        if time.time()>deadline.timestamp()-600:
            cutoff=True
            for process in processes:
                if process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
            break
        time.sleep(5)
    for process in processes: process.wait()
    for handle in handles: handle.close()
    reports=root/'results/cnn_extensions'
    if list(reports.glob('*/cnn/seed_*/result.json')):
        subprocess.run([sys.executable,str(root/'src/evaluate_cnn_extensions.py'),'--device','cuda'],cwd=root,check=True)
    with zipfile.ZipFile('/kaggle/working/cnn_training_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in reports.rglob('*.json'): archive.write(path,path.relative_to(reports))
    count=0
    for path in reports.glob('*/cnn/seed_*/result.json'):
        r=json.loads(path.read_text());relative=path.parent.relative_to(reports)
        artifact=root/'artifacts/cnn_extensions'/relative
        prefix=f"{r['condition']}_seed_{r['seed']}"
        with zipfile.ZipFile(f'/kaggle/working/{prefix}_model.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name in ['best.pt','vocabulary.json','model_config.json']: archive.write(artifact/name,name)
        with zipfile.ZipFile(f'/kaggle/working/{prefix}_predictions.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for prediction in artifact.glob('*_predictions.npz'): archive.write(prediction,prediction.name)
        count+=1
    status={'completed_runs':count,'scheduled_runs':6,'deadline_cutoff':cutoff,
            'exit_codes':[p.returncode for p in processes]}
    Path('/kaggle/working/cnn_execution_summary.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
    if count:
        # Remove reconstructible inputs and caches; selected weights and recovery states remain.
        for cache in (root/'artifacts').rglob('cache'):
            if cache.is_dir(): shutil.rmtree(cache)
        for name in ['data','src','configs','results']: shutil.rmtree(root/name,ignore_errors=True)
    if any(p.returncode for p in processes) and not cutoff: raise RuntimeError('A CNN queue failed; inspect its log')


if __name__=='__main__': main()
