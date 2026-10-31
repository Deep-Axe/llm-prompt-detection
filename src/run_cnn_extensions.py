"""Run the fixed CNN width/context comparisons with separate checkpoints and a deadline."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from evaluation import ROOT
from experiment_state import atomic_json
from prepare_large_corpus import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT)
    parser.add_argument('--deadline',required=True,help='ISO time with timezone; no new run is admitted beyond its conservative time budget')
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None: parser.error('Deadline needs a timezone')
    folder=ROOT/'artifacts/training_logs/cnn_extensions';folder.mkdir(parents=True,exist_ok=True)
    state={'completed':[],'deferred':[],'failed':[],'running':None,'deadline':args.deadline}
    durations={};seeds=[4442,4443,4444]
    def publish(): atomic_json(folder/'controller_status.json',state)
    for seed in seeds:
        for context in [256,512]:
            condition=f'wildjailbreak-100k-cnn-wide-{context}'
            relative=Path(condition)/'cnn'/f'seed_{seed}'
            artifact=ROOT/'artifacts/cnn_extensions'/relative
            report_path=ROOT/'results/cnn_extensions'/relative/'result.json'
            protocol=args.source_dir/f'configs/cnn_wide_{context}.json'
            task={'condition':condition,'seed':seed,'max_tokens':context}
            if report_path.exists():
                r=json.loads(report_path.read_text())
                if r['protocol_sha256']!=file_hash(protocol) or r['smoke_only']: raise ValueError('Existing run differs')
                state['completed'].append(task|{'already_complete':True});publish();continue
            # First finish one seed of each context; later seeds use the measured worst-cap estimate.
            estimate=durations.get(context,durations.get(256,300)*(2.5 if context==512 else 1))
            remaining=deadline.timestamp()-time.time()
            if remaining<estimate+600:
                state['deferred'].append(task|{'reason':'deadline budget','estimated_seconds':estimate});publish();continue
            command=[sys.executable,str(args.source_dir/'src/train_experiment.py'),'cnn','--protocol',str(protocol),
                '--condition',condition,'--seed',str(seed),'--device','cuda',
                '--run-root',str(ROOT/'artifacts/cnn_extensions'),'--result-root',str(ROOT/'results/cnn_extensions')]
            if seed!=4442:
                cache=ROOT/'artifacts/cnn_extensions'/condition/'cnn/seed_4442/cache'
                if (cache/'complete.json').exists(): command+=['--reuse-cache',str(cache)]
            if (artifact/'last.pt').exists(): command+=['--resume']
            log=folder/f'{condition}_seed_{seed}.log'
            state['running']=task;publish();start=time.perf_counter()
            print(f'Starting {condition} seed={seed}',flush=True)
            with log.open('a') as handle:
                process=subprocess.Popen(command,cwd=args.source_dir,stdout=handle,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if time.time()>deadline.timestamp()-300:
                        process.terminate();process.wait();state['failed'].append(task|{'reason':'deadline reached; resumable state retained'})
                        state['running']=None;publish();raise RuntimeError('Deadline reached')
                    time.sleep(5)
            if process.returncode!=0:
                state['failed'].append(task|{'exit_code':process.returncode});state['running']=None;publish()
                raise RuntimeError(f'Training failed: {log}')
            r=json.loads(report_path.read_text());elapsed=time.perf_counter()-start
            if r['status']!='complete' or r['train_rows']!=100000 or r['training_rows_seen']!=100000*r['epochs_run']:
                raise ValueError('Expected a complete 100000-row run')
            durations[context]=max(durations.get(context,0),elapsed*10/r['epochs_run']+60)
            state['completed'].append(task|{'wall_seconds':elapsed,'epochs_run':r['epochs_run']})
            state['running']=None;publish()
            subprocess.run([sys.executable,str(ROOT/'src/summarize_cnn_extensions.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'src/evaluate_cnn_extensions.py'),'--device','cuda'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'src/summarize_cnn_extensions.py')],cwd=ROOT,check=True)
    state['status']='complete' if not state['deferred'] else 'partial_deadline_budget';publish()
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__': main()
