"""Run repeated-seed training with bounded device concurrency and reusable caches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluation import ROOT
from experiment_state import atomic_json
from prepare_large_corpus import file_hash


def run_sweep(tasks, protocol_path, data, encoder, run_root, result_root, log_root, gpu_ids=(), status_path=None):
    protocol=json.loads(protocol_path.read_text())
    log_root.mkdir(parents=True,exist_ok=True)
    status_path=status_path or log_root / 'training_status.json'
    active={}; completed=[]; failed=[]; caches={}
    pending=list(tasks)
    def publish():
        atomic_json(status_path,{'completed':completed,'failed':failed,'pending':pending,
            'running':[{'model':job['task']['model'],'seed':job['task']['seed'],
                        'device':job['task']['device'],'gpu':slot,'log':str(job['log'])} for slot,job in active.items()]})
    def paths(task):
        relative=Path('wildjailbreak-100k') / task['model'] / f"seed_{task['seed']}"
        return run_root / relative,result_root / relative
    while pending or active:
        for slot,job in list(active.items()):
            code=job['process'].poll()
            if code is None: continue
            job['handle'].close()
            task=job['task']; artifact,output=paths(task)
            if code==0 and (output / 'result.json').exists():
                report=json.loads((output / 'result.json').read_text())
                if report['smoke_only'] or report['train_rows']!=100000 or report['status']!='complete':
                    raise ValueError('Expected complete 100000-row training result')
                if report['protocol_sha256']!=file_hash(protocol_path): raise ValueError('Result used a different protocol')
                completed.append(task | {'epochs_run':report['epochs_run'],'test_f1':report['metrics']['test']['f1']})
                print(f"Completed {task['model']} seed={task['seed']}",flush=True)
            else:
                failed.append(task | {'exit_code':code,'log':str(job['log'])})
                print(f"Failed {task['model']} seed={task['seed']} exit={code}",flush=True)
            del active[slot]
            publish()
        slots=list(gpu_ids) if gpu_ids else ['local']
        for slot in slots:
            if slot in active or not pending: continue
            task=pending.pop(0)
            if task['seed'] not in protocol['seeds']: raise ValueError('Unknown seed')
            artifact,output=paths(task)
            if (output / 'result.json').exists():
                report=json.loads((output / 'result.json').read_text())
                if report['protocol_sha256']!=file_hash(protocol_path): raise ValueError('Existing result used a different protocol')
                completed.append(task | {'already_complete':True})
                publish(); continue
            command=[sys.executable,str(ROOT / 'src/train_experiment.py'),task['model'],
                '--protocol',str(protocol_path),'--seed',str(task['seed']),'--device',task['device'],
                '--data',str(data),'--encoder',str(encoder),'--run-root',str(run_root),'--result-root',str(result_root)]
            env=os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            if gpu_ids: env['CUDA_VISIBLE_DEVICES']=str(slot)
            cache=caches.get(task['model'])
            if cache is None:
                # Complete preprocessing once before launching other seeds on this dataset.
                prep_log=log_root / f"{task['model']}_prepare.log"
                with prep_log.open('a') as handle:
                    subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
                cache=artifact / 'cache'; caches[task['model']]=cache
            if cache != artifact / 'cache': command+=['--reuse-cache',str(cache)]
            checkpoint=artifact / ('last.joblib' if task['model']=='mlp' else 'last.pt')
            if checkpoint.exists(): command+=['--resume']
            log=log_root / f"{task['model']}_seed_{task['seed']}.log"
            handle=log.open('a')
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[slot]={'task':task,'process':process,'handle':handle,'log':log}
            print(f"Started {task['model']} seed={task['seed']} device={task['device']} gpu={slot}",flush=True)
            publish()
        if active: time.sleep(5)
    publish()
    if failed: raise RuntimeError(f'{len(failed)} training runs failed; inspect training_status.json')
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,help='JSON list of model, seed, and device settings')
    p.add_argument('--protocol',type=Path,default=ROOT / 'configs/training_protocol.json')
    p.add_argument('--data',type=Path,default=ROOT / 'data/processed/large')
    p.add_argument('--encoder',type=Path,default=ROOT / 'data/models/distilbert')
    p.add_argument('--run-root',type=Path,default=ROOT / 'artifacts/experiments')
    p.add_argument('--result-root',type=Path,default=ROOT / 'results/experiments')
    p.add_argument('--log-root',type=Path,default=ROOT / 'artifacts/training_logs')
    p.add_argument('--gpus',nargs='*',default=[])
    args=p.parse_args()
    tasks=json.loads(args.tasks.read_text()) if args.tasks else [
        {'model':model,'seed':seed,'device':'cpu' if model=='mlp' else 'cuda'}
        for model in ['mlp','cnn'] for seed in [4442,4443,4444]]
    run_sweep(tasks,args.protocol,args.data,args.encoder,args.run_root,args.result_root,args.log_root,args.gpus)


if __name__=='__main__': main()
