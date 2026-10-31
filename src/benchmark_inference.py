"""Measure CPU request latency on one fixed validation workload in isolated processes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): d.update(block)
    return d.hexdigest()


def worker(name,protocol):
    architecture=name if name in ['mlp','cnn','bilstm','distilbert'] else 'cnn'
    wide=name.startswith('cnn-wide-')
    condition=f'wildjailbreak-100k-{name}' if wide else 'wildjailbreak-100k'
    base=ROOT/('artifacts/cnn_extensions' if wide else 'artifacts/experiments')/condition/architecture/'seed_4442'
    report=ROOT/('results/cnn_extensions' if wide else 'results/experiments')/condition/architecture/'seed_4442/result.json'
    manifest=json.loads((ROOT/'data/processed/large/manifest.json').read_text())
    data=ROOT/'data/processed/large/val.jsonl'
    if sha(data)!=manifest['partitions']['val']['sha256']: raise ValueError('Validation prompts changed')
    selected=sorted(random.Random(protocol['sample_seed']).sample(range(manifest['partitions']['val']['rows']),protocol['requests']))
    wanted=set(selected);texts=[];ids=[]
    with data.open() as handle:
        for i,line in enumerate(handle):
            if i in wanted:
                row=json.loads(line);texts.append(row['text']);ids.append(row['id'])
    if len(texts)!=protocol['requests']: raise ValueError('Incomplete benchmark workload')
    if architecture!='mlp':
        import torch
        torch.set_num_threads(protocol['torch_threads']);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    from predict_large_models import PromptClassifier,MODEL_FILES
    from predict_cnn import CNNClassifier
    folder=base/'model' if architecture=='distilbert' else base
    files=MODEL_FILES[architecture]+(['model_config.json'] if wide else [])
    before=time.perf_counter()
    classifier=CNNClassifier(folder,'cpu') if wide else PromptClassifier(folder,architecture,'cpu')
    load_seconds=time.perf_counter()-before
    for i in range(protocol['warmup_requests']): classifier.probabilities([texts[i%len(texts)]])
    timings=[]
    for text in texts:
        start=time.perf_counter();classifier.probabilities([text]);timings.append((time.perf_counter()-start)*1000)
    r=json.loads(report.read_text())
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),cpu)
    output={'model':name,'architecture':architecture,'condition':condition,'checkpoint_seed':4442,'device':'cpu',
        'max_tokens':r['config']['max_tokens'],'parameters':r['total_parameters'],'requests':len(texts),'batch_size':1,
        'median_ms':statistics.median(timings),'p95_ms':statistics.quantiles(timings,n=100,method='inclusive')[94],
        'mean_ms':statistics.fmean(timings),'checkpoint_load_seconds':load_seconds,
        'process_peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'latencies_ms':timings,'cpu':cpu,'python':platform.python_version(),
        'platform':platform.platform(),'torch_threads':protocol['torch_threads'],'blas_threads':protocol['blas_threads'],
        'workload_id_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'validation_sha256':sha(data),'protocol_sha256':sha(ROOT/'configs/inference_benchmark.json'),
        'benchmark_code_sha256':sha(Path(__file__)),'training_result_sha256':sha(report),
        'checkpoint_files_sha256':{n:sha(folder/n) for n in files},
        'prediction_code_sha256':sha(ROOT/('src/predict_cnn.py' if wide else 'src/predict_large_models.py')),
        'scope':protocol['scope'],'interpretation':protocol['interpretation']}
    folder=ROOT/'results/inference_benchmark';folder.mkdir(exist_ok=True)
    (folder/f'{name}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(f'{name}: median {output["median_ms"]:.2f} ms, p95 {output["p95_ms"]:.2f} ms',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=['mlp','cnn','bilstm','distilbert','cnn-wide-256','cnn-wide-512'])
    parser.add_argument('--worker')
    args=parser.parse_args();protocol=json.loads((ROOT/'configs/inference_benchmark.json').read_text())
    if args.worker: return worker(args.worker,protocol)
    for name in args.models:
        wide=name.startswith('cnn-wide-')
        model='cnn' if wide else name
        condition=f'wildjailbreak-100k-{name}' if wide else 'wildjailbreak-100k'
        report=ROOT/('results/cnn_extensions' if wide else 'results/experiments')/condition/model/'seed_4442/result.json'
        if not report.exists(): continue
        subprocess.run([sys.executable,str(Path(__file__)),'--worker',name],cwd=ROOT,check=True)
    records=[json.loads(p.read_text()) for p in (ROOT/'results/inference_benchmark').glob('*.json')]
    if len({r['workload_id_sha256'] for r in records})>1: raise ValueError('Benchmark workloads differ')
    fields=['model','device','max_tokens','parameters','requests','batch_size','median_ms','p95_ms','mean_ms','process_peak_rss_mb','checkpoint_load_seconds']
    with (ROOT/'results/inference_benchmark/comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in records: writer.writerow({k:r[k] for k in fields})


if __name__=='__main__': main()
