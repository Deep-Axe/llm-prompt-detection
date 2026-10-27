"""Aggregate isolated runs by condition, model, protocol, and seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiment_state import atomic_json, digest_json


def aggregate(reports):
    groups={}; seen=set(); conditions={}
    for r in reports:
        if r.get('status')!='complete' or r.get('smoke_only',False): continue
        identity=(r['protocol_sha256'],r['condition'],r['model'],r['seed'])
        if identity in seen: raise ValueError(f'Duplicate experiment seed: {identity}')
        seen.add(identity)
        condition=(r['protocol_sha256'],r['condition'])
        data_identity=digest_json(r['data_hashes'])
        if condition in conditions and conditions[condition]!=data_identity:
            raise ValueError('Same condition name has different partitions')
        conditions[condition]=data_identity
        key=(r['protocol_sha256'],r['condition'],r['model'])
        groups.setdefault(key,[]).append(r)
    summary=[]
    for (protocol,condition,model),runs in sorted(groups.items()):
        settings={digest_json({k:r['config'][k] for k in ['model_settings','max_tokens','batch_size','epoch_cap','patience','min_delta','threshold']}) for r in runs}
        if len(settings)!=1: raise ValueError('Different configurations were mixed across seeds')
        if len({digest_json(r.get('code_hashes',{})) for r in runs})!=1:
            raise ValueError('Different runner versions were mixed across seeds')
        if any(r['threshold']!=.5 for r in runs): raise ValueError('Unexpected threshold')
        if model in ('distilbert','deberta') and any(not r.get('encoder_gradient_verified') or not r.get('encoder_weight_changed') for r in runs):
            raise ValueError('Encoder fine-tuning is unverified')
        row={'protocol_sha256':protocol,'condition':condition,'model':model,'seeds':sorted(r['seed'] for r in runs),
             'runs':len(runs),'data_hashes':runs[0]['data_hashes'],'metrics':{}}
        for name in ['accuracy','precision','recall','f1','mcc','roc_auc','average_precision','brier_score']:
            values=np.array([r['metrics']['test'][name] for r in runs],dtype=float)
            row['metrics'][name]={'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(values)>1 else None}
        for split,key in [('official_eval','recall'),('official_eval','benign_acceptance')]:
            values=np.array([r['metrics'][split][key] for r in runs],dtype=float)
            row['metrics'][f'official_{key}']={'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(values)>1 else None}
        row['complete_seed_set']=row['seeds']==[4442,4443,4444]
        summary.append(row)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/experiments'))
    p.add_argument('--output',type=Path,default=Path('results/experiments/summary.json'))
    args=p.parse_args()
    files=sorted(args.root.rglob('result.json'))
    reports=[json.loads(path.read_text()) for path in files]
    result={'runs_found':len(reports),'smoke_runs_excluded':sum(bool(r.get('smoke_only')) for r in reports),
            'summary':aggregate(reports)}
    args.output.parent.mkdir(parents=True,exist_ok=True); atomic_json(args.output,result)
    with args.output.with_suffix('.csv').open('w',newline='') as f:
        fields=['condition','model','runs','complete_seed_set','test_f1_mean','test_f1_std']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); writer.writeheader()
        for r in result['summary']:
            writer.writerow({k:r[k] for k in fields[:4]} | {'test_f1_mean':r['metrics']['f1']['mean'],'test_f1_std':r['metrics']['f1']['std']})
    print(json.dumps({'runs_found':len(reports),'groups':len(result['summary']),'smoke_excluded':result['smoke_runs_excluded']}))


if __name__=='__main__':
    main()
