"""Compare uninterrupted and interrupted/resumed runs in separate directories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from evaluation import ROOT
from experiment_state import atomic_json
from prepare_large_corpus import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['mlp','cnn','bilstm','distilbert'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--root',type=Path,default=ROOT / 'artifacts/resume_verification')
    p.add_argument('--protocol',type=Path,default=ROOT / 'configs/training_protocol.json')
    p.add_argument('--condition',default='wildjailbreak-100k')
    args=p.parse_args()
    protocol=json.loads(args.protocol.read_text())
    baseline_before={path:file_hash(ROOT / path) for path in protocol.get('baseline_inventory',{}) if (ROOT / path).exists()}
    if any(sha!=protocol.get('baseline_inventory',{})[p] for p,sha in baseline_before.items()): raise ValueError('Baseline already differs from frozen inventory')
    report={'purpose':'bounded save/resume smoke verification; not a full training comparison',
            'protocol_sha256':file_hash(args.protocol),'models':{}}
    for name in args.models:
        base=[sys.executable,str(ROOT / 'src/train_experiment.py'),name,'--protocol',str(args.protocol),
              '--condition',args.condition,'--smoke','--device','cpu' if name=='mlp' else args.device,'--seed','4442','--checkpoint-every','100']
        def command(lane,extra=()):
            return base+['--run-root',str(args.root / lane / 'artifacts'),'--result-root',str(args.root / lane / 'results')]+list(extra)
        subprocess.run(command('reference'),cwd=ROOT,check=True)
        paused=subprocess.run(command('resumed',['--interrupt-after-updates','1']),cwd=ROOT)
        if paused.returncode!=75: raise ValueError(f'Expected controlled interruption for {name}: {paused.returncode}')
        relative=Path('smoke') / args.condition / name / 'seed_4442'
        checkpoint=args.root / 'resumed/artifacts' / relative / ('last.joblib' if name=='mlp' else 'last.pt')
        if name=='mlp':
            import joblib
            paused_state=joblib.load(checkpoint)
            optimizer_saved=hasattr(paused_state['model'],'_optimizer')
            scheduler_saved=scaler_saved=None
        else:
            paused_state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            optimizer_saved=bool(paused_state['optimizer']['state'])
            scheduler_saved=bool(paused_state['scheduler'])
            scaler_saved=bool(paused_state['scaler']) if name in ('distilbert','deberta') and args.device.startswith('cuda') else None
        if not optimizer_saved: raise ValueError('Optimizer state was not saved')
        if not all(k in paused_state['rng'] for k in ['python','numpy','torch','cuda']): raise ValueError('Missing RNG state')
        subprocess.run(command('resumed',['--resume']),cwd=ROOT,check=True)
        left=json.loads((args.root / 'reference/results' / relative / 'result.json').read_text())
        right=json.loads((args.root / 'resumed/results' / relative / 'result.json').read_text())
        for key in ['epochs_run','chosen_epoch','optimizer_updates','training_rows_seen','early_stopped']:
            if left[key]!=right[key]: raise ValueError(f'Resume mismatch {name}/{key}')
        if right['train_rows']!=protocol['smoke_policy']['train_rows'] or not right['smoke_only']:
            raise ValueError('Smoke run mislabeled')
        difference={}
        for split in ['train','val','test','official_eval']:
            with np.load(args.root / 'reference/artifacts' / relative / f'{split}_predictions.npz') as a, np.load(args.root / 'resumed/artifacts' / relative / f'{split}_predictions.npz') as b:
                if not np.array_equal(a['ids'],b['ids']) or not np.array_equal(a['labels'],b['labels']): raise ValueError('Prediction rows differ')
                gap=float(np.max(np.abs(a['probabilities']-b['probabilities'])))
                if not np.allclose(a['probabilities'],b['probabilities'],rtol=0,atol=1e-6): raise ValueError(f'Predictions differ: {name}/{split}/{gap}')
                difference[split]=gap
        for a,b in zip(left['history'],right['history']):
            if abs(a['train_loss']-b['train_loss'])>1e-6: raise ValueError('Training loss changed across resume')
        if name=='mlp':
            a=joblib.load(args.root / 'reference/artifacts' / relative / 'best.joblib')
            b=joblib.load(args.root / 'resumed/artifacts' / relative / 'best.joblib')
            weight_gap=max(float(np.max(np.abs(x-y))) for x,y in zip(a.coefs_+a.intercepts_,b.coefs_+b.intercepts_))
        else:
            a=torch.load(args.root / 'reference/artifacts' / relative / 'best.pt',map_location='cpu',weights_only=True)
            b=torch.load(args.root / 'resumed/artifacts' / relative / 'best.pt',map_location='cpu',weights_only=True)
            weight_gap=max(float((a[k]-b[k]).abs().max()) for k in a)
        if weight_gap>1e-6: raise ValueError(f'Resumed weights differ: {name}/{weight_gap}')
        report['models'][name]={'passed':True,'seed':4442,'smoke_only':True,'train_rows':right['train_rows'],
            'epochs_run':right['epochs_run'],'interrupted_after_updates':paused_state['state']['optimizer_updates'],
            'resumed_mid_epoch':paused_state['state']['epoch']==0 and paused_state['state']['cursor']>0,
            'optimizer_state_verified':optimizer_saved,'scheduler_state_verified':scheduler_saved,
            'amp_state_verified':scaler_saved,'rng_state_verified':True,'max_weight_difference':weight_gap,
            'max_probability_differences':difference,'gpu':right['environment']['gpu'],
            'encoder_gradient_verified':right['encoder_gradient_verified'],'encoder_weight_changed':right['encoder_weight_changed']}
        atomic_json(args.root / 'resume_verification.json',report)
        print(f'PASS {name}: max_weight_difference={weight_gap}',flush=True)
    baseline_after={path:file_hash(ROOT / path) for path in baseline_before}
    if baseline_before!=baseline_after: raise ValueError('Baseline files changed')
    report.update(status='passed',baseline_files_checked=len(baseline_before),baseline_unchanged=True)
    atomic_json(args.root / 'resume_verification.json',report)
    print(json.dumps({'status':'passed','models':list(report['models']),'baseline_unchanged':True}),flush=True)


if __name__=='__main__':
    main()
