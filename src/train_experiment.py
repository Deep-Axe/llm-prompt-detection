"""Isolated, resumable training under a frozen training protocol."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import random
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
from scipy import sparse
import torch
from evaluation import ROOT, metrics, code_hashes
from experiment_data import prepare
from experiment_state import atomic_json, atomic_torch, capture_rng, restore_rng, digest_json, update_validation
from large_data import PARTITIONS, validate_manifest
from neural_models import SentenceCNN, BiLSTM
from prepare_large_corpus import file_hash

CODE_FILES=['train_experiment.py','experiment_state.py','experiment_data.py','neural_models.py','large_data.py','evaluation.py','train_large_mlp.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['mlp','cnn','bilstm','distilbert','deberta'])
    parser.add_argument('--protocol',type=Path,default=ROOT / 'configs/training_protocol.json')
    parser.add_argument('--condition',default='wildjailbreak-100k')
    parser.add_argument('--data',type=Path,default=ROOT / 'data/processed/large')
    parser.add_argument('--encoder',type=Path,default=ROOT / 'data/models/distilbert')
    parser.add_argument('--seed',type=int,default=4442)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--run-root',type=Path,default=ROOT / 'artifacts/experiments')
    parser.add_argument('--result-root',type=Path,default=ROOT / 'results/experiments')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Bounded save/resume verification, never a full training result')
    parser.add_argument('--interrupt-after-updates',type=int,default=0,help='Exit 75 after this many optimizer updates')
    parser.add_argument('--checkpoint-every',type=int,default=100)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if args.seed not in protocol['seeds']: parser.error('Seed is outside the frozen protocol')
    if args.checkpoint_every<1 or args.interrupt_after_updates<0: parser.error('Invalid checkpoint interval')
    definition=protocol['models'][args.model]
    manifest=validate_manifest(args.data)
    hashes={s:manifest['partitions'][s]['sha256'] for s in PARTITIONS}
    expected=protocol['conditions'][args.condition]
    if hashes != {s:expected['partitions'][s]['sha256'] for s in PARTITIONS}:
        raise ValueError('Dataset differs from the frozen condition')
    transformer=args.model in ('distilbert','deberta')
    if transformer:
        source=json.loads((args.encoder / 'source.json').read_text())
        expected_source=definition['model_source']
        if any(source[k]!=expected_source[k] for k in ('repository','revision')):
            raise ValueError('Encoder source differs from the protocol')
        for name,sha in source.get('files_sha256',{}).items():
            if file_hash(args.encoder / name)!=sha: raise ValueError(f'Encoder checksum mismatch: {name}')
    config={'model':args.model,'condition':args.condition,'seed':args.seed,'data_hashes':hashes,
        'protocol_sha256':file_hash(args.protocol),'code_hashes':code_hashes(CODE_FILES),
        'model_settings':definition,'max_tokens':protocol['max_tokens'],'batch_size':definition['batch_size'],
        'epoch_cap':protocol['smoke_policy']['epoch_cap'] if args.smoke else definition['epoch_cap'],
        'patience':protocol['patience'],'min_delta':protocol['min_delta'],'threshold':protocol['threshold'],
        'accumulation':protocol['gradient_accumulation'], 'device_type':torch.device(args.device).type,
        'smoke_only':args.smoke,'smoke_train_rows':protocol['smoke_policy']['train_rows'],
        'smoke_eval_rows':protocol['smoke_policy']['evaluation_rows_per_partition']}
    if config['threshold']!=.5: raise ValueError('Metric harness requires the frozen 0.5 threshold')
    config['fingerprint']=digest_json(config)
    relative=Path(args.condition) / args.model / f'seed_{args.seed}'
    if args.smoke: relative=Path('smoke') / relative
    artifact=args.run_root / relative
    output=args.result_root / relative
    artifact.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    last=artifact / ('last.joblib' if args.model=='mlp' else 'last.pt')
    best_path=artifact / ('best.joblib' if args.model=='mlp' else 'best.pt')
    if (last.exists() or (output / 'result.json').exists()) and not args.resume:
        raise FileExistsError('Existing run: choose a new root or pass --resume')
    if args.resume and not last.exists(): raise FileNotFoundError('No resumable checkpoint in this run')
    atomic_json(output / 'config.json',config)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    cache_info=prepare(args.data,artifact,manifest,config,args.encoder)
    cache=artifact / 'cache'
    metadata={s:json.loads((cache / f'{s}_metadata.json').read_text()) for s in PARTITIONS}
    n=cache_info['counts']['train']
    scaler=None; scheduler=None; optimizer=None; layer_probes={}
    if args.model=='mlp':
        from sklearn.neural_network import MLPClassifier
        features=joblib.load(cache / 'features.joblib')
        model=MLPClassifier(hidden_layer_sizes=(64,),batch_size=config['batch_size'],random_state=args.seed,
                            alpha=.0001,learning_rate_init=.001,shuffle=True)
        minibatches=cache_info['feature_batches']['train']
        def feature_batch(split,i):
            x=features['tfidf'].transform(sparse.load_npz(cache / f'{split}_{i:05d}.npz'))
            y=np.load(cache / f'{split}_{i:05d}_labels.npy')
            return x,y
        def predict(split):
            return np.concatenate([model.predict_proba(feature_batch(split,i)[0])[:,1]
                                   for i in range(cache_info['feature_batches'][split])])
    else:
        if transformer:
            from transformers import AutoModelForSequenceClassification
            model=AutoModelForSequenceClassification.from_pretrained(args.encoder,num_labels=2,local_files_only=True)
            model.requires_grad_(True); model.gradient_checkpointing_enable()
            import re
            for name,param in model.base_model.named_parameters():
                found=re.search(r'(?:layer|block)\.(\d+)\.',name)
                if found and 'attention' in name and param.ndim==2:
                    layer_probes.setdefault(int(found[1]),(name,param))
            expected_layers=model.config.num_hidden_layers
            if len(layer_probes)!=expected_layers: raise ValueError('Cannot verify every encoder layer')
        else:
            vocabulary=cache_info['vocabulary']
            model=SentenceCNN(len(vocabulary)+2) if args.model=='cnn' else BiLSTM(len(vocabulary)+2)
        model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=definition['learning_rate'],weight_decay=definition.get('weight_decay',0.))
        arrays={s:tuple(np.load(cache / f'{s}_{kind}.npy',mmap_mode='r') for kind in ['ids','lengths','labels']) for s in PARTITIONS}
        minibatches=math.ceil(n/config['batch_size'])
        total_updates=math.ceil(minibatches/config['accumulation'])*config['epoch_cap']
        warmup=int(total_updates*definition.get('warmup_fraction',0))
        def rate(step):
            if not transformer: return 1.
            if step<warmup: return (step+1)/max(1,warmup)
            return max(0.,(total_updates-step)/max(1,total_updates-warmup))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
        amp=transformer and device.type=='cuda'
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        def forward(split,indices,train=False):
            ids,lengths,labels=arrays[split]
            x=torch.as_tensor(np.array(ids[indices],dtype=np.int64),device=device)
            lengths=torch.as_tensor(np.array(lengths[indices],dtype=np.int64),device=device)
            y=torch.as_tensor(np.array(labels[indices],dtype=np.float32),device=device)
            with torch.autocast(device.type,enabled=amp):
                if transformer:
                    mask=torch.arange(x.shape[1],device=device)[None,:]<lengths[:,None]
                    logits=model(input_ids=x,attention_mask=mask).logits
                    return torch.nn.functional.cross_entropy(logits,y.long()) if train else logits.softmax(-1)[:,1].float().cpu().numpy()
                logits=model(x,lengths)
                return torch.nn.functional.binary_cross_entropy_with_logits(logits,y) if train else logits.sigmoid().float().cpu().numpy()
        def predict(split):
            model.eval()
            with torch.inference_mode():
                return np.concatenate([forward(split,slice(i,i+config['batch_size']))
                    for i in range(0,cache_info['counts'][split],config['batch_size'])])
    state={'epoch':0,'cursor':0,'history':[],'best_score':None,'chosen_epoch':None,'bad_epochs':0,
           'early_stopped':False,'optimizer_updates':0,'update_attempts':0,'epoch_loss_sum':0.,
           'epoch_rows_seen':0,'training_rows_seen':0,'training_seconds':0.,'encoder_gradient_verified':False}
    if transformer:
        state['encoder_probe_name'],probe=next(iter(layer_probes.values()))
        state['encoder_initial_probe']=probe.detach().cpu().clone()
    if args.resume:
        saved=joblib.load(last) if args.model=='mlp' else torch.load(last,map_location='cpu',weights_only=False)
        if saved['fingerprint']!=config['fingerprint']: raise ValueError('Resume configuration or code differs from checkpoint')
        state=saved['state']
        if args.model=='mlp': model=saved['model']
        else:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'])
    elapsed_start=time.perf_counter()
    previous_seconds=state['training_seconds']
    def checkpoint():
        if device.type=='cuda': torch.cuda.synchronize(device)
        state['training_seconds']=previous_seconds+time.perf_counter()-elapsed_start
        payload={'fingerprint':config['fingerprint'],'config':config,'state':state,'rng':capture_rng()}
        if args.model=='mlp':
            payload['model']=model
            temp=last.with_suffix('.tmp'); joblib.dump(payload,temp); os.replace(temp,last)
        else:
            payload.update(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict())
            atomic_torch(last,payload)
    def publish_status(status):
        atomic_json(output / 'status.json',{'status':status,'model':args.model,'seed':args.seed,
            'smoke_only':args.smoke,'epoch':state['epoch'],'next_batch':state['cursor'],
            'optimizer_updates':state['optimizer_updates'],'checkpoint':str(last)})
    for epoch in range(state['epoch'],config['epoch_cap']):
        if state['early_stopped']: break
        if args.model!='mlp': model.train(); optimizer.zero_grad(set_to_none=True)
        order=np.random.default_rng(args.seed+epoch).permutation(minibatches if args.model=='mlp' else n)
        for step in range(state['cursor'],minibatches):
            if args.model=='mlp':
                x,y=feature_batch('train',order[step])
                model.partial_fit(x,y,classes=np.array([0,1]))
                loss=float(model.loss_); count=len(y); updated=True; boundary=True
            else:
                indices=order[step*config['batch_size']:(step+1)*config['batch_size']]
                loss_tensor=forward('train',indices,True)
                window=min(config['accumulation'],minibatches-(step//config['accumulation'])*config['accumulation'])
                scaler.scale(loss_tensor/window).backward()
                loss=float(loss_tensor.detach()); count=len(indices)
                boundary=(step+1)%config['accumulation']==0 or step+1==minibatches
                updated=False
                if boundary:
                    scaler.unscale_(optimizer)
                    if transformer and not state['encoder_gradient_verified']:
                        state['encoder_gradient_verified']=all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            and p.grad.abs().sum().item()>0 for _,p in layer_probes.values())
                    torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
                    old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    updated=scaler.get_scale()>=old_scale
                    if updated: scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            state['epoch_loss_sum']+=loss*count; state['epoch_rows_seen']+=count
            state['training_rows_seen']+=count; state['cursor']=step+1
            if boundary:
                state['update_attempts']+=1; state['optimizer_updates']+=int(updated)
                interrupted=args.interrupt_after_updates and state['optimizer_updates']>=args.interrupt_after_updates
                if interrupted or state['update_attempts']%args.checkpoint_every==0: checkpoint()
                if interrupted:
                    publish_status('interrupted_for_verification'); print('Saved resumable checkpoint',flush=True)
                    return 75
            if step%200==0: print(f'{args.model} epoch={epoch+1} batch={step+1}/{minibatches}',flush=True)
        val=metrics(metadata['val'],predict('val'))
        state['history'].append({'epoch':epoch+1,'train_loss':state['epoch_loss_sum']/state['epoch_rows_seen'],'val':val})
        improved=update_validation(state,val['f1'],config['patience'],config['min_delta'])
        if improved:
            if args.model=='mlp':
                temp=best_path.with_suffix('.tmp'); joblib.dump(model,temp); os.replace(temp,best_path)
            else: atomic_torch(best_path,model.state_dict())
        state.update(epoch=epoch+1,cursor=0,epoch_loss_sum=0.,epoch_rows_seen=0)
        checkpoint(); publish_status('training')
        print(f'{args.model} validation_f1={val["f1"]:.6f} bad_epochs={state["bad_epochs"]}',flush=True)
    if args.model=='mlp': model=joblib.load(best_path)
    else: model.load_state_dict(torch.load(best_path,map_location=device,weights_only=True))
    encoder_changed=None
    if transformer:
        current=dict(model.base_model.named_parameters())[state['encoder_probe_name']].detach().cpu()
        encoder_changed=not torch.equal(state['encoder_initial_probe'],current)
        if not state['encoder_gradient_verified'] or not encoder_changed: raise ValueError('Encoder fine-tuning verification failed')
    scores={}
    for split in PARTITIONS:
        values=predict(split)
        scores[split]=metrics(metadata[split],values)
        for family in sorted({r['family'] for r in metadata[split]}):
            indices=[i for i,r in enumerate(metadata[split]) if r['family']==family]
            scores[f'{split}/{family}']=metrics([metadata[split][i] for i in indices],values[indices])
        np.savez_compressed(artifact / f'{split}_predictions.npz',ids=np.array([r['id'] for r in metadata[split]]),
            labels=np.array([r['label'] for r in metadata[split]]),probabilities=values)
    result={'schema_version':1,'status':'complete','model':args.model,'condition':args.condition,'seed':args.seed,
        'protocol_sha256':config['protocol_sha256'],'run_fingerprint':config['fingerprint'],'config':config,
        'smoke_only':args.smoke,'source_train_rows':manifest['partitions']['train']['rows'],'train_rows':n,
        'training_rows_seen':state['training_rows_seen'],'epochs_run':state['epoch'],'epoch_cap':config['epoch_cap'],
        'early_stopped':state['early_stopped'],'chosen_epoch':state['chosen_epoch'],'history':state['history'],
        'optimizer_updates':state['optimizer_updates'],'update_attempts':state['update_attempts'],
        'training_and_validation_seconds':state['training_seconds'],'threshold':config['threshold'],
        'selection':protocol['selection'],'data_hashes':hashes,'code_hashes':config['code_hashes'],'metrics':scores,
        'encoder_frozen':False if transformer else None,'encoder_gradient_verified':state['encoder_gradient_verified'] if transformer else None,
        'encoder_weight_changed':encoder_changed,'verified_encoder_layers':len(layer_probes) if transformer else None,
        'total_parameters':sum(a.size for a in model.coefs_+model.intercepts_) if args.model=='mlp' else sum(p.numel() for p in model.parameters()),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':str(device),
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None},
        'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        'checkpoint_sha256':file_hash(best_path),'checkpoint_format':'sklearn joblib' if args.model=='mlp' else 'torch state_dict',
        'resumed':args.resume}
    atomic_json(output / 'result.json',result)
    publish_status('complete')
    print(json.dumps({'model':args.model,'smoke_only':args.smoke,'epochs':state['epoch'],'test_f1':scores['test']['f1']}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
