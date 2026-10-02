"""Fixed-budget GPU/CPU training for CNN, BiLSTM and frozen DistilBERT.

Examples: python src/train_neural.py cnn --device cuda
          python src/train_neural.py bilstm --device cuda
          python src/train_neural.py distilbert --device cuda
"""
import argparse
from collections import Counter
import json
import platform
import random
import re
import subprocess
import time
import numpy as np
import torch
from torch import nn
from evaluation import ROOT,load_groups,metrics,hashes,error_records,code_hashes
from neural_models import SentenceCNN,BiLSTM

TOKEN=re.compile(r'\w+|[^\w\s]',re.UNICODE)


def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def encode(rows,vocab,max_tokens):
    ids=np.zeros((len(rows),max_tokens),dtype=np.int64)
    lengths=np.zeros(len(rows),dtype=np.int64)
    for i,r in enumerate(rows):
        tokens=TOKEN.findall(r['text'].lower())[:max_tokens]
        values=[vocab.get(t,1) for t in tokens] or [1]
        lengths[i]=len(values)
        ids[i,:len(values)]=values
    return torch.from_numpy(ids),torch.from_numpy(lengths)


def setup_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['cnn','bilstm','distilbert'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--seed',type=int,default=4442)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.max_tokens<5: parser.error('Invalid training dimensions')
    setup_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    groups=load_groups()
    unique=groups['train']+groups['val']+groups['test']+groups['heldout']+groups['hard_negative']
    artifact=ROOT/'artifacts'/args.model
    artifact.mkdir(parents=True,exist_ok=True)
    encoder=None; tokenizer=None; vocab=None; model_source=None
    start=time.perf_counter()
    if args.model=='distilbert':
        from transformers import AutoTokenizer, AutoModel
        source=ROOT/'data/models/distilbert'
        model_source=json.loads((source/'source.json').read_text())
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        encoder=AutoModel.from_pretrained(source,local_files_only=True).to(device)
        encoder.requires_grad_(False); encoder.eval()
        features=[]
        # Use small encoder batches; only frozen representations stay in RAM.
        with torch.inference_mode():
            for i in range(0,len(unique),8):
                inputs=tokenizer([r['text'] for r in unique[i:i+8]],padding=True,truncation=True,
                                 max_length=args.max_tokens,return_tensors='pt').to(device)
                features.append(encoder(**inputs).last_hidden_state[:,0,:].cpu())
                if i%400==0: print(f'encoded {min(i+8,len(unique))}/{len(unique)} prompts',flush=True)
        # Clone outside inference mode so autograd may save head inputs.
        all_features=torch.cat(features).clone()
        index={r['id']:i for i,r in enumerate(unique)}
        encoded={s:(all_features[[index[r['id']] for r in rows]],) for s,rows in groups.items()}
        model=nn.Linear(encoder.config.dim,1).to(device)
        trainable=sum(p.numel() for p in model.parameters())
        total=trainable+sum(p.numel() for p in encoder.parameters())
    else:
        counts=Counter(t for r in groups['train'] for t in TOKEN.findall(r['text'].lower()))
        terms=[t for t,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])) if n>=2][:10000]
        vocab={t:i+2 for i,t in enumerate(terms)}
        encoded={s:encode(rows,vocab,args.max_tokens) for s,rows in groups.items()}
        model=(SentenceCNN(len(vocab)+2) if args.model=='cnn' else BiLSTM(len(vocab)+2)).to(device)
        trainable=total=sum(p.numel() for p in model.parameters())
        (artifact/'vocabulary.json').write_text(json.dumps(vocab)+'\n')
    synchronize(device)
    feature_seconds=time.perf_counter()-start
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=nn.BCEWithLogitsLoss()
    generator=torch.Generator().manual_seed(args.seed)
    y=torch.tensor([r['label'] for r in groups['train']],dtype=torch.float32)
    loss_history=[]
    start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order=torch.randperm(len(y),generator=generator)
        loss_sum=0
        for i in range(0,len(y),args.batch_size):
            ix=order[i:i+args.batch_size]
            values=[x[ix].to(device) for x in encoded['train']]
            logits=model(*values).reshape(-1)
            loss=criterion(logits,y[ix].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); loss_sum+=loss.item()*len(ix)
        loss_history.append(loss_sum/len(y))
        print(f'{args.model} epoch {epoch+1}/{args.epochs}: loss {loss_history[-1]:.4f}',flush=True)
    synchronize(device)
    train_seconds=time.perf_counter()-start
    if encoder is not None:
        assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    model.eval()
    def predict(values,batch=32):
        output=[]
        with torch.inference_mode():
            for i in range(0,len(values[0]),batch):
                inputs=[v[i:i+batch].to(device) for v in values]
                output.append(torch.sigmoid(model(*inputs).reshape(-1)).cpu().numpy())
        return np.concatenate(output)
    probabilities={s:predict(x) for s,x in encoded.items()}
    scores={s:metrics(rows,probabilities[s]) for s,rows in groups.items()}
    def predict_text(rows):
        if encoder is None: return predict(encode(rows,vocab,args.max_tokens))
        output=[]
        with torch.inference_mode():
            for i in range(0,len(rows),8):
                inputs=tokenizer([r['text'] for r in rows[i:i+8]],padding=True,truncation=True,
                                 max_length=args.max_tokens,return_tensors='pt').to(device)
                features=encoder(**inputs).last_hidden_state[:,0,:]
                output.extend(torch.sigmoid(model(features).reshape(-1)).cpu().tolist())
        return np.array(output)
    # Full inference measurement, including tokenizer/encoder, never cached-head-only.
    timing_rows=groups['test'][:64]
    predict_text(timing_rows[:8])
    times=[]
    for _ in range(3):
        synchronize(device); start=time.perf_counter(); predict_text(timing_rows); synchronize(device)
        times.append((time.perf_counter()-start)/len(timing_rows)*1000)
    latencies=[]
    for row in timing_rows[:20]:
        synchronize(device); start=time.perf_counter(); predict_text([row]); synchronize(device)
        latencies.append((time.perf_counter()-start)*1000)
    torch.save({'model':args.model,'state_dict':model.state_dict(),'config':vars(args),'model_source':model_source},artifact/'checkpoint.pt')
    result={'model':args.model,'implementation':'PyTorch','seed':args.seed,'threshold':.5,
        'tuning':'none; fixed budget, no validation checkpoint selection','epochs_run':args.epochs,
        'max_tokens':args.max_tokens,'batch_size':args.batch_size,'learning_rate':.001,
        'trainable_parameters':trainable,'total_parameters':total,
        'feature_fit_seconds':feature_seconds,'training_seconds':train_seconds,
        'batch_ms_per_example_including_features':float(np.median(times)),
        'single_request_latency_ms_median':float(np.median(latencies)),
        'single_request_latency_ms_p95':float(np.percentile(latencies,95)),
        'latency_sample_size':len(latencies),'timing_batch_sample_size':len(timing_rows),
        'epoch_train_loss':loss_history,'encoder_frozen':True if encoder is not None else None,
        'encoder_pooling':'first [CLS] hidden state' if encoder is not None else None,
        'model_source':model_source,'data_hashes':hashes(),'metrics':scores,
        'code_hashes':code_hashes(['train_neural.py','neural_models.py','evaluation.py']),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            'cuda_runtime':torch.version.cuda,'cpu_threads':torch.get_num_threads()},
        'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
    ROOT.joinpath('results').mkdir(exist_ok=True)
    (ROOT/f'results/{args.model}.json').write_text(json.dumps(result,indent=2)+'\n')
    errors={s:error_records(groups[s],probabilities[s]) for s in ['test','heldout','hard_negative']}
    (ROOT/f'results/{args.model}_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    print(json.dumps({'model':args.model,'test_f1':scores['test']['f1'],'jbb_recall':scores['test_jbb_harmful']['recall'],
                      'benign_acceptance':scores['hard_negative']['benign_acceptance'],'training_seconds':train_seconds},indent=2))


if __name__=='__main__': main()
