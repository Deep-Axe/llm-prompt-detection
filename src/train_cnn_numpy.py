"""Small NumPy sentence CNN with learned embeddings and max-over-time pooling.

A CPU baseline without a pretrained vocabulary or a deep-learning framework.
Three convolution widths (3, 4, 5), 16 filters each, 16-dimensional embeddings.
Fixed five-epoch Adam run, 128 tokens; training-only vocabulary, fixed splits.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from train_mlp import read, metrics

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class SentenceCNN:
    def __init__(self, vocab_size, embedding_dim=16, filters=16, widths=(3,4,5), seed=4442, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.widths = widths
        self.p = {'embedding': rng.normal(0,.1,(vocab_size,embedding_dim)).astype(dtype),
                  'head': rng.normal(0,.1,(len(widths)*filters,)).astype(dtype),
                  'bias': np.zeros(1,dtype=dtype)}
        self.p['embedding'][0] = 0
        for k in widths:
            self.p[f'w{k}'] = rng.normal(0,np.sqrt(2/(k*embedding_dim)),(k,embedding_dim,filters)).astype(dtype)
            self.p[f'b{k}'] = np.zeros(filters,dtype=dtype)
        self.m = {k: np.zeros_like(v) for k,v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k,v in self.p.items()}
        self.step = 0

    def forward(self, ids, lengths):
        emb = self.p['embedding'][ids]
        pooled, cache = [], []
        for k in self.widths:
            # [batch, valid positions, kernel width, embedding dimension]
            windows = np.lib.stride_tricks.sliding_window_view(emb,k,axis=1).transpose(0,1,3,2)
            z = np.einsum('bpkd,kdf->bpf',windows,self.p[f'w{k}'],optimize=True)+self.p[f'b{k}']
            valid = np.arange(z.shape[1])[None,:] < np.maximum(1,lengths-k+1)[:,None]
            z = np.where(valid[:,:,None], z, -np.inf)
            indices = z.argmax(axis=1)
            selected = z[np.arange(len(ids))[:,None], indices, np.arange(z.shape[2])[None,:]]
            pooled.append(np.maximum(selected,0))
            cache.append((windows,indices,selected))
        h = np.concatenate(pooled,axis=1)
        logits = h@self.p['head']+self.p['bias'][0]
        prob = 1/(1+np.exp(-np.clip(logits,-50,50)))
        return prob,(ids,emb,h,cache,logits)

    def loss_grad(self, ids, lengths, y):
        prob,(ids,emb,h,cache,logits) = self.forward(ids,lengths)
        loss = np.mean(np.logaddexp(0,logits)-y*logits)
        dz = (prob-y)/len(y)
        grads = {'head': h.T@dz, 'bias': np.array([dz.sum()],dtype=emb.dtype)}
        dh = dz[:,None]*self.p['head'][None,:]
        demb = np.zeros_like(emb)
        offset = 0
        for k,(windows,indices,selected) in zip(self.widths,cache):
            filters = selected.shape[1]
            dsel = dh[:,offset:offset+filters]*(selected > 0)
            offset += filters
            # Only the maximum position contributes to each filter.
            chosen = windows[np.arange(len(ids))[:,None],indices]
            grads[f'w{k}'] = np.einsum('bfkd,bf->kdf',chosen,dsel,optimize=True)
            grads[f'b{k}'] = dsel.sum(axis=0)
            for j in range(k):
                contribution = dsel[:,:,None]*self.p[f'w{k}'][j].T[None,:,:]
                np.add.at(demb,(np.arange(len(ids))[:,None],indices+j),contribution)
        grads['embedding'] = np.zeros_like(self.p['embedding'])
        np.add.at(grads['embedding'], ids.ravel(), demb.reshape(-1,emb.shape[2]))
        grads['embedding'][0] = 0
        return float(loss), grads

    def update(self, grads, rate=.001):
        self.step += 1
        for key,g in grads.items():
            self.m[key] = .9*self.m[key]+.1*g
            self.v[key] = .999*self.v[key]+.001*g*g
            self.p[key] -= rate*(self.m[key]/(1-.9**self.step))/(np.sqrt(self.v[key]/(1-.999**self.step))+1e-8)
        self.p['embedding'][0] = 0


def encode(rows,vocab,max_tokens=128):
    ids = np.zeros((len(rows),max_tokens),dtype=np.int32)
    lengths = np.zeros(len(rows),dtype=np.int32)
    for i,r in enumerate(rows):
        tokens = TOKEN.findall(r['text'].lower())[:max_tokens]
        lengths[i] = len(tokens)
        ids[i,:len(tokens)] = [vocab.get(t,1) for t in tokens]
    return ids,lengths


def probabilities(model,encoded):
    ids,lengths = encoded
    return np.concatenate([model.forward(ids[i:i+64],lengths[i:i+64])[0] for i in range(0,len(ids),64)])


def main():
    path = ROOT/'data/processed/corpus.jsonl'
    rows = read(path)
    groups = {s:[r for r in rows if r['split']==s] for s in ['train','val','test','heldout']}
    groups['hard_negative'] = read(ROOT/'data/processed/hard_negatives.jsonl')
    groups['test_jbb_harmful'] = [r for r in groups['test'] if r['source']=='jbb_harmful']
    groups['test_dan'] = [r for r in groups['test'] if r['source']=='dan']
    start = time.perf_counter()
    counts = Counter(t for r in groups['train'] for t in TOKEN.findall(r['text'].lower()))
    vocab = {t:i+2 for i,(t,n) in enumerate(sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:10000]) if n>=2}
    # Assign contiguous ids after the frequency filter.
    vocab = {t:i+2 for i,t in enumerate(vocab)}
    encoded = {s:encode(g,vocab) for s,g in groups.items()}
    feature_seconds = time.perf_counter()-start
    model = SentenceCNN(len(vocab)+2)
    rng = np.random.default_rng(4442)
    ids,lengths = encoded['train']
    y = np.array([r['label'] for r in groups['train']],dtype=np.float32)
    epoch_loss = []
    start = time.perf_counter()
    for epoch in range(5):
        order = rng.permutation(len(ids))
        loss_sum = 0
        for i in range(0,len(order),64):
            ix = order[i:i+64]
            loss,g = model.loss_grad(ids[ix],lengths[ix],y[ix])
            model.update(g)
            loss_sum += loss*len(ix)
        epoch_loss.append(loss_sum/len(ids))
        print(f'epoch {epoch+1}: loss {epoch_loss[-1]:.4f}',flush=True)
    training_seconds = time.perf_counter()-start
    scores = {s:metrics(g,probabilities(model,encoded[s])) for s,g in groups.items()}
    probabilities(model,encoded['test'])
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        probabilities(model,encode(groups['test'],vocab))
        timings.append((time.perf_counter()-start)/len(groups['test'])*1000)
    report = {'model':'NumPy 1D sentence CNN', 'seed':4442, 'threshold':.5,
              'tuning':'none; fixed five-epoch budget', 'epochs_run':5,
              'max_tokens':128, 'embedding_dim':16,'filters_per_width':16,'widths':[3,4,5],
              'vocabulary_size_including_pad_unk':len(vocab)+2,
              'total_trainable_parameters':sum(p.size for p in model.p.values()),
              'feature_fit_seconds':feature_seconds,'training_seconds':training_seconds,
              'batch_ms_per_example_including_features':float(np.median(timings)),
              'epoch_train_loss':epoch_loss,'numpy_version':np.__version__,
              'corpus_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':scores}
    (ROOT/'results/cnn_numpy.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
