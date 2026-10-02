"""Disk-backed hashed word/character TF-IDF + MLP, fit on 100k train rows only."""
import argparse
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from evaluation import ROOT, metrics, code_hashes
from large_data import PARTITIONS, TOKEN, rows, batches, validate_manifest


def prefix(text, limit=256):
    # Keep punctuation and spacing; all models have a 256-token context budget.
    for i, match in enumerate(TOKEN.finditer(text)):
        if i+1 == limit:
            return text[:match.end()]
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/large')
    p.add_argument('--output', type=Path, default=ROOT / 'results/large')
    p.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/large/mlp')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--seed', type=int, default=4442)
    args = p.parse_args()
    manifest = validate_manifest(args.data)
    args.artifacts.mkdir(parents=True, exist_ok=True); args.output.mkdir(parents=True, exist_ok=True)
    vectorizers = [HashingVectorizer(n_features=16384, ngram_range=(1,2), alternate_sign=False, norm=None, dtype=np.float32),
                   HashingVectorizer(n_features=16384, analyzer='char', ngram_range=(3,5), alternate_sign=False, norm=None, dtype=np.float32)]
    def counts(batch):
        texts = [prefix(r['text']) for r in batch]
        return sparse.hstack([v.transform(texts) for v in vectorizers], format='csr', dtype=np.float32)
    start = time.perf_counter(); df = np.zeros(32768, dtype=np.int64); files = []
    for i, batch in enumerate(batches(args.data / 'train.jsonl', 1000)):
        x = counts(batch)
        presence = x.copy(); presence.data[:] = 1
        df += np.asarray(presence.sum(axis=0)).ravel().astype(np.int64)
        path = args.artifacts / f'counts_{i:03d}.npz'; sparse.save_npz(path, x)
        labels = np.array([r['label'] for r in batch], dtype=np.int64)
        np.save(args.artifacts / f'labels_{i:03d}.npy', labels)
        files.append((path, args.artifacts / f'labels_{i:03d}.npy'))
        if i % 10 == 0:
            print(f'MLP feature batch {i+1}/100', flush=True)
    n = manifest['partitions']['train']['rows']
    tfidf = TfidfTransformer(sublinear_tf=True)
    tfidf.idf_ = (np.log((1+n)/(1+df))+1).astype(np.float32)
    feature_seconds = time.perf_counter()-start
    model = MLPClassifier(hidden_layer_sizes=(64,), batch_size=256, random_state=args.seed,
                          alpha=.0001, learning_rate_init=.001, shuffle=True)
    best = -1; history = []; rng = np.random.default_rng(args.seed)
    def predict(split):
        probabilities = []
        for batch in batches(args.data / f'{split}.jsonl', 1000):
            probabilities.append(model.predict_proba(tfidf.transform(counts(batch)))[:,1])
        return np.concatenate(probabilities)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        loss_sum = 0.
        for index in rng.permutation(len(files)):
            xp, yp = files[index]
            x = tfidf.transform(sparse.load_npz(xp)); y = np.load(yp)
            model.partial_fit(x, y, classes=np.array([0,1]))
            loss_sum += model.loss_ * len(y)
        val = predict('val')
        val_rows = [{'label':r['label']} for r in rows(args.data / 'val.jsonl')]
        score = metrics(val_rows, val)
        history.append({'epoch':epoch+1, 'train_loss':loss_sum/n, 'val':score})
        print(f'MLP epoch {epoch+1}/{args.epochs}: validation F1 {score["f1"]:.4f}', flush=True)
        if score['f1'] > best:
            best = score['f1']; chosen = epoch+1
            joblib.dump({'vectorizers':vectorizers, 'tfidf':tfidf, 'model':model, 'max_tokens':256}, args.artifacts / 'checkpoint.joblib')
    training_seconds = time.perf_counter()-start
    model = joblib.load(args.artifacts / 'checkpoint.joblib')['model']
    scores = {}
    for split in PARTITIONS:
        pred = predict(split)
        metadata = [{k:r[k] for k in ('id','label','family')} for r in rows(args.data / f'{split}.jsonl')]
        scores[split] = metrics(metadata, pred)
        for family in sorted(set(r['family'] for r in metadata)):
            ix = [i for i,r in enumerate(metadata) if r['family']==family]
            scores[f'{split}/{family}'] = metrics([metadata[i] for i in ix], pred[ix])
        np.savez_compressed(args.artifacts / f'{split}_predictions.npz', ids=np.array([r['id'] for r in metadata]),
                            labels=np.array([r['label'] for r in metadata]), probabilities=pred)
    result = {'model':'mlp', 'representation':'hashed word 1-2 and character 3-5 TF-IDF, 32768 features; IDF fitted only on train',
              'hashing_limit':'Hash collisions are possible; n-gram vocabulary differs from the small-corpus baseline.',
              'max_tokens':256, 'seed':args.seed, 'train_rows':n, 'epochs_run':args.epochs, 'chosen_epoch':chosen,
              'selection':'highest validation F1; fixed threshold 0.5', 'metrics':scores, 'history':history,
              'feature_seconds':feature_seconds, 'training_and_validation_seconds':training_seconds,
              'total_parameters':sum(a.size for a in model.coefs_+model.intercepts_),
              'trainable_parameters':sum(a.size for a in model.coefs_+model.intercepts_),
              'data_hashes':{s:manifest['partitions'][s]['sha256'] for s in PARTITIONS},
              'code_hashes':code_hashes(['train_large_mlp.py','large_data.py','evaluation.py']),
              'environment':{'python':platform.python_version(), 'device':'cpu'}}
    (args.output / 'mlp.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'model':'mlp','test_f1':scores['test']['f1'], 'official_eval_f1':scores['official_eval']['f1']}, indent=2))


if __name__ == '__main__':
    main()
