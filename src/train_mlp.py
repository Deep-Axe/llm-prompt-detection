"""Unoptimized TF-IDF + MLP run on the imported, fixed corpus splits.

Run from the repository root: OPENBLAS_NUM_THREADS=1 python3 src/train_mlp.py
No test or held-out rows are used to fit features or model weights.
"""
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.neural_network import MLPClassifier
from evaluation import ROOT, read, metrics, load_groups, hashes, error_records, code_hashes


def main():
    corpus = ROOT/'data/processed/corpus.jsonl'
    groups = load_groups()
    features = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1,2), max_features=8000, min_df=2, sublinear_tf=True)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=8000, min_df=2, sublinear_tf=True))])
    start = time.perf_counter()
    xtrain = features.fit_transform([r['text'] for r in groups['train']])
    feature_seconds = time.perf_counter()-start
    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=20, batch_size=64,
                          random_state=4442, early_stopping=False, tol=0, n_iter_no_change=20)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(xtrain, [r['label'] for r in groups['train']])
    training_seconds = time.perf_counter()-start
    scores = {}
    probabilities = {}
    for name, group in groups.items():
        x = features.transform([r['text'] for r in group])
        probabilities[name] = model.predict_proba(x)[:,1]
        scores[name] = metrics(group, probabilities[name])
    # Wall-clock batch throughput including feature extraction; not single-request latency.
    text = [r['text'] for r in groups['test']]
    model.predict_proba(features.transform(text))
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        model.predict_proba(features.transform(text))
        timings.append((time.perf_counter()-start)/len(text)*1000)
    latencies=[]
    for row in groups['test'][:20]:
        start=time.perf_counter()
        model.predict_proba(features.transform([row['text']]))
        latencies.append((time.perf_counter()-start)*1000)
    report = {'model': 'TF-IDF word+character n-grams + 64-unit MLP',
        'seed': 4442, 'threshold': .5, 'tuning': 'none; fixed 20-epoch budget',
        'feature_count': xtrain.shape[1], 'epochs_run': model.n_iter_,
        'total_trainable_parameters': int(sum(a.size for a in model.coefs_+model.intercepts_)),
        'feature_fit_seconds': feature_seconds, 'training_seconds': training_seconds,
        'single_request_latency_ms_median':float(np.median(latencies)),
        'single_request_latency_ms_p95':float(np.percentile(latencies,95)),
        'latency_sample_size':len(latencies),
        'batch_ms_per_example_including_features': float(np.median(timings)),
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'machine': platform.machine(), 'processor': platform.processor(), 'device':'cpu'},
        'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'code_hashes':code_hashes(['train_mlp.py','evaluation.py']),
        'data_hashes': hashes(),
        'warnings': [str(w.message) for w in caught], 'metrics': scores}
    import joblib
    artifact=ROOT/'artifacts/mlp'
    artifact.mkdir(parents=True,exist_ok=True)
    joblib.dump({'features':features,'model':model},artifact/'checkpoint.joblib')
    (ROOT/'results/mlp_errors.json').write_text(json.dumps({s:error_records(groups[s],probabilities[s]) for s in ['test','heldout','hard_negative']},indent=2)+'\n')
    (ROOT/'results/mlp.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
