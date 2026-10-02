"""Shared data loading and metrics for all four prompt detectors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve,
    auc, log_loss, brier_score_loss)

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_groups():
    rows = read(ROOT/'data/processed/corpus.jsonl')
    groups = {s:[r for r in rows if r['split']==s] for s in ['train','val','test','heldout']}
    groups['hard_negative'] = read(ROOT/'data/processed/hard_negatives.jsonl')
    groups['test_jbb_harmful'] = [r for r in groups['test'] if r['source']=='jbb_harmful']
    groups['test_dan'] = [r for r in groups['test'] if r['source']=='dan']
    keys = [' '.join(r['text'].split()).casefold() for r in rows+groups['hard_negative']]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate text in supplied partitions')
    return groups


def metrics(rows, probabilities):
    y = np.array([r['label'] for r in rows])
    p = np.asarray(probabilities,dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities')
    pred = p >= .5
    both = len(np.unique(y))==2
    pr_precision, pr_recall, _ = precision_recall_curve(y,p) if both else (None,None,None)
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)) if both else None,
        'recall':float(recall_score(y,pred,zero_division=0)) if y.sum() else None,
        'benign_acceptance':float((~pred[y==0]).mean()) if (y==0).any() else None,
        'f1':float(f1_score(y,pred,zero_division=0)) if both else None,
        'mcc':float(matthews_corrcoef(y,pred)) if both else None,
        'roc_auc':float(roc_auc_score(y,p)) if both else None,
        'pr_auc':float(auc(pr_recall,pr_precision)) if both else None,
        'average_precision':float(average_precision_score(y,p)) if both else None,
        'log_loss':float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
        'brier_score':float(brier_score_loss(y,p)),
        'true_positives':int(((y==1)&pred).sum()),'true_negatives':int(((y==0)&~pred).sum()),
        'false_positives':int(((y==0)&pred).sum()),'false_negatives':int(((y==1)&~pred).sum())}


def hashes():
    return {name:hashlib.sha256((ROOT/'data/processed'/name).read_bytes()).hexdigest()
            for name in ['corpus.jsonl','hard_negatives.jsonl']}


def error_records(rows, probabilities):
    # Keep the full prompt texts out of committed diagnostics.
    return [{'id':r['id'],'source':r['source'],'family':r['family'],'gold':r['label'],
             'probability':float(p),'prediction':int(p>=.5)}
            for r,p in zip(rows,probabilities) if int(p>=.5)!=r['label']]


def code_hashes(names):
    return {name:hashlib.sha256((ROOT/"src"/name).read_bytes()).hexdigest() for name in names}
