"""Analyze saved classifier errors without training or running GPU inference."""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from evaluation import ROOT, metrics
from experiment_state import atomic_json
from large_data import TOKEN, rows
from prepare_large_corpus import file_hash


def validate_predictions(records, archive):
    ids=np.asarray([r['id'] for r in records])
    labels=np.asarray([r['label'] for r in records])
    if len(set(ids))!=len(ids) or not np.isin(labels,[0,1]).all():
        raise ValueError('Evaluation records must have unique IDs and binary labels')
    if not np.array_equal(archive['ids'],ids) or not np.array_equal(archive['labels'],labels):
        raise ValueError('Prediction identifiers or labels differ from evaluation records')
    probabilities=np.asarray(archive['probabilities'],dtype=np.float64)
    if probabilities.shape!=labels.shape or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('Invalid prediction probabilities')
    return probabilities


def length_bin(length, upper_bounds):
    lower=1
    for upper in upper_bounds:
        if length<=upper: return f'{lower}-{upper}'
        lower=upper+1
    return f'>{upper_bounds[-1]}'


def error_masks(labels, probabilities):
    predicted=probabilities>=0.5
    wrong=predicted!=np.asarray(labels)[None,:]
    return {'every_seed':wrong.all(axis=0),'any_seed':wrong.any(axis=0),
            'seed_sensitive':wrong.any(axis=0)&~wrong.all(axis=0)}


def write_csv(path, records):
    if not records: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)


def evaluation_paths(name, condition):
    if name in ('wildjailbreak-test','wildjailbreak-official'):
        split='test' if name.endswith('-test') else 'official_eval'
        folder=ROOT/'data/processed/large'
        manifest=json.loads((folder/'manifest.json').read_text())
        return folder/f'{split}.jsonl',manifest['partitions'][split],split,False
    if name=='wildguardmix-transfer':
        folder=ROOT/'data/evaluation/wildguardmix-transfer'
        manifest=json.loads((folder/'manifest.json').read_text())
        return folder/'test.jsonl',manifest['partitions']['test'],'test',True
    raise ValueError(f'Unknown evaluation: {name}')


def analyze(protocol_path, output, review_path):
    protocol=json.loads(protocol_path.read_text())
    if protocol['threshold']!=0.5: raise ValueError('Analysis requires the recorded 0.5 threshold')
    condition=protocol['training_condition']
    per_seed=[]; consensus=[]; overlaps=[]; reviews=[]; provenance={}
    for evaluation in protocol['evaluations']:
        data,definition,split,external=evaluation_paths(evaluation,condition)
        if file_hash(data)!=definition['sha256']: raise ValueError('Evaluation data differs from its manifest')
        records=list(rows(data)); labels=np.asarray([r['label'] for r in records])
        if len(records)!=definition['rows']: raise ValueError('Evaluation row count differs')
        lengths=[sum(1 for _ in TOKEN.finditer(r['text'])) for r in records]
        buckets=[length_bin(n,protocol['length_upper_bounds']) for n in lengths]
        selections=[('all','all',np.arange(len(records)))]
        for family in sorted({r['family'] for r in records}):
            selections.append(('family',family,np.asarray([i for i,r in enumerate(records) if r['family']==family])))
        for bucket in dict.fromkeys(length_bin(n,protocol['length_upper_bounds']) for n in [1,*[upper+1 for upper in protocol['length_upper_bounds']]]):
            indices=np.asarray([i for i,b in enumerate(buckets) if b==bucket],dtype=int)
            if len(indices): selections.append(('length',bucket,indices))
        stable_errors={}
        provenance[evaluation]={'rows':len(records),'data_sha256':definition['sha256'],
            'label_counts':{'benign':int((labels==0).sum()),'harmful':int(labels.sum())},'runs':[]}
        for model in protocol['models']:
            probabilities=[]
            for seed in protocol['seeds']:
                relative=Path(condition)/model/f'seed_{seed}'
                if external:
                    prediction=ROOT/'artifacts/transfer'/evaluation/relative/'test_predictions.npz'
                    report_path=ROOT/'results/transfer'/evaluation/relative/'result.json'
                else:
                    prediction=ROOT/'artifacts/experiments'/relative/f'{split}_predictions.npz'
                    report_path=ROOT/'results/experiments'/relative/'result.json'
                report=json.loads(report_path.read_text())
                if report['status']!='complete' or report['threshold']!=0.5 or report.get('smoke_only',False):
                    raise ValueError('Analysis requires complete evaluation records')
                with np.load(prediction) as archive:
                    values=validate_predictions(records,archive)
                recomputed=metrics(records,values)
                for metric,value in recomputed.items():
                    recorded=report['metrics'][split][metric]
                    if value is None:
                        if recorded is not None: raise ValueError('Metric availability differs')
                    elif not np.isclose(value,recorded,atol=1e-8,rtol=0): raise ValueError('Saved metrics differ from predictions')
                probabilities.append(values)
                provenance[evaluation]['runs'].append({'model':model,'seed':seed,
                    'report_sha256':file_hash(report_path),'predictions_sha256':file_hash(prediction)})
                for kind,name,indices in selections:
                    subset=[records[i] for i in indices]
                    score=metrics(subset,values[indices])
                    score['false_positive_rate']=1-score['benign_acceptance'] if score['benign_acceptance'] is not None else None
                    score['false_negative_rate']=1-score['recall'] if score['recall'] is not None else None
                    per_seed.append({'evaluation':evaluation,'model':model,'seed':seed,'slice_type':kind,'slice':name,
                        'benign':int((labels[indices]==0).sum()),'harmful':int(labels[indices].sum()),**score})
            probabilities=np.stack(probabilities)
            masks=error_masks(labels,probabilities)
            stable_errors[model]=masks['every_seed']
            consensus.append({'evaluation':evaluation,'model':model,'n':len(records),
                **{f'{name}_{label_name}':int((mask&(labels==label)).sum())
                   for name,mask in masks.items() for label,label_name in [(0,'false_positives'),(1,'false_negatives')]}})
            means=probabilities.mean(axis=0)
            for label in [0,1]:
                candidates=np.flatnonzero(masks['every_seed']&(labels==label))
                ordered=sorted(candidates,key=lambda i:((-means[i] if label==0 else means[i]),records[i]['id']))
                for index in ordered[:10]:
                    record=records[index]
                    reviews.append({'evaluation':evaluation,'model':model,'error':'false_positive' if label==0 else 'false_negative',
                        'id':record['id'],'gold':int(label),'family':record['family'],'regex_tokens':lengths[index],
                        'seeds':protocol['seeds'],'probabilities':probabilities[:,index].tolist(),
                        'mean_harm_probability':float(means[index]),'review_status':'unreviewed','text':record['text']})
        for a,b in itertools.combinations(protocol['models'],2):
            for label,label_name in [(0,'benign'),(1,'harmful')]:
                left=stable_errors[a]&(labels==label);right=stable_errors[b]&(labels==label)
                union=int((left|right).sum());intersection=int((left&right).sum())
                overlaps.append({'evaluation':evaluation,'label':label_name,'model_a':a,'model_b':b,
                    'shared_errors_every_seed':intersection,'union_errors_every_seed':union,
                    'jaccard':intersection/union if union else None})
        shared=np.logical_and.reduce(list(stable_errors.values()))
        provenance[evaluation]['all_models_wrong_every_seed']={
            'false_positives':int((shared&(labels==0)).sum()),'false_negatives':int((shared&(labels==1)).sum())}
    grouped=defaultdict(list)
    for row in per_seed:
        grouped[(row['evaluation'],row['model'],row['slice_type'],row['slice'])].append(row)
    summaries=[]
    names=['accuracy','f1','recall','benign_acceptance','false_positive_rate','false_negative_rate','mcc',
           'false_positives','false_negatives','brier_score']
    for key,runs in grouped.items():
        if sorted(r['seed'] for r in runs)!=protocol['seeds']: raise ValueError('Missing or duplicate seed')
        row=dict(zip(['evaluation','model','slice_type','slice'],key))
        row.update(n=runs[0]['n'],benign=runs[0]['benign'],harmful=runs[0]['harmful'],seed_count=len(runs))
        for name in names:
            values=[r[name] for r in runs if r[name] is not None]
            row[name+'_mean']=float(np.mean(values)) if values else None
            row[name+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summaries.append(row)
    review_exists=review_path.exists()
    if review_exists:
        with review_path.open() as handle:
            existing=[json.loads(line) for line in handle]
        def review_keys(records):
            return {(r['evaluation'],r['model'],r['error'],r['id']) for r in records}
        if review_keys(existing)!=review_keys(reviews):
            raise ValueError('Existing review samples differ; choose a new review output to preserve annotations')
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'summary.json',{'protocol':protocol,'protocol_sha256':file_hash(protocol_path),
        'analysis_code_sha256':file_hash(Path(__file__)),'metrics_code_sha256':file_hash(ROOT/'src/evaluation.py'),
        'sources':provenance,'slice_summaries':summaries,'stable_error_counts':consensus,
        'limits':['Three-seed stability is descriptive, not a statistical confidence interval.',
                  'Length buckets use regex tokens; DistilBERT truncates WordPiece tokens.',
                  'Cross-source distributions and labels differ; no scores were used for model selection.',
                  'High-confidence error samples require manual review; labels are not independently adjudicated.']})
    write_csv(output/'slices.csv',summaries)
    write_csv(output/'per_seed.csv',per_seed)
    write_csv(output/'stable_errors.csv',consensus)
    write_csv(output/'error_overlap.csv',overlaps)
    review_path.parent.mkdir(parents=True,exist_ok=True)
    if not review_exists:
        with review_path.open('w') as handle:
            for record in reviews: handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps({'evaluations':list(provenance),'slices':len(summaries),'private_review_samples':len(reviews),'existing_review_preserved':review_exists,
                      'shared_stable_errors':{k:v['all_models_wrong_every_seed'] for k,v in provenance.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/error_analysis.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/error_analysis/wildjailbreak-100k')
    parser.add_argument('--review-output',type=Path,default=ROOT/'traige/error_review/wildjailbreak-100k.jsonl')
    args=parser.parse_args()
    analyze(args.protocol,args.output,args.review_output)
