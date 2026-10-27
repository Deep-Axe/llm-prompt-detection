"""Summarize classifier metrics across seeds alongside the original baseline."""
import csv
import json
from evaluation import ROOT
from experiment_state import atomic_json
from aggregate_experiments import aggregate

MODELS=['mlp','cnn','bilstm','distilbert']


def summarize():
    folder=ROOT / 'results/experiments'; folder.mkdir(parents=True,exist_ok=True)
    reports=[json.loads(p.read_text()) for p in folder.rglob('result.json')]
    groups=aggregate(reports)
    atomic_json(folder / 'summary.json',{'runs_found':len(reports),'summary':groups})
    comparison=[]
    for model in MODELS:
        baseline=json.loads((ROOT / f'results/large/{model}.json').read_text())
        matching=[g for g in groups if g['model']==model and g['condition']=='wildjailbreak-100k']
        if len(matching)>1: raise ValueError('Select one training configuration for the comparison')
        group=matching[0] if matching else None
        complete=bool(group and group['complete_seed_set'])
        comparison.append({'model':model,'baseline_test_f1':baseline['metrics']['test']['f1'],
            'seed_count':group['runs'] if group else 0,'complete_seed_set':complete,
            'test_f1_mean':group['metrics']['f1']['mean'] if complete else None,
            'test_f1_std':group['metrics']['f1']['std'] if complete else None,
            'test_accuracy_mean':group['metrics']['accuracy']['mean'] if complete else None,
            'test_mcc_mean':group['metrics']['mcc']['mean'] if complete else None,
            'official_harmful_recall_mean':group['metrics']['official_recall']['mean'] if complete else None,
            'official_benign_acceptance_mean':group['metrics']['official_benign_acceptance']['mean'] if complete else None})
    atomic_json(folder / 'comparison.json',comparison)
    with (folder / 'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparison[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(comparison)
    with (folder / 'validation_curves.csv').open('w',newline='') as handle:
        fields=['model','seed','epoch','train_loss','validation_f1','validation_mcc','selected_epoch']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for report in sorted(reports,key=lambda r:(r['model'],r['seed'])):
            if report.get('smoke_only'): continue
            for epoch in report['history']:
                writer.writerow({'model':report['model'],'seed':report['seed'],'epoch':epoch['epoch'],
                    'train_loss':epoch['train_loss'],'validation_f1':epoch['val']['f1'],
                    'validation_mcc':epoch['val']['mcc'],'selected_epoch':report['chosen_epoch']})
    return comparison


if __name__=='__main__':
    print(json.dumps(summarize(),indent=2))
