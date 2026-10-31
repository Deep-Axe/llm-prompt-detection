"""Compare CNN width and context using common data, vocabulary, and seed sets."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluation import ROOT
from experiment_state import atomic_json


def summarize():
    output=ROOT/'results/cnn_extensions';output.mkdir(parents=True,exist_ok=True)
    conditions=[('wildjailbreak-100k',ROOT/'results/experiments',64,64,256),
        ('wildjailbreak-100k-cnn-wide-256',output,128,128,256),
        ('wildjailbreak-100k-cnn-wide-512',output,128,128,512)]
    comparison=[];curves=[];data_hashes=None
    for condition,folder,embedding,filters,context in conditions:
        reports=[]
        for seed in [4442,4443,4444]:
            path=folder/condition/'cnn'/f'seed_{seed}'/'result.json'
            if not path.exists(): continue
            r=json.loads(path.read_text())
            if r['status']!='complete' or r['smoke_only'] or r['train_rows']!=100000 or r['threshold']!=.5:
                raise ValueError('Expected complete comparable CNN runs')
            if data_hashes is None: data_hashes=r['data_hashes']
            if data_hashes!=r['data_hashes']: raise ValueError('CNN conditions used different partitions')
            if r['config']['max_tokens']!=context: raise ValueError('Unexpected context')
            settings=r['config']['model_settings']
            if settings['embedding_dim']!=embedding or settings['filters']!=filters: raise ValueError('Unexpected CNN architecture')
            reports.append(r)
            for e in r['history']:
                curves.append({'condition':condition,'seed':seed,'epoch':e['epoch'],'train_loss':e['train_loss'],
                    'validation_f1':e['val']['f1'],'selected_epoch':r['chosen_epoch']})
        if not reports: continue
        if len({json.dumps(r['code_hashes'],sort_keys=True) for r in reports})!=1: raise ValueError('Runner versions differ within a condition')
        complete=sorted(r['seed'] for r in reports)==[4442,4443,4444]
        row={'condition':condition,'embedding_dim':embedding,'filters_per_width':filters,'max_tokens':context,
            'train_rows':100000,'parameters':reports[0]['total_parameters'],'seed_count':len(reports),'complete_seed_set':complete}
        values={name:[r['metrics'][split][metric] for r in reports] for name,split,metric in [
            ('test_f1','test','f1'),('test_accuracy','test','accuracy'),('test_mcc','test','mcc'),
            ('official_harmful_recall','official_eval','recall'),('official_benign_acceptance','official_eval','benign_acceptance')]}
        values['training_and_validation_seconds']=[r['training_and_validation_seconds'] for r in reports]
        values['peak_gpu_allocated_mb']=[r['peak_gpu_allocated_mb'] for r in reports]
        transfer_reports=[]
        for r in reports:
            path=(ROOT/'results/transfer/wildguardmix-transfer'/condition/'cnn'/f"seed_{r['seed']}"/'result.json'
                if context==256 and embedding==64 else output/'transfer'/condition/f"seed_{r['seed']}"/'result.json')
            if path.exists(): transfer_reports.append(json.loads(path.read_text()))
        row['transfer_seed_count']=len(transfer_reports)
        for metric in ['f1','recall','benign_acceptance']:
            values['transfer_'+metric]=[r['metrics']['test'][metric] for r in transfer_reports]
        for name,samples in values.items():
            row[name+'_mean']=float(np.mean(samples)) if samples else None
            row[name+'_std']=float(np.std(samples,ddof=1)) if len(samples)>1 else None
        comparison.append(row)
    atomic_json(output/'comparison.json',{'comparison':comparison,
        'interpretation':'Width changes at 256 tokens; context changes from 256 to 512 with the same wide architecture and vocabulary.',
        'limits':'The shared test was already used for the reference. The reference retains its earlier runner revision. Threshold and model settings were fixed before these extensions.'})
    for filename,records in [('comparison.csv',comparison),('validation_curves.csv',curves)]:
        if records:
            with (output/filename).open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    return comparison


if __name__=='__main__': print(json.dumps(summarize(),indent=2))
