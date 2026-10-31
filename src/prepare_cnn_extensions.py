"""Freeze CNN capacity and context comparisons using the original training vocabulary."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
from evaluation import ROOT
from experiment_state import atomic_json
from large_data import rows, word_tokens, validate_manifest
from prepare_large_corpus import file_hash


def main():
    data=ROOT/'data/processed/large'
    manifest=validate_manifest(data)
    reference=ROOT/'artifacts/experiments/wildjailbreak-100k/cnn/seed_4442/vocabulary.json'
    vocabulary=json.loads(reference.read_text())
    counts=Counter();n=0
    for record in rows(data/'train.jsonl'):
        counts.update(word_tokens(record['text'],256));n+=1
    reconstructed={t:i+2 for i,(t,_) in enumerate(sorted(counts.items(),key=lambda v:(-v[1],v[0]))[:20000])}
    if n!=100000 or reconstructed!=vocabulary: raise ValueError('Original CNN vocabulary is not reproduced by training-only fitting')
    for seed in [4442,4443,4444]:
        other=ROOT/f'artifacts/experiments/wildjailbreak-100k/cnn/seed_{seed}/vocabulary.json'
        if file_hash(other)!=file_hash(reference): raise ValueError('Reference seeds have different vocabularies')
    destination=ROOT/'data/models/cnn/wildjailbreak-100k-vocabulary.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and file_hash(destination)!=file_hash(reference): raise ValueError('Fixed vocabulary already differs')
    if not destination.exists(): shutil.copy2(reference,destination)
    source={'path':str(destination.relative_to(ROOT)),'sha256':file_hash(destination),
        'train_partition_sha256':manifest['partitions']['train']['sha256'],'fit_max_tokens':256,
        'vocabulary_limit':20000,'reference_condition':'wildjailbreak-100k',
        'policy':'Reuse the same training-only vocabulary for capacity and context comparisons'}
    base=json.loads((ROOT/'configs/dataset_scaling.json').read_text())
    base.pop('baseline_inventory',None)
    base['models']={'cnn':deepcopy(base['models']['cnn'])}
    base['models']['cnn'].update(embedding_dim=128,filters=128)
    base['vocabulary_source']=source
    base['reference_protocol_sha256']=file_hash(ROOT/'configs/training_protocol.json')
    base['comparison_policy']='100000 identical training prompts; identical evaluation assignments; fixed vocabulary; width comparison at 256 tokens followed by context comparison with identical wide architecture'
    for length in [256,512]:
        protocol=deepcopy(base)
        condition=f'wildjailbreak-100k-cnn-wide-{length}'
        protocol['name']=condition;protocol['max_tokens']=length
        protocol['conditions']={condition:deepcopy(base['conditions']['wildjailbreak-100k'])}
        protocol['conditions'][condition]['data_directory']='data/processed/large'
        protocol['comparison_axis']='embedding dimension and convolution filters' if length==256 else 'input context length at fixed wide architecture'
        atomic_json(ROOT/f'configs/cnn_wide_{length}.json',protocol)
    folder=ROOT/'results/cnn_extensions';folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'vocabulary_verification.json',{'status':'verified','training_rows':n,
        'source':source,'reconstructed_from_training_only':True,'reference_seed_vocabularies_identical':True,
        'vocabulary_entries':len(vocabulary),'validation_and_test_used_for_fitting':False})
    print(json.dumps({'training_rows':n,'vocabulary_entries':len(vocabulary),'conditions':[256,512]}))


if __name__=='__main__': main()
