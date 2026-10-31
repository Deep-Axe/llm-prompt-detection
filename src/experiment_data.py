"""Prepare run-specific caches without touching the baseline artifacts."""
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from large_data import PARTITIONS, rows, word_tokens
from train_large_mlp import prefix
from experiment_state import atomic_json, digest_json
from evaluation import ROOT
from prepare_large_corpus import file_hash


def selected_rows(data, split, limit):
    return rows(data / f'{split}.jsonl') if limit is None else islice(rows(data / f'{split}.jsonl'), limit)


def cache_fingerprint(config):
    representation = 'tfidf' if config['model']=='mlp' else 'encoder' if config['model'] in ('distilbert','deberta') else 'regex'
    settings = {'representation':representation, 'data_hashes':config['data_hashes'],
                'max_tokens':config['max_tokens'], 'smoke_only':config['smoke_only'],
                'vocabulary_source':config.get('vocabulary_source'),
                'limits':[config['smoke_train_rows'],config['smoke_eval_rows']] if config['smoke_only'] else None,
                'feature_batch_size':config['batch_size'] if representation=='tfidf' else None,
                'encoder_source':config['model_settings'].get('model_source') if representation=='encoder' else None,
                'code_hashes':{name:config['code_hashes'][name] for name in ['experiment_data.py','large_data.py','train_large_mlp.py']}}
    return digest_json(settings)



def load_fixed_vocabulary(path, expected_sha256):
    if file_hash(path)!=expected_sha256:
        raise ValueError('Fixed vocabulary checksum differs')
    vocabulary=json.loads(path.read_text())
    if not vocabulary or len(vocabulary)>20000 or any(not isinstance(k,str) or type(v) is not int for k,v in vocabulary.items()):
        raise ValueError('Invalid vocabulary schema')
    if sorted(vocabulary.values())!=list(range(2,len(vocabulary)+2)):
        raise ValueError('Vocabulary IDs must be contiguous, reserving zero and one')
    return vocabulary


def prepare(data, folder, manifest, config, encoder, reuse_cache=None):
    started=time.perf_counter()
    cache = folder / 'cache'
    marker = cache / 'complete.json'
    fingerprint = cache_fingerprint(config)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Cache belongs to another data representation')
        return saved | {'reused':True}
    if reuse_cache is not None:
        reuse_cache=Path(reuse_cache)
        saved=json.loads((reuse_cache / 'complete.json').read_text())
        if saved['fingerprint']!=fingerprint:
            raise ValueError('Requested cache uses different data or preprocessing')
        def link_or_copy(source,destination):
            destination=Path(destination)
            if destination.exists(): destination.unlink()
            try: os.link(source,destination)
            except OSError: shutil.copy2(source,destination)
        shutil.copytree(reuse_cache,cache,dirs_exist_ok=True,copy_function=link_or_copy,
                        ignore=shutil.ignore_patterns('complete.json'))
        atomic_json(marker,saved)
        print(f'Reused preprocessing cache: {reuse_cache}',flush=True)
        return saved | {'reused':True}
    cache.mkdir(parents=True, exist_ok=True)
    limits = {s: min(manifest['partitions'][s]['rows'], config['smoke_train_rows'] if s=='train'
                     else config['smoke_eval_rows']) if config['smoke_only'] else None for s in PARTITIONS}
    info = {'fingerprint':fingerprint, 'counts':{}, 'vocabulary':None}
    if config['model'] == 'mlp':
        import joblib
        vectorizers = [HashingVectorizer(n_features=16384, ngram_range=(1,2), alternate_sign=False, norm=None, dtype=np.float32),
                       HashingVectorizer(n_features=16384, analyzer='char', ngram_range=(3,5), alternate_sign=False, norm=None, dtype=np.float32)]
        df = np.zeros(32768,dtype=np.int64)
        for split in PARTITIONS:
            print(f'Preparing TF-IDF features: {split}',flush=True)
            metadata = []; chunk = []; index = 0
            def save_chunk(chunk, index):
                texts = [prefix(r['text'],config['max_tokens']) for r in chunk]
                x = sparse.hstack([v.transform(texts) for v in vectorizers],format='csr',dtype=np.float32)
                sparse.save_npz(cache / f'{split}_{index:05d}.npz',x)
                np.save(cache / f'{split}_{index:05d}_labels.npy', np.array([r['label'] for r in chunk]))
                if split=='train':
                    presence=x.copy(); presence.data[:]=1
                    df[:] += np.asarray(presence.sum(axis=0)).ravel().astype(np.int64)
            for r in selected_rows(data,split,limits[split]):
                metadata.append({k:r[k] for k in ('id','label','family','group')})
                chunk.append(r)
                if len(chunk)==config['batch_size']:
                    save_chunk(chunk,index); index+=1; chunk=[]
            if chunk: save_chunk(chunk,index); index+=1
            info['counts'][split]=len(metadata)
            info.setdefault('feature_batches',{})[split]=index
            (cache / f'{split}_metadata.json').write_text(json.dumps(metadata)+'\n')
        tfidf=TfidfTransformer(sublinear_tf=True)
        tfidf.idf_=(np.log((1+info['counts']['train'])/(1+df))+1).astype(np.float32)
        joblib.dump({'vectorizers':vectorizers,'tfidf':tfidf},cache / 'features.joblib')
    else:
        transformer=config['model'] in ('distilbert','deberta')
        if transformer:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True)
        else:
            source=config.get('vocabulary_source')
            if source:
                if source['train_partition_sha256']!=manifest['partitions']['train']['sha256']:
                    raise ValueError('Fixed vocabulary was fitted on a different training partition')
                vocabulary=load_fixed_vocabulary(ROOT/source['path'],source['sha256'])
            else:
                counts=Counter()
                for r in selected_rows(data,'train',limits['train']):
                    counts.update(word_tokens(r['text'],config['max_tokens']))
                vocabulary={t:i+2 for i,(t,_) in enumerate(sorted(counts.items(),key=lambda v:(-v[1],v[0]))[:20000])}
            info['vocabulary']=vocabulary
            atomic_json(cache / 'vocabulary.json',vocabulary)
        for split in PARTITIONS:
            print(f'Encoding token sequences: {split}',flush=True)
            n=limits[split] or manifest['partitions'][split]['rows']
            ids=np.lib.format.open_memmap(cache / f'{split}_ids.npy',mode='w+',dtype='int32',shape=(n,config['max_tokens']))
            lengths=np.lib.format.open_memmap(cache / f'{split}_lengths.npy',mode='w+',dtype='int32',shape=(n,))
            labels=np.lib.format.open_memmap(cache / f'{split}_labels.npy',mode='w+',dtype='float32',shape=(n,))
            metadata=[]; offset=0; batch=[]
            def encode(batch):
                nonlocal offset
                size=len(batch)
                if transformer:
                    encoded=tokenizer([r['text'] for r in batch],padding='max_length',truncation=True,
                                      max_length=config['max_tokens'],return_tensors='np')
                    ids[offset:offset+size]=encoded['input_ids']
                    lengths[offset:offset+size]=encoded['attention_mask'].sum(1)
                else:
                    ids[offset:offset+size]=0
                    for i,r in enumerate(batch):
                        tokens=word_tokens(r['text'],config['max_tokens'])
                        ids[offset+i,:len(tokens)]=[vocabulary.get(t,1) for t in tokens]
                        lengths[offset+i]=len(tokens)
                labels[offset:offset+size]=[r['label'] for r in batch]
                metadata.extend({k:r[k] for k in ('id','label','family','group')} for r in batch)
                offset+=size
            for r in selected_rows(data,split,limits[split]):
                batch.append(r)
                if len(batch)==128: encode(batch); batch=[]
            if batch: encode(batch)
            if offset != n: raise ValueError('Unexpected row count')
            ids.flush(); lengths.flush(); labels.flush()
            info['counts'][split]=n
            (cache / f'{split}_metadata.json').write_text(json.dumps(metadata)+'\n')
    if len({r['label'] for r in json.loads((cache / 'val_metadata.json').read_text())}) != 2:
        raise ValueError('Validation requires both classes for F1 selection')
    info['cache_build_seconds']=time.perf_counter()-started
    atomic_json(marker,info)
    return info | {'reused':False}
