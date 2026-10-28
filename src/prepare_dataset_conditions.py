"""Freeze nested training pools and an overlap-filtered external evaluation set."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from audit_data_conditions import build_rows, id_digest
from evaluation import ROOT
from experiment_state import atomic_json
from large_data import PARTITIONS, rows, validate_manifest
from prepare_large_corpus import LABELS, digest, file_hash, upstream_rows


def write_records(path,records):
    totals=Counter(); groups=set(); identifiers=[]
    with path.open('w',encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            totals[f'label_{record["label"]}']+=1; totals[record['family']]+=1
            groups.add(record['group']); identifiers.append(record['id'])
    return {'rows':len(identifiers),'groups':len(groups),'counts':dict(totals),
            'sha256':file_hash(path),'id_sha256':id_digest(identifiers)}


def link_or_copy(source,destination):
    import os
    try: os.link(source,destination)
    except OSError: shutil.copy2(source,destination)


def audit_partitions(folder):
    ids={}; groups={}
    for split in PARTITIONS:
        ids[split]=set(); groups[split]=set()
        for r in rows(folder/f'{split}.jsonl'):
            if r['id']!=digest(r['text']) or r['id'] in ids[split]: raise ValueError('Invalid or repeated prompt ID')
            ids[split].add(r['id']); groups[split].add(r['group'])
    exact={f'{a}/{b}':len(ids[a]&ids[b]) for i,a in enumerate(PARTITIONS) for b in PARTITIONS[i+1:]}
    requests={f'{a}/{b}':len(groups[a]&groups[b]) for i,a in enumerate(PARTITIONS[:3]) for b in PARTITIONS[:3][i+1:]}
    if any(exact.values()) or any(requests.values()): raise ValueError('Partition overlap found')
    return {'exact_prompt_overlap':exact,'main_request_overlap':requests,
            'official_request_overlap':'unavailable: upstream evaluation omits original requests'}


def freeze_pool(raw,baseline,output_root,size,db,expected):
    name=f'wildjailbreak-{size//1000}k' if size%1000==0 else f'wildjailbreak-{size}'
    target=output_root/name
    if target.exists():
        validate_manifest(target)
        return json.loads((target/'manifest.json').read_text())
    staging=output_root/f'.{name}.building'
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    def train_records():
        for (payload,) in db.execute('SELECT payload FROM payloads ORDER BY id LIMIT ?',(size,)):
            yield json.loads(payload)
    record=write_records(staging/'train.jsonl',train_records())
    if record['rows']!=size or record['id_sha256']!=expected['id_sha256']:
        raise ValueError('Materialized pool differs from the source audit')
    baseline_manifest=json.loads((baseline/'manifest.json').read_text())
    manifest={k:v for k,v in baseline_manifest.items() if k!='partitions'}
    manifest.update(condition=name,train_pool_before_cap=183005,partitions={'train':record},
        source_train_rows=size,nested_training_order='normalized prompt ID, ascending',
        label_definition='prompt harmfulness; completion and refusal labels excluded',
        prefix_100k_sha256=file_hash(baseline/'train.jsonl'))
    for split in PARTITIONS[1:]:
        link_or_copy(baseline/f'{split}.jsonl',staging/f'{split}.jsonl')
        manifest['partitions'][split]=baseline_manifest['partitions'][split]
    h=hashlib.sha256()
    with (staging/'train.jsonl').open('rb') as handle:
        for i,line in enumerate(handle):
            if i==100000: break
            h.update(line)
    if h.hexdigest()!=manifest['prefix_100k_sha256']: raise ValueError('Frozen 100k prefix changed')
    audit=audit_partitions(staging)
    manifest['overlap_audit']=audit
    atomic_json(staging/'manifest.json',manifest)
    staging.rename(target)
    public=ROOT/'results/datasets'/name; public.mkdir(parents=True,exist_ok=True)
    atomic_json(public/'manifest.json',manifest)
    atomic_json(public/'overlap_audit.json',audit)
    print(f'Frozen {name}: {size} unique training prompts',flush=True)
    return manifest


def clean_wildguard(records):
    usable={}; conflicts=set(); counts=Counter()
    for r in records:
        counts['raw_rows']+=1
        label=r['prompt_harm_label']
        if label not in ('harmful','unharmful'):
            counts['missing_prompt_harm_label']+=1; continue
        text=r['prompt'] or ''
        if not text.strip(): counts['empty_prompts']+=1; continue
        key=digest(text); mapped=int(label=='harmful')
        if key in conflicts: counts['rows_after_conflict']+=1; continue
        if key in usable:
            if usable[key]['label']!=mapped:
                conflicts.add(key); del usable[key]; counts['conflicting_prompt_ids']+=1
            else: counts['duplicate_rows_same_label']+=1
            continue
        family=('adversarial' if bool(r['adversarial']) else 'vanilla')+('_harmful' if mapped else '_benign')
        usable[key]={'id':key,'group':key,'text':text,'label':mapped,'family':family,'source':'wildguardmix'}
    counts['unique_labeled_prompts']=len(usable)
    return usable,dict(counts)


def parquet_records(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=['prompt','prompt_harm_label','adversarial']):
        yield from batch.to_pylist()


def freeze_wildguard(raw,reference_db,output_root,evaluation_root):
    train,train_counts=clean_wildguard(parquet_records(raw/'train/wildguard_train.parquet'))
    test,test_counts=clean_wildguard(parquet_records(raw/'test/wildguard_test.parquet'))
    source=json.loads((raw/'source.json').read_text())
    for name,sha in source['files_sha256'].items():
        if file_hash(raw/name)!=sha: raise ValueError('WildGuardMix source changed')
    wildjailbreak_ids={row[0] for row in reference_db.execute('SELECT id FROM rows')}
    overlap_test=set(test)&wildjailbreak_ids
    external=[test[key]|{'split':'test'} for key in sorted(set(test)-wildjailbreak_ids)]
    candidate_ids=set(train)-wildjailbreak_ids-set(test)
    candidates=[train[key]|{'split':'train_candidate'} for key in sorted(candidate_ids)]
    target=evaluation_root/'wildguardmix-transfer'; target.mkdir(parents=True,exist_ok=True)
    if (target/'manifest.json').exists():
        frozen=json.loads((target/'manifest.json').read_text())
        if frozen['source']!=source or file_hash(target/'test.jsonl')!=frozen['partitions']['test']['sha256']:
            raise ValueError('Frozen external evaluation changed')
        candidate_folder=output_root/'wildguardmix-filtered'
        candidate=json.loads((candidate_folder/'manifest.json').read_text())
        if file_hash(candidate_folder/'train_candidates.jsonl')!=candidate['partition']['sha256']:
            raise ValueError('Frozen training candidates changed')
        return frozen
    external_stats=write_records(target/'test.jsonl',external)
    candidate_folder=output_root/'wildguardmix-filtered'; candidate_folder.mkdir(parents=True,exist_ok=True)
    candidate_stats=write_records(candidate_folder/'train_candidates.jsonl',candidates)
    audit={'train_source':train_counts,'test_source':test_counts,
        'test_prompts_removed_for_wildjailbreak_overlap':len(overlap_test),
        'train_prompts_removed_for_wildjailbreak_overlap':len(set(train)&wildjailbreak_ids),
        'train_prompts_removed_for_own_test_overlap':len((set(train)-wildjailbreak_ids)&set(test)),
        'train_candidates_vs_external_test_overlap':len(candidate_ids&{r['id'] for r in external}),
        'candidate_train_rows':len(candidates),'external_test_rows':len(external),
        'overlap_scope':'normalized exact prompt matching; original-request and semantic overlap unavailable'}
    manifest={'condition':'wildguardmix-transfer','role':'external evaluation only','source':source,
        'task':'binary prompt harmfulness','label_mapping':{'unharmful':0,'harmful':1},
        'partitions':{'test':external_stats},'audit':audit,
        'selection_policy':'all validation-selected model checkpoints and seeds; fixed threshold 0.5; no tuning on this set',
        'model_inputs':'prompt text only; response text, refusal labels, and adversarial flags excluded',
        'group_definition':'normalized exact prompt ID; original requests not supplied'}
    atomic_json(target/'manifest.json',manifest)
    public=ROOT/'results/datasets/wildguardmix-transfer'; public.mkdir(parents=True,exist_ok=True)
    atomic_json(public/'manifest.json',manifest)
    candidate_manifest={'condition':'wildguardmix-filtered','role':'unused training candidates',
        'source':source,'task':'binary prompt harmfulness','label_mapping':{'unharmful':0,'harmful':1},
        'partition':candidate_stats,'audit':audit,'trained':False}
    atomic_json(candidate_folder/'manifest.json',candidate_manifest)
    atomic_json(public/'training_candidates_manifest.json',candidate_manifest)
    combined={'role':'combined-pool assessment; no combined training performed',
        'wildjailbreak_training_rows':183005,'additional_unique_training_candidates':len(candidates),
        'combined_unique_training_candidates':183005+len(candidates),
        'decision':'Evaluate nested WildJailbreak scaling and external transfer first. A combined study also needs validation coverage for both sources.',
        'deduplication':'All WildJailbreak partitions and the original WildGuard test were excluded from new training candidates.',
        'semantic_overlap_limit':'Only normalized exact overlap is measured.'}
    atomic_json(public/'combined_pool_assessment.json',combined)
    print(json.dumps(audit,indent=2),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'data/raw/large/wildjailbreak')
    p.add_argument('--baseline',type=Path,default=ROOT/'data/processed/large')
    p.add_argument('--wildguard',type=Path,default=ROOT/'data/raw/large/wildguardmix')
    p.add_argument('--output-root',type=Path,default=ROOT/'data/processed/conditions')
    p.add_argument('--evaluation-root',type=Path,default=ROOT/'data/evaluation')
    args=p.parse_args()
    baseline=validate_manifest(args.baseline)
    for name,sha in baseline['raw_file_sha256'].items():
        if file_hash(args.raw/name)!=sha: raise ValueError('WildJailbreak source changed')
    reference,counts=build_rows(args.raw)
    eligible={row[0]:tuple(row[1:]) for row in reference.execute("SELECT id,grp,label,family FROM rows WHERE split='train'")}
    if len(eligible)!=183005: raise ValueError('Eligible pool changed')
    scratch=ROOT/'artifacts/data_condition_build'; scratch.mkdir(parents=True,exist_ok=True)
    dbpath=scratch/'payloads.sqlite'; dbpath.unlink(missing_ok=True)
    db=sqlite3.connect(dbpath)
    db.execute('CREATE TABLE payloads (id TEXT PRIMARY KEY,payload TEXT)')
    for r in upstream_rows(args.raw/'train/train.tsv'):
        if r['id'] not in eligible: continue
        if (r['group'],r['label'],r['family'])!=eligible[r['id']]: continue
        r['split']='train'
        db.execute('INSERT OR IGNORE INTO payloads VALUES (?,?)',(r['id'],json.dumps(r,ensure_ascii=False)))
    db.commit()
    audited=json.loads((ROOT/'results/data_conditions.json').read_text())
    args.output_root.mkdir(parents=True,exist_ok=True);args.evaluation_root.mkdir(parents=True,exist_ok=True)
    for size in [183005,150000]:
        freeze_pool(args.raw,args.baseline,args.output_root,size,db,audited['wildjailbreak_pools']['pools'][str(size)])
    freeze_wildguard(args.wildguard,reference,args.output_root,args.evaluation_root)
    db.close();reference.close();dbpath.unlink()


if __name__=='__main__': main()
