"""Freeze the completed baseline and the protocol for isolated Part C runs."""
import argparse
import json
from pathlib import Path
import subprocess
from huggingface_hub import HfApi
from evaluation import ROOT
from experiment_state import atomic_json
from prepare_large_corpus import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'configs/part_c_protocol.json')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Protocol already exists; use a new filename to define another version')
    baseline = ROOT / 'results/large'
    reports = {n: json.loads((baseline / f'{n}.json').read_text()) for n in ['mlp','cnn','bilstm','distilbert']}
    source = json.loads((ROOT / 'data/models/distilbert/source.json').read_text())
    inventory = {str(p.relative_to(ROOT)): file_hash(p) for p in baseline.iterdir() if p.is_file()}
    checkpoints = json.loads((baseline / 'checkpoint_manifest.json').read_text())
    for record in checkpoints.values():
        for filename in record['files']:
            path = ROOT / record['location'] / filename
            inventory[str(path.relative_to(ROOT))] = file_hash(path)
    models = {
        'mlp': {'epoch_cap':10, 'batch_size':256, 'learning_rate':.001, 'hidden_units':64,
                'features':32768, 'representation':'hashed word 1-2 / character 3-5 TF-IDF'},
        'cnn': {'epoch_cap':10, 'batch_size':64, 'learning_rate':.001,
                'embedding_dim':64, 'filters':64, 'widths':[3,4,5]},
        'bilstm': {'epoch_cap':10, 'batch_size':64, 'learning_rate':.001,
                   'embedding_dim':64, 'hidden_size':64, 'pooling':'mean and max'},
        'distilbert': {'epoch_cap':5, 'batch_size':32, 'learning_rate':2e-5,
                       'weight_decay':.01, 'warmup_fraction':.1, 'model_source':source},
        'deberta': {'epoch_cap':5, 'batch_size':16, 'learning_rate':2e-5,
                    'weight_decay':.01, 'warmup_fraction':.1,
                    'model_source':{'repository':'microsoft/deberta-v3-base',
                                    'revision':HfApi().model_info('microsoft/deberta-v3-base').sha}},
    }
    protocol = {'schema_version':1, 'name':'part-c-v1',
        'baseline_results_commit':'a73b606',
        'repository_commit_at_freeze':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'seeds':[4442,4443,4444], 'threshold':.5, 'selection':'highest validation F1',
        'patience':2, 'min_delta':0., 'max_tokens':256, 'vocabulary_limit':20000,
        'gradient_accumulation':1, 'gradient_clip':1., 'mixed_precision':'CUDA encoder runs only',
        'gradient_checkpointing':'pretrained encoders',
        'models':models,
        'conditions':{'wildjailbreak-100k':json.loads((baseline / 'data_manifest.json').read_text())},
        'baseline_reports':reports, 'baseline_inventory':inventory,
        'smoke_policy':{'purpose':'save/resume verification only; excluded from experiment rankings',
                        'train_rows':512,'evaluation_rows_per_partition':128,'epoch_cap':2},
        'resume_policy':'same configuration, code hashes, data hashes, and protocol; optimizer-update boundaries',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, protocol)
    print(f'Frozen protocol: {args.output}')


if __name__ == '__main__':
    main()
