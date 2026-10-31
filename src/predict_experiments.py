"""Load one published experiment checkpoint by condition, architecture, and seed."""
import argparse
import json
from pathlib import Path


def from_hub(repo_id,condition,architecture,seed=4442,device='cpu',revision=None,cache_dir=None):
    from huggingface_hub import snapshot_download
    if __package__:
        from .predict_large_models import PromptClassifier,MODEL_FILES
        from .predict_cnn import CNNClassifier
    else:
        from predict_large_models import PromptClassifier,MODEL_FILES
        from predict_cnn import CNNClassifier
    if architecture not in MODEL_FILES: raise ValueError('Unknown architecture')
    if Path(condition).is_absolute() or '..' in Path(condition).parts or '/' in condition: raise ValueError('Invalid condition')
    subfolder=f'experiments/{condition}/{architecture}/seed_{int(seed)}'
    names=MODEL_FILES[architecture]+(['model_config.json'] if architecture=='cnn' else [])
    snapshot=Path(snapshot_download(repo_id=repo_id,repo_type='model',revision=revision,cache_dir=cache_dir,
        allow_patterns=[f'{subfolder}/{name}' for name in names]))
    folder=snapshot/subfolder
    classifier=CNNClassifier(folder,device) if architecture=='cnn' and (folder/'model_config.json').exists() else PromptClassifier(folder,architecture,device)
    classifier.source={'type':'huggingface_hub','repo_id':repo_id,'revision':snapshot.name,'subfolder':subfolder}
    return classifier


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id',default='DeeAxe/llm-prompt-detection')
    parser.add_argument('--condition',required=True)
    parser.add_argument('--model',choices=['mlp','cnn','bilstm','distilbert'],required=True)
    parser.add_argument('--seed',type=int,default=4442)
    parser.add_argument('--revision')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--text',required=True)
    args=parser.parse_args()
    classifier=from_hub(args.repo_id,args.condition,args.model,args.seed,args.device,args.revision)
    probability=classifier.probabilities([args.text])[0]
    print(json.dumps({'source':classifier.source,'harmful_probability':probability,'label':'harmful' if probability>=.5 else 'benign'}))
