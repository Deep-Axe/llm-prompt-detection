"""Run CNN inference with the architecture and context saved in its checkpoint bundle."""
import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path
import re

TOKEN=re.compile(r'\w+|[^\w\s]',re.UNICODE)
FILES=['best.pt','vocabulary.json','model_config.json']


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


class CNNClassifier:
    def __init__(self,folder,device='cpu'):
        import torch
        if __package__:
            from .neural_models import SentenceCNN
        else:
            from neural_models import SentenceCNN
        self.folder=Path(folder);self.device=device
        self.config=json.loads((self.folder/'model_config.json').read_text())
        if self.config['architecture']!='cnn' or self.config['threshold']!=0.5:
            raise ValueError('Expected a CNN checkpoint with threshold 0.5')
        for filename,key in [('best.pt','checkpoint_sha256'),('vocabulary.json','vocabulary_sha256')]:
            if sha256(self.folder/filename)!=self.config[key]: raise ValueError(f'Checkpoint bundle checksum mismatch: {filename}')
        self.vocabulary=json.loads((self.folder/'vocabulary.json').read_text())
        if sorted(self.vocabulary.values())!=list(range(2,len(self.vocabulary)+2)):
            raise ValueError('Invalid vocabulary IDs')
        settings=self.config['architecture_settings']
        self.max_tokens=self.config['max_tokens']
        if type(self.max_tokens) is not int or self.max_tokens<max(settings['widths']):
            raise ValueError('Context must accommodate every convolution width')
        self.model=SentenceCNN(len(self.vocabulary)+2,embedding_dim=settings['embedding_dim'],
                              filters=settings['filters'],widths=tuple(settings['widths']))
        self.model.load_state_dict(torch.load(self.folder/'best.pt',map_location='cpu',weights_only=True))
        self.model.to(device).eval()
        self.source={'type':'local_files','folder':str(self.folder)}

    @classmethod
    def from_hub(cls,repo_id,subfolder,device='cpu',revision=None,cache_dir=None,local_files_only=False):
        from huggingface_hub import snapshot_download
        if Path(subfolder).is_absolute() or '..' in Path(subfolder).parts:
            raise ValueError('Expected a relative model subfolder')
        snapshot=Path(snapshot_download(repo_id=repo_id,repo_type='model',revision=revision,
            allow_patterns=[f'{subfolder}/{name}' for name in FILES],cache_dir=cache_dir,
            local_files_only=local_files_only))
        classifier=cls(snapshot/subfolder,device)
        classifier.source={'type':'huggingface_hub','repo_id':repo_id,'revision':snapshot.name,'subfolder':subfolder}
        return classifier

    def probabilities(self,texts):
        import torch
        if not texts: return []
        values=[]
        for text in texts:
            tokens=[match.group() for match in islice(TOKEN.finditer(text.lower()),self.max_tokens)]
            values.append([self.vocabulary.get(token,1) for token in tokens] or [1])
        ids=torch.zeros((len(values),self.max_tokens),dtype=torch.long,device=self.device)
        lengths=torch.tensor([len(v) for v in values],device=self.device)
        for index,row in enumerate(values): ids[index,:len(row)]=torch.tensor(row,device=self.device)
        with torch.inference_mode(): return self.model(ids,lengths).sigmoid().cpu().tolist()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',type=Path)
    source.add_argument('--repo-id')
    parser.add_argument('--subfolder')
    parser.add_argument('--revision')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--text',required=True)
    args=parser.parse_args()
    if args.repo_id:
        if not args.subfolder: parser.error('--subfolder is required for Hub inference')
        classifier=CNNClassifier.from_hub(args.repo_id,args.subfolder,args.device,args.revision)
    else:
        if args.subfolder or args.revision: parser.error('Subfolder and revision apply to Hub inference')
        classifier=CNNClassifier(args.checkpoint,args.device)
    probability=classifier.probabilities([args.text])[0]
    print(json.dumps({'source':classifier.source,'architecture':'cnn',
        'architecture_settings':classifier.config['architecture_settings'],'max_tokens':classifier.max_tokens,
        'harmful_probability':probability,'label':'harmful' if probability>=.5 else 'benign'}))
