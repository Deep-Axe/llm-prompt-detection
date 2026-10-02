"""Load a published checkpoint and return prompt harmfulness probabilities."""
import argparse
import json
from pathlib import Path
import re

TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)
DEFAULT_REPO = 'DeeAxe/llm-prompt-detection'
MODEL_FILES = {
    'mlp': ['checkpoint.joblib'],
    'cnn': ['best.pt', 'vocabulary.json'],
    'bilstm': ['best.pt', 'vocabulary.json'],
    'distilbert': ['config.json', 'model.safetensors', 'tokenizer.json',
                  'tokenizer_config.json', 'vocab.txt'],
}


def prefix(text, limit=256):
    for i, match in enumerate(TOKEN.finditer(text)):
        if i + 1 == limit:
            return text[:match.end()]
    return text


class PromptClassifier:
    def __init__(self, folder, architecture, device='cpu'):
        if architecture not in MODEL_FILES:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.folder = Path(folder)
        self.architecture = architecture
        self.device = device
        self.source = {'type': 'local_files', 'folder': str(self.folder)}
        if architecture == 'mlp':
            import joblib
            self.checkpoint = joblib.load(self.folder / 'checkpoint.joblib')
        else:
            import torch
            if architecture == 'distilbert':
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.folder, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.folder, local_files_only=True)
            else:
                if __package__:
                    from .neural_models import SentenceCNN, BiLSTM
                else:
                    from neural_models import SentenceCNN, BiLSTM
                self.vocabulary = json.loads((self.folder / 'vocabulary.json').read_text())
                model_type = SentenceCNN if architecture == 'cnn' else BiLSTM
                self.model = model_type(len(self.vocabulary) + 2)
                self.model.load_state_dict(torch.load(self.folder / 'best.pt', map_location='cpu', weights_only=True))
            self.model.to(device).eval()

    @classmethod
    def from_hub(cls, repo_id=DEFAULT_REPO, architecture='distilbert', device='cpu',
                 revision=None, cache_dir=None, local_files_only=False):
        """Fetch one published architecture, cache it, and run inference locally."""
        if architecture not in MODEL_FILES:
            raise ValueError(f'Unknown architecture: {architecture}')
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            repo_id=repo_id, repo_type='model', revision=revision, cache_dir=cache_dir,
            allow_patterns=[f'{architecture}/{name}' for name in MODEL_FILES[architecture]],
            local_files_only=local_files_only,
        ))
        classifier = cls(snapshot / architecture, architecture, device)
        classifier.source = {'type': 'huggingface_hub', 'repo_id': repo_id,
                             'revision': snapshot.name}
        return classifier

    def probabilities(self, texts):
        if not texts:
            return []
        if self.architecture == 'mlp':
            from scipy import sparse
            c = self.checkpoint
            inputs = [prefix(t, c['max_tokens']) for t in texts]
            counts = sparse.hstack([v.transform(inputs) for v in c['vectorizers']], format='csr')
            return c['model'].predict_proba(c['tfidf'].transform(counts))[:, 1].tolist()
        import torch
        with torch.inference_mode():
            if self.architecture == 'distilbert':
                inputs = self.tokenizer(texts, padding='max_length', truncation=True,
                                        max_length=256, return_tensors='pt').to(self.device)
                scores = self.model(**inputs).logits.softmax(-1)[:, 1]
            else:
                values = []
                for text in texts:
                    tokens = [m.group() for m in TOKEN.finditer(text.lower())][:256] or ['[UNK]']
                    values.append([self.vocabulary.get(t, 1) for t in tokens])
                ids = torch.zeros((len(values), 256), dtype=torch.long, device=self.device)
                lengths = torch.tensor([len(v) for v in values], device=self.device)
                for i, row in enumerate(values):
                    ids[i, :len(row)] = torch.tensor(row, device=self.device)
                scores = self.model(ids, lengths).sigmoid()
            return scores.cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--root', type=Path, help='Local folder containing architecture subfolders')
    source.add_argument('--repo-id', help=f'Hugging Face model repository (default: {DEFAULT_REPO})')
    p.add_argument('--revision', help='Hub commit, tag, or branch; defaults to latest main')
    p.add_argument('--cache-dir', type=Path, help='Override the Hugging Face download cache')
    p.add_argument('--offline', action='store_true', help='Load the Hub model from its existing cache')
    p.add_argument('--model', choices=['mlp', 'cnn', 'bilstm', 'distilbert'], required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.root:
        if args.revision or args.cache_dir or args.offline:
            p.error('--revision, --cache-dir, and --offline apply to Hub loading')
        classifier = PromptClassifier(args.root / args.model, args.model, args.device)
    else:
        classifier = PromptClassifier.from_hub(args.repo_id or DEFAULT_REPO, args.model, args.device,
                                              args.revision, args.cache_dir, args.offline)
    score = classifier.probabilities([args.text])[0]
    print(json.dumps({'model': args.model, 'source': classifier.source, 'execution': 'local',
                      'device': args.device, 'harmful_probability': score,
                      'label': 'harmful' if score >= .5 else 'benign'}))


if __name__ == '__main__':
    main()
