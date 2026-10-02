"""Load a published checkpoint and return prompt harmfulness probabilities."""
import argparse
import json
from pathlib import Path
import re

TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def prefix(text, limit=256):
    for i, match in enumerate(TOKEN.finditer(text)):
        if i + 1 == limit:
            return text[:match.end()]
    return text


class PromptClassifier:
    def __init__(self, folder, architecture, device='cpu'):
        self.folder = Path(folder)
        self.architecture = architecture
        self.device = device
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
                from neural_models import SentenceCNN, BiLSTM
                self.vocabulary = json.loads((self.folder / 'vocabulary.json').read_text())
                model_type = SentenceCNN if architecture == 'cnn' else BiLSTM
                self.model = model_type(len(self.vocabulary) + 2)
                self.model.load_state_dict(torch.load(self.folder / 'best.pt', map_location='cpu', weights_only=True))
            self.model.to(device).eval()

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
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=['mlp', 'cnn', 'bilstm', 'distilbert'], required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    score = PromptClassifier(args.root / args.model, args.model, args.device).probabilities([args.text])[0]
    print(json.dumps({'harmful_probability': score, 'label': 'harmful' if score >= .5 else 'benign'}))


if __name__ == '__main__':
    main()
