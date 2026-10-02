"""Stage only deployable checkpoints, inference code, cards, and aggregate results."""
import argparse
import json
from pathlib import Path
import shutil
from evaluation import ROOT
from prepare_large_corpus import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', default='DeeAxe/llm-prompt-detection')
    args = p.parse_args()
    target = ROOT / 'artifacts/huggingface/llm-prompt-detection'
    target.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ROOT / 'results/large/checkpoint_manifest.json').read_text())
    for name, record in inventory.items():
        source = ROOT / record['location']
        folder = target / name
        folder.mkdir(exist_ok=True)
        for relative, expected in record['files'].items():
            if file_hash(source / relative) != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {name}/{relative}')
            destination = folder / Path(relative).name
            shutil.copy2(source / relative, destination)
        shutil.copy2(source / 'metadata.json', folder / 'metadata.json')
        metadata = json.loads((folder / 'metadata.json').read_text())
        metadata['files'] = {Path(p).name: sha for p, sha in metadata['files'].items()}
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        shutil.copy2(source / 'README.md', folder / 'README.md')
        shutil.copy2(ROOT / f'results/large/{name}.json', folder / 'training_results.json')
    # Transformer files live directly in its subfolder for from_pretrained(..., subfolder=...).
    card = (target / 'distilbert/README.md').read_text().replace('Load the model/ directory', 'Load the distilbert/ directory')
    (target / 'distilbert/README.md').write_text(card)
    shutil.copy2(ROOT / 'src/predict_large_models.py', target / 'predict.py')
    shutil.copy2(ROOT / 'src/neural_models.py', target / 'neural_models.py')
    shutil.copy2(ROOT / 'LICENSE', target / 'LICENSE_MIT')
    shutil.copy2(ROOT / 'artifacts/huggingface_base_license/LICENSE', target / 'distilbert/LICENSE')
    for filename in ['comparison.csv', 'comparison.json', 'comparison.png', 'uncertainty.json',
                     'data_manifest.json', 'checkpoint_verification.json']:
        shutil.copy2(ROOT / f'results/large/{filename}', target / filename)
    hub_check = ROOT / 'results/large/hub_inference_verification.json'
    if hub_check.exists():
        shutil.copy2(hub_check, target / hub_check.name)
    (target / 'requirements.txt').write_text('torch==2.5.1\ntransformers==4.46.3\nhuggingface-hub==0.36.2\nscikit-learn==1.3.2\nnumpy==1.24.4\nscipy==1.15.3\njoblib\n')
    (target / 'LICENSE.md').write_text('''# Licenses and attribution

Original inference code and the MLP/CNN/BiLSTM checkpoints are released under
the MIT license in LICENSE_MIT. The fine-tuned DistilBERT checkpoint retains
the upstream Apache-2.0 license in distilbert/LICENSE. Its base model is
distilbert/distilbert-base-uncased, revision 12040accade4e8a0f71eabdb258fecc2e7e948be.
It was modified by fine-tuning the full encoder and classifier for three epochs.

Training data: Jiang et al., WildTeaming at Scale (2024), WildJailbreak,
https://huggingface.co/datasets/allenai/wildjailbreak, revision
5ddc12a7894f842b0619b8e1c7ee496b198af009, ODC-BY-1.0.
No training prompts or individual predictions are distributed in this repository.
Dataset and model licenses remain distinct from the original project code license.
''')
    (target / 'README.md').write_text(f'''---
language: en
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE.md
datasets:
- allenai/wildjailbreak
tags:
- text-classification
- prompt-harmfulness
- jailbreak-detection
- mlp
- cnn
- bilstm
- distilbert
---

# LLM prompt detection: four baseline architectures

We are comparing how much model complexity an input-side guardrail needs.
This repository contains a TF-IDF MLP, a sentence CNN, a bidirectional LSTM,
and a fully fine-tuned DistilBERT. The project combines literature and dataset
review with data acquisition, preparation, training, and evaluation.

The implemented task is **prompt harmfulness classification**. Label 0 means
benign and label 1 means harmful, with a fixed probability threshold of 0.5.
This experiment does not establish context-dependent prompt-injection detection.

Code and report provenance: https://github.com/Deep-Axe/llm-prompt-detection

## Data and training

All four models use the same 100,000 unique WildJailbreak training prompts,
39,502 validation prompts, and 38,911 test prompts. The official evaluation
adds 2,000 harmful and 210 benign prompts. Inputs exclude model completions.
We use upstream vanilla/adversarial benign and harmful data types as labels.
Normalized duplicates and label conflicts are removed; variants of the same
underlying vanilla request stay together in the main splits. Official evaluation
does not expose those identifiers, so only exact overlap can be checked there.

Each architecture trains for three epochs, seed 4442, selecting the checkpoint
with highest validation F1. MLP uses 32,768 hashed word/character TF-IDF features
and a 64-unit hidden layer. CNN and BiLSTM use a training-only 20,000-term
vocabulary and learned 64-dimensional embeddings. All models have a 256-token
budget, but regex and WordPiece tokenization differ. MLP trains on local CPU;
CNN and BiLSTM on an RTX 3060. DistilBERT trains on one Kaggle T4 with batch 32,
FP16, gradient checkpointing, and AdamW at 2e-5. **All 66,955,010 parameters
are trainable**, with encoder gradients and changed weights verified.

## Results

| Model | Main test F1 | Official harmful recall | Official benign acceptance |
| --- | ---: | ---: | ---: |
| MLP | 0.9355 | 0.8000 | 0.5857 |
| CNN | 0.9290 | 0.8585 | 0.4476 |
| BiLSTM | 0.9423 | 0.8570 | 0.5571 |
| DistilBERT | 0.9792 | 0.9055 | 0.6000 |

![Comparison](comparison.png)

DistilBERT leads the main test, but falsely flags 40% of official benign prompts.
These single-seed baselines are not ready-made production guardrails. The data
are largely synthetic, related tactics can cross request groups, and truncation
can remove relevant text. Bootstrap intervals describe test sampling uncertainty,
not training-seed variation. Full metrics, histories, hashes, and model settings
are in each folder's training_results.json and metadata.json. Checkpoints were
reloaded and predictions checked against saved evaluation outputs.

## Load DistilBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="distilbert")
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder="distilbert").eval()
inputs = tokenizer("Explain how rainbows form.", return_tensors="pt",
                   truncation=True, max_length=256)
with torch.inference_mode():
    harmful_probability = model(**inputs).logits.softmax(-1)[0, 1].item()
```

## Load any of the four models

Use Python 3.10 and install requirements.txt in a separate environment. The
project's inference helper accepts this repository ID directly and fetches only
the selected architecture's files:

```bash
python src/predict_large_models.py --repo-id {args.repo_id} --model cnn --text "Explain how rainbows form."
```

From the source repository, Python usage is:

```python
from src.predict_large_models import PromptClassifier
classifier = PromptClassifier.from_hub("{args.repo_id}", architecture="cnn")
probabilities = classifier.probabilities(["Explain how rainbows form."])
```

Downloads are cached. Inference executes on the local CPU by default; select
`--device cuda` to use a local GPU. `--revision COMMIT_SHA` pins a release and
`--offline` reuses an existing cached version. Output includes the resolved Hub
revision. This model repository distributes checkpoints; no hosted inference
endpoint is deployed.

You can also download the repository and run its bundled helper:

```python
from huggingface_hub import snapshot_download
folder = snapshot_download("{args.repo_id}")
print(folder)
```

```bash
python /path/to/snapshot/predict.py --root /path/to/snapshot --model cnn --text "Explain how rainbows form."
```

Model choices are mlp, cnn, bilstm, and distilbert; the default device is CPU.
MLP's joblib checkpoint should only be loaded from a trusted source. The helper
reproduces the saved vocabulary, token limits, and training-fitted TF-IDF.
CNN/BiLSTM state dictionaries load with torch.load(..., weights_only=True).

## Attribution

WildJailbreak: Jiang et al., *WildTeaming at Scale* (2024), ODC-BY-1.0,
revision 5ddc12a7894f842b0619b8e1c7ee496b198af009.
DistilBERT base: distilbert/distilbert-base-uncased, Apache-2.0,
revision 12040accade4e8a0f71eabdb258fecc2e7e948be.
See LICENSE.md for file-specific licenses. Prompt data and individual predictions
are excluded; this repository distributes checkpoints and aggregate results.
''')
    print(target)


if __name__ == '__main__':
    main()
