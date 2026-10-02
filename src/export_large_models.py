"""Prepare local model metadata/cards and a public checksum-only inventory.

No upload occurs. Weights, vocabulary, and individual predictions remain ignored.
"""
import json
from pathlib import Path
import shutil
from evaluation import ROOT
from prepare_large_corpus import file_hash


def main():
    inventory = {}
    for name in ['mlp','cnn','bilstm','distilbert']:
        report = json.loads((ROOT / f'results/large/{name}.json').read_text())
        folder = ROOT / f'artifacts/large/{name}'
        if name in ['cnn','bilstm']:
            shutil.copy2(folder / 'tokens/vocabulary.json', folder / 'vocabulary.json')
            checkpoint_files = ['best.pt','vocabulary.json']
        elif name == 'mlp':
            checkpoint_files = ['checkpoint.joblib']
        else:
            checkpoint_files = ['model/config.json','model/model.safetensors','model/tokenizer.json',
                                'model/tokenizer_config.json','model/vocab.txt']
        metadata = {'architecture':name, 'labels':{'0':'benign','1':'harmful'}, 'threshold':.5,
                    'max_tokens':256, 'training_rows':report['train_rows'],
                    'chosen_epoch':report['chosen_epoch'], 'data_hashes':report['data_hashes'],
                    'run_code_hashes':report['code_hashes'], 'test_metrics':report['metrics']['test'],
                    'official_eval_metrics':report['metrics']['official_eval'],
                    'files':{p:file_hash(folder / p) for p in checkpoint_files}}
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        card = f'''# {name.upper()} prompt harmfulness classifier

Trained on 100,000 unique WildJailbreak prompts for three epochs; the checkpoint
was selected using validation F1, with threshold fixed at 0.5. Label 0 means
benign and label 1 means harmful. Upstream model completions are never inputs.

Main test F1: {report['metrics']['test']['f1']:.4f}.
Official harmful recall: {report['metrics']['official_eval/adversarial_harmful']['recall']:.4f}.
Official benign acceptance: {report['metrics']['official_eval/adversarial_benign']['benign_acceptance']:.4f}.

Data: Jiang et al., WildTeaming at Scale (2024),
https://huggingface.co/datasets/allenai/wildjailbreak, revision
5ddc12a7894f842b0619b8e1c7ee496b198af009, ODC-BY-1.0.
The main partitions group variants by their underlying vanilla request.
The official evaluation omits those request identifiers; exact overlaps are
removed, but underlying-request overlap cannot be checked for that set.

These are single-seed baseline results on mostly synthetic data. The classifier
does not establish context-dependent instruction injection or deployment safety.
Prompts beyond the 256-token context budget are truncated. Regex tokenization
and WordPiece use different token boundaries. See metadata.json for checksums,
metrics, and code provenance. The code in the source repository is MIT-licensed;
data and base models retain their upstream licenses.
'''
        if name == 'distilbert':
            card += '\nAll DistilBERT encoder layers and the classification head were fine-tuned.\nBase model: distilbert/distilbert-base-uncased (Apache-2.0), revision\n12040accade4e8a0f71eabdb258fecc2e7e948be. Load the model/ directory with\nAutoModelForSequenceClassification and AutoTokenizer from transformers.\n'
            (folder / 'model/README.md').write_text(card)
        elif name == 'mlp':
            card += '\nLoad checkpoint.joblib using joblib with scikit-learn 1.3.2. It contains\ntwo HashingVectorizers, training-fitted TF-IDF, and the 64-unit MLP. Apply\nthe 256-regex-token prefix from src/train_large_mlp.py before vectorization.\nOnly load trusted joblib files.\n'
        else:
            card += f'\nLoad best.pt using torch.load(..., map_location="cpu", weights_only=True).\nConstruct {"SentenceCNN" if name=="cnn" else "BiLSTM"}(len(vocabulary)+2) from src/neural_models.py\nand use the saved vocabulary with src/large_data.py word_tokens.\n'
        (folder / 'README.md').write_text(card)
        inventory[name] = {'files':metadata['files'], 'test_f1':report['metrics']['test']['f1'],
                           'location':f'artifacts/large/{name}', 'uploaded':False}
    (ROOT / 'results/large/checkpoint_manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
    print('Prepared four local model cards and checkpoint checksum inventory; no upload performed.')


if __name__ == '__main__':
    main()
