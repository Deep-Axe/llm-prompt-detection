"""Verify wider CNN bundles, context handling, and fixed-vocabulary isolation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural_models import SentenceCNN
from predict_cnn import CNNClassifier
from experiment_data import load_fixed_vocabulary,cache_fingerprint
from predict_large_models import PromptClassifier


def write_bundle(folder,context=512,wide=True):
    embedding=filters=128 if wide else 64
    model=SentenceCNN(4,embedding_dim=embedding,filters=filters)
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model.embedding.weight[3,0]=1
        model.convs[0].weight[0,0,0]=1
        model.head.weight[0,0]=1;model.head.bias.fill_(-.5)
    torch.save(model.state_dict(),folder/'best.pt')
    (folder/'vocabulary.json').write_text(json.dumps({'plain':2,'tail':3}))
    digest=lambda name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
    config={'architecture':'cnn','architecture_settings':{'embedding_dim':embedding,'filters':filters,'widths':[3,4,5]},
        'max_tokens':context,'threshold':0.5,'vocabulary_sha256':digest('vocabulary.json'),'checkpoint_sha256':digest('best.pt')}
    (folder/'model_config.json').write_text(json.dumps(config))
    return config


class CNNExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_512_context_uses_tokens_after_256_with_identical_architecture_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);config=write_bundle(folder)
            text='plain '*300+'tail plain plain plain'
            long=CNNClassifier(folder)
            self.assertGreater(long.probabilities([text])[0],.5)
            config['max_tokens']=256
            (folder/'model_config.json').write_text(json.dumps(config))
            short=CNNClassifier(folder)
            self.assertLess(short.probabilities([text])[0],.5)
            self.assertEqual(long.vocabulary,short.vocabulary)
            self.assertEqual(sum(p.numel() for p in long.model.parameters()),sum(p.numel() for p in short.model.parameters()))

    def test_default_256_bundle_matches_existing_cnn_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);write_bundle(folder,context=256,wide=False)
            texts=['plain plain tail plain plain','','plain '*300+'tail plain plain']
            self.assertEqual(CNNClassifier(folder).probabilities(texts),PromptClassifier(folder,'cnn').probabilities(texts))

    def test_bundle_and_vocabulary_checksums_reject_modified_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);config=write_bundle(folder)
            load_fixed_vocabulary(folder/'vocabulary.json',config['vocabulary_sha256'])
            (folder/'vocabulary.json').write_text(json.dumps({'plain':2,'tail':8}))
            with self.assertRaises(ValueError): CNNClassifier(folder)
            with self.assertRaises(ValueError): load_fixed_vocabulary(folder/'vocabulary.json',config['vocabulary_sha256'])
            digest=hashlib.sha256((folder/'vocabulary.json').read_bytes()).hexdigest()
            with self.assertRaises(ValueError): load_fixed_vocabulary(folder/'vocabulary.json',digest)

    def test_cache_identity_keeps_context_and_vocabulary_separate(self):
        config={'model':'cnn','seed':4442,'data_hashes':{'train':'same'},'max_tokens':256,'smoke_only':False,
            'model_settings':{},'code_hashes':dict.fromkeys(['experiment_data.py','large_data.py','train_large_mlp.py'],'code')}
        baseline=cache_fingerprint(config)
        self.assertNotEqual(baseline,cache_fingerprint(config|{'max_tokens':512}))
        self.assertNotEqual(baseline,cache_fingerprint(config|{'vocabulary_source':{'sha256':'different'}}))


if __name__=='__main__': unittest.main()
