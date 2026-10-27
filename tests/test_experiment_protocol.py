from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from aggregate_experiments import aggregate
from experiment_state import capture_rng, restore_rng, update_validation, atomic_torch


class ProtocolTests(unittest.TestCase):
    def test_early_stopping_state_survives_checkpoint(self):
        state={'epoch':0,'best_score':None,'bad_epochs':0,'chosen_epoch':None,'early_stopped':False}
        self.assertTrue(update_validation(state,.8,2))
        state['epoch']=1
        self.assertFalse(update_validation(state,.7,2))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'; atomic_torch(path,state)
            resumed=torch.load(path,weights_only=False)
        resumed['epoch']=2
        self.assertFalse(update_validation(resumed,.8,2))
        self.assertTrue(resumed['early_stopped'])
        self.assertEqual(resumed['chosen_epoch'],1)

    def test_rng_roundtrip_restores_python_numpy_and_torch(self):
        import random
        before=capture_rng()
        expected=(random.random(),np.random.rand(),torch.rand(4))
        restore_rng(before)
        actual=(random.random(),np.random.rand(),torch.rand(4))
        self.assertEqual(expected[0],actual[0]); self.assertEqual(expected[1],actual[1])
        self.assertTrue(torch.equal(expected[2],actual[2]))

    def test_aggregation_accepts_extra_encoder_and_excludes_smoke(self):
        def report(seed,score):
            keys=['accuracy','precision','recall','f1','mcc','roc_auc','average_precision','brier_score']
            return {'status':'complete','smoke_only':False,'protocol_sha256':'protocol','condition':'data',
                'model':'deberta','seed':seed,'data_hashes':{'test':'fixed'},'threshold':.5,
                'encoder_gradient_verified':True,'encoder_weight_changed':True,
                'config':dict(model_settings={},max_tokens=256,batch_size=16,epoch_cap=5,patience=2,min_delta=0,threshold=.5),
                'metrics':{'test':dict.fromkeys(keys,score),'official_eval':{'recall':score,'benign_acceptance':score}}}
        a,b=report(4442,.8),report(4443,.9)
        smoke=deepcopy(a); smoke['smoke_only']=True
        result=aggregate([a,b,smoke])
        self.assertEqual(result[0]['runs'],2)
        self.assertAlmostEqual(result[0]['metrics']['f1']['mean'],.85)
        self.assertFalse(result[0]['complete_seed_set'])
        with self.assertRaises(ValueError): aggregate([a,a])
        b['code_hashes']={'trainer':'changed'}
        with self.assertRaises(ValueError): aggregate([a,b])
        b.pop('code_hashes')
        b['data_hashes']['test']='changed'
        with self.assertRaises(ValueError): aggregate([a,b])


if __name__=='__main__': unittest.main()
