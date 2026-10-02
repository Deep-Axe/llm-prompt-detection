"""Finite-difference check of the hand-written CNN backward pass."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train_cnn_numpy import SentenceCNN


class GradientTest(unittest.TestCase):
    def test_backward_matches_finite_differences(self):
        model = SentenceCNN(8,embedding_dim=3,filters=2,widths=(2,3),dtype=np.float64)
        ids = np.array([[2,3,4,5,6],[3,4,2,0,0]])
        lengths = np.array([5,3])
        y = np.array([1.,0.])
        _, grads = model.loss_grad(ids,lengths,y)
        rng = np.random.default_rng(7)
        for name,param in model.p.items():
            for flat in rng.choice(param.size,min(10,param.size),replace=False):
                ix = np.unravel_index(flat,param.shape)
                if name=='embedding' and ix[0]==0:
                    continue
                original = param[ix]
                param[ix] = original+1e-5
                plus,_ = model.loss_grad(ids,lengths,y)
                param[ix] = original-1e-5
                minus,_ = model.loss_grad(ids,lengths,y)
                param[ix] = original
                self.assertAlmostEqual(grads[name][ix],(plus-minus)/2e-5,places=6,msg=f'{name}{ix}')


if __name__ == '__main__':
    unittest.main()
