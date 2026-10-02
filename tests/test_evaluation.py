import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation import metrics


class MetricsTests(unittest.TestCase):
    def test_single_class_does_not_claim_auc(self):
        result=metrics([{'label':0},{'label':0}],np.array([.1,.8]))
        self.assertIsNone(result['roc_auc'])
        self.assertIsNone(result['f1'])
        self.assertEqual(result['benign_acceptance'],.5)
        self.assertEqual(result['false_positives'],1)

    def test_perfect_mixed_prediction(self):
        result=metrics([{'label':0},{'label':1}],np.array([.1,.9]))
        self.assertEqual(result['f1'],1)
        self.assertEqual(result['roc_auc'],1)
        self.assertEqual(result['pr_auc'],1)

    def test_nonfinite_probabilities_fail(self):
        with self.assertRaises(ValueError): metrics([{'label':0}],np.array([np.nan]))


if __name__=='__main__': unittest.main()
