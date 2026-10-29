"""Check prediction alignment and cross-seed error counting."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze_classifier_errors import validate_predictions, error_masks, length_bin


class ErrorAnalysisTests(unittest.TestCase):
    def test_prediction_alignment_rejects_reordered_rows_and_invalid_probabilities(self):
        records=[{'id':'a','label':0},{'id':'b','label':1}]
        archive={'ids':np.array(['a','b']),'labels':np.array([0,1]),'probabilities':np.array([.2,.9])}
        np.testing.assert_equal(validate_predictions(records,archive),[.2,.9])
        with self.assertRaises(ValueError): validate_predictions(list(reversed(records)),archive)
        for invalid in [np.array([np.nan,.9]),np.array([-.1,.9]),np.array([[.2,.9]])]:
            with self.assertRaises(ValueError): validate_predictions(records,archive|{'probabilities':invalid})

    def test_stable_errors_require_all_seeds_and_use_inclusive_threshold(self):
        labels=np.array([0,1,1,0])
        probabilities=np.array([[.8,.1,.8,.2],[.7,.2,.4,.3],[.9,.8,.6,.4]])
        masks=error_masks(labels,probabilities)
        np.testing.assert_equal(masks['every_seed'],[True,False,False,False])
        np.testing.assert_equal(masks['any_seed'],[True,True,True,False])
        np.testing.assert_equal(masks['seed_sensitive'],[False,True,True,False])
        masks=error_masks(np.array([0,1]),np.full((3,2),.5))
        np.testing.assert_equal(masks['every_seed'],[True,False])

    def test_length_bucket_boundaries_do_not_skip_or_double_count_tokens(self):
        bounds=[64,128,256,512]
        cases={1:'1-64',64:'1-64',65:'65-128',128:'65-128',129:'129-256',256:'129-256',
               257:'257-512',512:'257-512',513:'>512'}
        for length,expected in cases.items(): self.assertEqual(length_bin(length,bounds),expected)


if __name__=='__main__': unittest.main()
