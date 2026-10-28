"""Check nested data integrity and external-source label handling."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from condition_data import condition_directory, expected_training_rows
from prepare_dataset_conditions import clean_wildguard
from prepare_large_corpus import digest, file_hash

ROOT=Path(__file__).resolve().parents[1]


class DatasetConditionTests(unittest.TestCase):
    def test_conflicting_labels_remove_every_copy_and_missing_labels_are_excluded(self):
        def row(text,label):
            return {'prompt':text,'prompt_harm_label':label,'adversarial':False}
        clean,counts=clean_wildguard([row('Same prompt','harmful'),row(' SAME   PROMPT ','unharmful'),
            row('same prompt','harmful'),row('Retained','unharmful'),row('retained','unharmful'),
            row('','harmful'),row('Missing',None)])
        self.assertEqual(set(clean),{digest('Retained')})
        self.assertEqual(clean[digest('Retained')]['label'],0)
        self.assertEqual(counts['conflicting_prompt_ids'],1)
        self.assertEqual(counts['missing_prompt_harm_label'],1)

    def test_condition_resolution_uses_distinct_data_directories(self):
        protocol=json.loads((ROOT/'configs/dataset_scaling.json').read_text())
        directories=[condition_directory(protocol,c) for c in protocol['conditions']]
        self.assertEqual(len(set(directories)),3)
        self.assertEqual(expected_training_rows(protocol,'wildjailbreak-183005'),183005)
        with self.assertRaises(KeyError): condition_directory(protocol,'unknown')

    def test_frozen_conditions_contain_the_same_100k_prefix_and_heldout_files(self):
        protocol=json.loads((ROOT/'configs/dataset_scaling.json').read_text())
        baseline=protocol['conditions']['wildjailbreak-100k']['partitions']
        for condition in ['wildjailbreak-150k','wildjailbreak-183005']:
            directory=condition_directory(protocol,condition)
            if not directory.exists(): self.skipTest('Local raw data is unavailable')
            fingerprint=hashlib.sha256()
            with (directory/'train.jsonl').open('rb') as handle:
                for i,line in enumerate(handle):
                    if i==100000: break
                    fingerprint.update(line)
            self.assertEqual(fingerprint.hexdigest(),baseline['train']['sha256'])
            for split in ['val','test','official_eval']:
                self.assertEqual(file_hash(directory/f'{split}.jsonl'),baseline[split]['sha256'])
            audit=json.loads((directory/'manifest.json').read_text())['overlap_audit']
            self.assertFalse(any(audit['exact_prompt_overlap'].values()))
            self.assertFalse(any(audit['main_request_overlap'].values()))

    def test_external_test_is_frozen_before_transfer_evaluation(self):
        protocol=json.loads((ROOT/'configs/transfer_evaluation.json').read_text())
        self.assertFalse(protocol['fit_on_external_data'])
        self.assertEqual(protocol['threshold'],0.5)
        path=ROOT/protocol['data_directory']/'test.jsonl'
        if not path.exists(): self.skipTest('Local evaluation data is unavailable')
        self.assertEqual(file_hash(path),protocol['manifest']['partitions']['test']['sha256'])
        records=[json.loads(line) for line in path.open()]
        self.assertEqual(len(records),1695)
        self.assertEqual(len({r['id'] for r in records}),1695)
        self.assertEqual(sum(r['label'] for r in records),754)


if __name__=='__main__': unittest.main()
