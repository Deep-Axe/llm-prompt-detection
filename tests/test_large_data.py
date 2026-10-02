import csv
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_large_corpus import digest, upstream_rows, split_for_group


class LargeDataTests(unittest.TestCase):
    def test_normalization_handles_case_whitespace_and_unicode(self):
        self.assertEqual(digest('  HELLO\nworld '), digest('hello world'))
        self.assertEqual(digest('ＡＢＣ'), digest('abc'))

    def test_variants_keep_request_group_and_prompt_label(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'train.tsv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['vanilla','adversarial','completion','data_type'], delimiter='\t')
                writer.writeheader()
                writer.writerow({'vanilla':'request A', 'adversarial':'', 'completion':'refusal', 'data_type':'vanilla_harmful'})
                writer.writerow({'vanilla':'request A', 'adversarial':'variant A', 'completion':'safe refusal', 'data_type':'adversarial_harmful'})
            records = list(upstream_rows(path))
            self.assertEqual([r['label'] for r in records], [1,1])
            self.assertEqual(records[0]['group'], records[1]['group'])
            self.assertNotEqual(records[0]['id'], records[1]['id'])
            self.assertEqual(split_for_group(records[0]['group']), split_for_group(records[1]['group']))

    def test_official_eval_schema_and_unknown_label(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'eval.tsv'
            path.write_text('adversarial\tlabel\tdata_type\nexample\t0\tadversarial_benign\n')
            self.assertEqual(next(upstream_rows(path))['label'], 0)
            path.write_text('adversarial\tlabel\tdata_type\nexample\t0\tunknown\n')
            with self.assertRaises(ValueError):
                list(upstream_rows(path))


if __name__ == '__main__':
    unittest.main()
