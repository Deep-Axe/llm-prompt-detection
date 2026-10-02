"""Refresh the ignored report's results table while preserving its edited prose."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'results/large'
    rows = json.loads((folder / 'comparison.json').read_text())
    length = json.loads((folder / 'data_audit.json').read_text())['length_only_baseline']['metrics']
    names = {'mlp': 'TF--IDF MLP', 'cnn': '1D CNN', 'bilstm': 'BiLSTM', 'distilbert': 'DistilBERT'}
    table_rows = []
    for r in rows:
        values = [r[k] for k in ['test_accuracy', 'test_f1', 'test_mcc', 'official_harmful_recall', 'official_benign_acceptance']]
        table_rows.append(names[r['model']] + ' & ' + ' & '.join(f'{v:.4f}' for v in values) + r' \\')
    values = [length['test'][k] for k in ['accuracy', 'f1', 'mcc']] + [length['official_eval'][k] for k in ['recall', 'benign_acceptance']]
    table_rows.append('Length rule & ' + ' & '.join(f'{v:.4f}' for v in values) + r' \\')
    path = ROOT / 'report/sections/preliminary_results.tex'
    text = path.read_text()
    # Only the body of the explicitly labeled results table is generated.
    pattern = r'(\\label\{tab:preliminary\}.*?\\midrule\n).*?(\n\\bottomrule)'
    updated, count = re.subn(pattern, lambda m: m[1] + '\n'.join(table_rows) + m[2], text, flags=re.DOTALL)
    if count != 1:
        raise ValueError('Expected one results table labeled tab:preliminary')
    path.write_text(updated)
    print(f'Refreshed the results table in {path}; retained report prose.')


if __name__ == '__main__':
    main()
