"""Run the fixed corpus experiment and export comparison tables."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    scripts=[['audit_data.py'],['length_baseline.py'],['train_mlp.py'],['download_encoder.py']]
    scripts += [['train_neural.py',model,'--device',args.device] for model in ['cnn','bilstm','distilbert']]
    scripts += [['summarize_results.py']]
    for script,*options in scripts:
        subprocess.run([sys.executable,str(ROOT/'src'/script),*options],cwd=ROOT,check=True)


if __name__=='__main__': main()
