"""Plot source-wise false-positive and false-negative rates across training seeds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/error_analysis/wildjailbreak-100k/slices.csv')
    args=parser.parse_args()
    with args.input.open() as handle:
        rows=[r for r in csv.DictReader(handle) if r['slice_type']=='all']
    models=['mlp','cnn','bilstm','distilbert']
    evaluations=['wildjailbreak-test','wildjailbreak-official','wildguardmix-transfer']
    labels=['WildJailbreak test\nBenign n=19,131\nHarmful n=19,780',
            'WildJailbreak official\nBenign n=210\nHarmful n=2,000',
            'WildGuardMix transfer\nBenign n=941\nHarmful n=754']
    names=['MLP','CNN','BiLSTM','DistilBERT'];colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    lookup={(r['evaluation'],r['model']):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.9))
    x=np.arange(3);width=.18
    for axis,key,title in zip(axes,['false_positive_rate','false_negative_rate'],
                              ['Benign prompts rejected','Harmful prompts missed']):
        for index,(model,name,color) in enumerate(zip(models,names,colors)):
            means=[float(lookup[(e,model)][key+'_mean']) for e in evaluations]
            deviations=[float(lookup[(e,model)][key+'_std']) for e in evaluations]
            axis.bar(x+(index-1.5)*width,means,width,color=color,label=name,yerr=deviations,capsize=2)
        axis.set_xticks(x,labels,fontsize=8)
        axis.set_title(title);axis.set_ylabel('Fraction of prompts in the labeled class')
        axis.set_ylim(bottom=0)
        axis.spines['top'].set_visible(False);axis.spines['right'].set_visible(False)
        axis.set_axisbelow(True);axis.grid(axis='y',alpha=.2)
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('100,000-row classifiers: errors across evaluation sources',fontsize=13)
    fig.text(.5,.025,'Threshold 0.5; means and sample standard deviations across seeds 4442–4444. Evaluation distributions differ.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    for extension in ['png','pdf']:
        fig.savefig(args.input.parent/f'source_errors.{extension}',dpi=180)
    provenance={'source_csv_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'plot_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__}
    (args.input.parent/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(args.input.parent/'source_errors.png')


if __name__=='__main__': main()
