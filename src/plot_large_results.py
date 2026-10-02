"""Render a standalone comparison figure from completed shared-partition runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'results/large'
    table = json.loads((folder / 'comparison.json').read_text())
    ci = json.loads((folder / 'uncertainty.json').read_text())['f1_95_percent_intervals']
    names = [r['model'] for r in table]
    labels = ['MLP','CNN','BiLSTM','DistilBERT']
    colors = ['#0072B2','#E69F00','#009E73','#CC79A7']
    fig, axes = plt.subplots(1,2,figsize=(10,4.2), gridspec_kw={'width_ratios':[1,1.2]})
    values = np.array([r['test_f1'] for r in table])
    intervals = np.array([ci[name] for name in names])
    errors = np.maximum(0,np.stack([values-intervals[:,0],intervals[:,1]-values]))
    axes[0].bar(labels, values, color=colors, yerr=errors, capsize=4)
    axes[0].set_ylim(0,1.06); axes[0].set_ylabel('F1 score')
    axes[0].set_title('Main grouped test: 38,911 prompts')
    for i,value in enumerate(values):
        axes[0].text(i,value+.025,f'{value:.3f}',ha='center',fontsize=9)
    x = np.arange(len(names)); width = .35
    harmful = [r['official_harmful_recall'] for r in table]
    benign = [r['official_benign_acceptance'] for r in table]
    axes[1].bar(x-width/2,harmful,width,label='Harmful recall',color='#0072B2')
    axes[1].bar(x+width/2,benign,width,label='Benign acceptance',color='#E69F00')
    axes[1].set_xticks(x,labels); axes[1].set_ylim(0,1.06)
    axes[1].set_title('Official challenge: 2,000 harmful / 210 benign')
    axes[1].legend(loc='lower right',frameon=False,fontsize=9)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Prompt harmfulness detection · 100,000 unique training prompts',fontsize=13)
    fig.text(.5,.025,'Three epochs; validation-selected checkpoints; one training seed. F1 intervals: 95% bootstrap over request groups.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(folder / 'comparison.png',dpi=180)
    fig.savefig(folder / 'comparison.pdf')
    print(f'Saved {folder / "comparison.png"}')


if __name__ == '__main__':
    main()
