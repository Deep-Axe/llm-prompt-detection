"""Update the ignored report's training section from the completed larger run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / 'results/large'
    rows = json.loads((results / 'comparison.json').read_text())
    d = json.loads((results / 'distilbert.json').read_text())
    names = {'mlp':'TF--IDF MLP','cnn':'1D CNN','bilstm':'BiLSTM','distilbert':'DistilBERT'}
    table = '\n'.join(f'{names[r["model"]]} & {r["test_f1"]:.3f} & {r["test_mcc"]:.3f} & {r["official_harmful_recall"]:.3f} & {r["official_benign_acceptance"]:.3f} \\\\' for r in rows)
    timing = '\n'.join(f'{names[r["model"]]} & {r["training_and_validation_seconds"]:.1f} & {r["chosen_epoch"]} \\\\' for r in rows)
    text = r'''\section{Training and evaluation}
\label{sec:preliminary}

The first experiment used a small corpus assembled from DAN templates, JailbreakBench requests, and Alpaca instructions. Its mixed-test scores were high, but a length-only rule already reached an $F_1$ of 0.906. A frozen DistilBERT encoder with a trained head was included in that initial comparison. Those results remain separate from the larger experiment below.

The expanded experiment uses WildJailbreak \citep{jiang2024wildteaming}, which includes benign and harmful requests in both vanilla and adversarial forms. We downloaded the provider's training and evaluation files at an immutable revision after access was enabled. Prompt labels follow the data type; model completions are excluded. A refusal in the completion column therefore does not make a harmful input benign.

After normalization, duplicate removal, and conflict checks, we assigned each underlying vanilla request and its adversarial variants to one main partition. From an eligible training pool of 183,005 prompts, we selected exactly 100,000 unique training prompts by hash. Validation contains 39,502 prompts and test contains 38,911. Another 2,210 prompts form the official evaluation. Exact prompt overlap is excluded across all partitions. The official file omits vanilla requests, so the stronger check for underlying-request overlap is possible only for the main splits.

All four architectures train for three epochs with seed 4442. We select the checkpoint with the highest validation $F_1$ and evaluate the test partitions afterwards. The probability threshold stays at 0.5. These are basic baseline runs without a hyperparameter search. All models use a 256-token context budget, although regex and WordPiece token boundaries differ.

The MLP uses hashed word and character n-grams with 32,768 features and a 64-unit hidden layer. Its inverse document frequencies are fitted only on training prompts. Disk-backed batches keep the feature matrices within local RAM limits; hashing introduces possible collisions. The CNN and BiLSTM share a vocabulary built only from training text, capped at 20,000 terms, and learn 64-dimensional embeddings. The CNN uses 64 filters for each of widths three, four, and five. The BiLSTM has 64 hidden units per direction and combines mean and max pooling.

The expanded DistilBERT run fine-tunes every encoder layer as well as the classifier. It uses AdamW at $2\times10^{-5}$, weight decay 0.01, a ten-percent warmup, and linear learning-rate decay. Runtime checks confirm nonzero gradients in every encoder layer and a change in an encoder attention weight matrix. All ''' + f'{d["trainable_parameters"]:,}' + r''' model parameters are trainable.

\begin{table}[htbp]
\centering
\caption{Results after training each architecture on the same 100,000 prompts. Official harmful recall uses 2,000 prompts; benign acceptance uses 210.}
\label{tab:preliminary}
\small
\begin{tabular}{lrrrr}
\toprule
Model & Test $F_1$ & Test MCC & Harmful recall & Benign acceptance \\
\midrule
''' + table + r'''
\bottomrule
\end{tabular}
\end{table}

The main test split and official challenge measure different conditions. The official set is heavily weighted toward harmful prompts, so a high $F_1$ there can coexist with poor benign acceptance. We therefore report the two class-specific rates alongside the main test scores. The larger corpus's length-only rule reaches test $F_1=0.694$: length remains a useful shortcut, but its score is well below the trained classifiers.

MLP training runs on the local CPU, while CNN and BiLSTM use the local 6\,GB RTX 3060. A local full-encoder benchmark verified that DistilBERT fine-tuning is feasible in VRAM, but projected roughly 27 minutes per epoch with small batches and gradient accumulation. With only about 1--2\,GB system RAM free, Kaggle is the more practical location for the sustained transformer run. The completed run's hardware and configuration are retained in its result record.

\begin{table}[htbp]
\centering
\caption{Training and validation wall time for these runs. Different hardware prevents interpreting these values as an architecture speed ranking.}
\small
\begin{tabular}{lrr}
\toprule
Model & Time (s) & Selected epoch \\
\midrule
''' + timing + r'''
\bottomrule
\end{tabular}
\end{table}

The recorded metrics include accuracy, precision, recall, $F_1$, Matthews correlation, ROC-AUC, average precision, log loss, and Brier score, with separate data-type slices. Paired bootstrap intervals resample underlying test request groups rather than treating related variants as independent. These intervals cover test sampling uncertainty; they do not capture variation across training seeds or mistakes in the upstream labels.

The corpus is largely synthetic. Grouping exact underlying requests reduces one source of leakage, but different requests can still share tactics or be paraphrases. Harmfulness classification also does not establish whether a harmless-looking instruction is an injection in a particular application context. Reviewed natural user prompts, multiple training seeds, and a separate context-aware injection condition remain necessary for broader conclusions. Code and aggregate results are published separately from prompt text, checkpoints, and the local report.
'''
    path = ROOT / 'report/sections/preliminary_results.tex'
    path.write_text(text)
    print(f'Updated {path}')


if __name__ == '__main__':
    main()
