"""Update the local, ignored TeX report from the recorded four-model run."""
import json
from evaluation import ROOT


def main():
    result=json.loads((ROOT/'results/comparison.json').read_text())
    b=json.loads((ROOT/'data/processed/length_baseline.json').read_text())['splits']
    lines=[r'''\section{Preliminary training and results}
\label{sec:preliminary}

Four models were trained using the fixed corpus partitions: a TF--IDF MLP, a sentence CNN, a bidirectional LSTM, and a frozen DistilBERT encoder with a linear classification head. These are initial runs with fixed configurations, seed 4442, and a probability threshold of 0.5. No hyperparameter search or validation-based checkpoint selection was performed.

The MLP combines up to 8,000 word n-gram and 8,000 character n-gram features with a 64-unit hidden layer, trained for a 20-epoch budget. The CNN uses 64-dimensional learned embeddings and 64 filters for each of widths three, four, and five, followed by ReLU and max pooling. The BiLSTM uses 64-dimensional embeddings, a hidden size of 64 in each direction, and mean and max pooling. Both sequence models are trained for five epochs, with padding excluded from their pooled representations.

DistilBERT is loaded from a recorded, immutable source revision. Its encoder weights remain frozen and it runs in evaluation mode while producing the first CLS hidden state. Only the linear head is trained, for five epochs. CNN, BiLSTM, and the head use Adam at a learning rate of 0.001. Encoder batches contain eight prompts and training batches contain 32 examples.

The local RTX 3060 has 6 GB of VRAM, enough for these configurations. The MLP runs on CPU; the three PyTorch models run on the GPU. The sequence limit is 128 tokens. Regex tokenization for CNN and BiLSTM differs from DistilBERT's WordPiece tokenization, and the MLP uses the full text. These differences are part of the preliminary setup rather than evidence of an inherent architectural advantage.

\begin{table}[htbp]
\centering
\caption{Fixed-budget results on the shared corpus. JBB recall uses 15 harmful test goals; benign acceptance uses 100 separate matched requests.}
\label{tab:preliminary}
\small
\begin{tabular}{lrrrr}
\toprule
Model & Test $F_1$ & JBB recall & Held-out recall & Benign acceptance \\
\midrule''']
    lines.append(f"Length rule & {b['test']['f1']:.3f} & {b['test_jbb_harmful']['recall']:.3f} & {b['heldout']['recall']:.3f} & {b['hard_negative']['accuracy']:.3f} "+r'\\')
    names={'mlp':'TF--IDF MLP','cnn':'1D CNN','bilstm':'BiLSTM','distilbert':'Frozen DistilBERT'}
    for r in result['models']:
        lines.append(f"{names[r['model']]} & {r['test_f1']:.3f} & {r['jbb_test_recall']:.3f} & {r['heldout_recall']:.3f} & {r['benign_acceptance']:.3f} "+r'\\')
    lines.append(r'''\bottomrule
\end{tabular}
\end{table}

The separate slices are necessary to interpret the mixed-test score. A model can identify long DAN templates and still miss short harmful requests, or reject harmless requests that share a topic with harmful ones. The length rule's perfect held-out recall shows that this positive-only slice does not by itself demonstrate broad attack recognition. Its lack of benign examples prevents measuring false positives there.

The 15 harmful JBB test goals are a small diagnostic set; one changed prediction moves recall by about 6.7 percentage points. Error records retain identifiers, source tags, family tags, labels, and probabilities, allowing these mistakes to be examined without publishing prompt text. Alpaca's labels remain assumed benign, and residual template similarity makes the ordinary test partition easier than a fully grouped test.

\begin{table}[htbp]
\centering
\caption{Local execution cost. Single-request latency includes text processing and the full encoder, where applicable. CPU and GPU timings describe different deployed configurations.}
\label{tab:cost}
\small
\begin{tabular}{lrrr}
\toprule
Model & Trainable parameters & Training (s) & Request median (ms) \\
\midrule''')
    for r in result['models']:
        lines.append(f"{names[r['model']]} & {r['trainable_parameters']:,} & {r['training_seconds']:.2f} & {r['single_request_ms']:.2f} "+r'\\')
    lines.append(r'''\bottomrule
\end{tabular}
\end{table}

Training time excludes feature extraction and frozen-encoder passes, which are recorded separately. Latency is measured after warm-up on 20 test examples, with CUDA synchronization around GPU measurements. DistilBERT timing includes its encoder rather than only the inexpensive cached-feature head. Total parameters and trainable parameters are both reported in the JSON records because freezing reduces training cost without removing encoder memory or inference work.

The complete metrics include accuracy, precision, recall, Matthews correlation, ROC-AUC, trapezoidal PR-AUC, average precision, log loss, and Brier score. Their records share the same corpus hashes. These single-seed runs establish a reproducible starting point; additional seeds, validation-based tuning, reviewed benign labels, and a versioned grouped-template experiment are needed before drawing broader conclusions.
''')
    destination=ROOT/'report/sections/preliminary_results.tex'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
