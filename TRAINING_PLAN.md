# Training setup and remaining work

The local machine has an AMD Ryzen 9 5900HX, 16 GB system RAM, and an NVIDIA RTX 3060 laptop GPU with 6 GB VRAM. At setup, about 5 GB VRAM and 2.5 GB system RAM were available. `nvidia-smi` reports driver 535.309.01. The experiment uses PyTorch 2.5.1 with the CUDA 12.1 runtime, avoiding a driver upgrade.

Local training is appropriate for 1,764 training rows. Small learned-embedding sequence models fit comfortably in the GPU. DistilBERT remains frozen and processes eight prompts at a time; cached representations allow inexpensive head training. This avoids sending the prompt corpus to a new service. Kaggle would be useful for larger runs or multi-seed searches, but is unnecessary for this initial comparison.

## Fixed protocol

The imported corpus and community holdout are unchanged. All models share row identifiers, labels, training/validation/test partitions, and source-specific evaluation. Use seed 4442 and threshold 0.5. The validation metrics are recorded without hyperparameter or checkpoint selection. This is a fixed-budget preliminary run, not a tuned benchmark.

| Model | Representation and configuration | Budget |
| --- | --- | --- |
| MLP | Training-fitted word 1–2 grams and character 3–5 grams, up to 8,000 features each; 64-unit hidden layer | 20 epochs, batch 64, CPU |
| CNN | Trainable 64-dimensional embeddings; widths 3/4/5, 64 filters each, ReLU and max pooling | 5 epochs, batch 32, GPU |
| BiLSTM | Trainable 64-dimensional embeddings; hidden size 64 in each direction; mean and max pooling | 5 epochs, batch 32, GPU |
| DistilBERT | Frozen `distilbert-base-uncased`, first CLS representation, one trainable linear head | Encoder batch 8; head 5 epochs, batch 32, GPU |

Sequence models use a limit of 128 tokens. Regex and WordPiece tokenization differ, and the MLP uses full text. Record those representation differences when interpreting the results. The sequence optimizers use Adam at 0.001 with gradient clipping; no hyperparameter search is performed. DistilBERT source revision and hashes are committed separately from its downloaded weights.

## Evaluation

Report accuracy, precision, recall, F1, MCC, ROC-AUC, trapezoidal PR-AUC, average precision, log loss, and Brier score on mixed-class partitions. Report recall on positive-only slices and acceptance on benign-only slices. Include total and trainable parameters, feature/encoder time, weight-training time, synchronized batch cost, and warmed single-request latency. Retain prompt IDs and source/family tags in error records; omit prompt text from published diagnostics.

The length diagnostic and source slices are essential: DAN templates are much longer than Alpaca, and many ordinary test templates resemble training examples. Neither high mixed F1 nor positive-only holdout recall alone is evidence of deployment robustness. The matched benign slice and short harmful JBB requests reveal different failure modes.

## After the preliminary run

1. Repeat chosen configurations across several seeds; tune only on validation data.
2. Review assumed benign labels and obtain longer realistic benign prompts.
3. Define a second corpus condition with related template variants grouped before splitting; run every architecture under that condition.
4. Evaluate prompt injection as a separately documented condition that preserves its label meaning and application context.
5. Compare latency on the same processor if the claim concerns architecture rather than the deployed CPU/GPU configuration.

The report remains local and ignored by Git. The code, aggregate results, and source manifests are published; raw data, processed prompt text, checkpoints, caches, and downloaded model weights remain ignored.
