# Training setup

## Expanded experiment

The larger condition uses WildJailbreak, pinned at revision `5ddc12a7894f842b0619b8e1c7ee496b198af009`. Its vanilla and adversarial benign examples avoid relying exclusively on short Alpaca instructions for the negative class. Labels describe prompt harmfulness; response safety and prompt injection labels from other datasets are not silently merged into this condition.

After normalization, deduplication, conflict removal, and request grouping, the experiment uses 100,000 training prompts, 39,502 validation prompts, and 38,911 test prompts. The eligible training pool had 183,005 rows; the 100,000-row cap gives every architecture the same fixed training budget. There are another 2,210 official evaluation prompts. The main split groups adversarial variants and their underlying vanilla request before assigning partitions. Exact overlap is excluded from official evaluation, but the official file lacks underlying request identifiers. See `results/large/data_manifest.json` and `data_audit.json` for the recorded checks and counts.

| Model | Configuration | Training location |
| --- | --- | --- |
| MLP (Ashlesh) | Hashed word 1–2 / character 3–5 TF-IDF, 32,768 features, 64-unit hidden layer; Adam 0.001, batch 256 | Local CPU, disk-backed feature batches |
| CNN (Aditya) | Training-only 20,000-token vocabulary, 64-dimensional embeddings, widths 3/4/5 and 64 filters; AdamW 0.001, batch 64 | Local RTX 3060 |
| BiLSTM (Deepam) | Same vocabulary/embeddings, 64 hidden units per direction, mean/max pooling; AdamW 0.001, batch 64 | Local RTX 3060 |
| DistilBERT (Deepam) | All encoder and classifier parameters trainable; AdamW 2e-5, weight decay 0.01, 10% warmup then linear decay, batch 32, FP16 and gradient checkpointing | Kaggle T4 GPU |

Each architecture runs three epochs with seed 4442 and selects its checkpoint by validation F1. Threshold stays at 0.5. Every model trains on every one of the 100,000 selected prompts each epoch. All have a 256-token limit, although regex and WordPiece tokenization differ. This is a basic baseline comparison, with no hyperparameter search or multi-seed claim.

System RAM availability, rather than the existence of a GPU, drives the split between local and Kaggle execution. The local machine has a working 6 GB RTX 3060, but only around 1–2 GB system RAM was free when the jobs started. Small sequence models fit easily; feature and token caches stay on disk. Kaggle provides more room for full transformer fine-tuning without competing with the local desktop. The CNN capacity benchmark is recorded separately from the completed training results.

Full fine-tuning checks are explicit: all encoder parameters require gradients, each encoder layer must receive nonzero gradients, and an encoder attention weight matrix must differ from its initial value after training. A frozen encoder with only a trained classifier cannot pass the expanded comparison checks.

The corpus remains largely synthetic. Grouping requests reduces leakage from variants of the same request but does not guarantee unseen tactics. Harmfulness labels do not establish whether an otherwise benign request is an instruction injection in a particular application context. Report main test F1/MCC/AUC alongside official harmful recall and benign acceptance. Keep the official class imbalance visible. Neither the 100,000-row count nor a high F1 establishes deployment robustness.

## Earlier small-corpus setup

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

## Team assignments

The synopsis assigns TF-IDF + MLP and acquisition to M. Ashlesh Mallya; CNN and preprocessing to S Aditya; BiLSTM, frozen DistilBERT, and evaluation to Deepam Ahuja. All four models are included in the assisted implementation and initial run. Shared training and metric utilities avoid duplicating the experimental protocol; they do not transfer the first two models to Deepam. Commit authorship is kept accurate, and the team should record their own subsequent review and implementation work separately.

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
