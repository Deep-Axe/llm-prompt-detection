# Recorded preliminary results

Fixed seed 4442, unchanged corpus partitions, and probability threshold 0.5. No hyperparameter search was performed.

| Model | Test F1 | JBB test recall | Held-out recall | Matched benign acceptance |
| --- | ---: | ---: | ---: | ---: |
| Length rule | 0.906 | 0.000 | 1.000 | 1.000 |
| mlp | 0.971 | 0.667 | 1.000 | 0.550 |
| cnn | 0.941 | 0.533 | 1.000 | 0.530 |
| bilstm | 0.951 | 0.733 | 1.000 | 0.430 |
| distilbert | 0.929 | 0.200 | 1.000 | 0.880 |

The 15 positive JBB test requests form a small diagnostic slice. The held-out slice has only positive examples, so it measures recall rather than overall classification quality. Alpaca is assumed benign; label review has not been completed. The length and template-similarity shortcuts remain.

The MLP uses full text and runs on CPU. CNN and BiLSTM use the first 128 regex tokens; DistilBERT uses 128 WordPiece tokens including special tokens, with only its linear head trained. Token budgets have the same numerical cap but different tokenization. CUDA timings are synchronized. Batch timings include text processing and, for DistilBERT, the encoder; cached embeddings are not used for latency measurement.

MLP and GPU-model timings are measured on different processors and cannot establish an intrinsic architecture speed ranking. JSON files retain full metrics, parameter counts, training/encoding times, GPU peak allocation, error identifiers, and corpus hashes. Checkpoints and downloaded weights remain local in ignored directories.

These are initial single-seed, fixed-budget results. They do not establish tuned performance or robustness to real indirect prompt injection.
