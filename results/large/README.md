# 100,000-row model comparison

WildJailbreak prompt harmfulness classification. All four models use the same 100,000 unique training prompts, 39,502 validation prompts, and 38,911 test prompts. The 2,210 official evaluation prompts form an additional check.

| Model | Test accuracy | Test F1 | Test MCC | Official harmful recall | Official benign acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp | 0.9353 | 0.9355 | 0.8710 | 0.8000 | 0.5857 |
| cnn | 0.9258 | 0.9290 | 0.8528 | 0.8585 | 0.4476 |
| bilstm | 0.9410 | 0.9423 | 0.8820 | 0.8570 | 0.5571 |
| distilbert | 0.9789 | 0.9792 | 0.9578 | 0.9055 | 0.6000 |

Three epochs per architecture; checkpoint selection uses validation F1 only. The decision threshold is 0.5. CNN/BiLSTM use learned embeddings, and all DistilBERT encoder layers are updated. MLP uses hashed word/character TF-IDF to bound RAM; its IDF is fitted only on training data.

The main partitions keep each underlying vanilla request and its adversarial variants together. Exact normalized duplicates and conflicting labels are removed. The official evaluation does not expose underlying vanilla requests, so only exact prompt overlap can be checked there. Grouping does not guarantee that different requests never use similar attack tactics.

All models have a 256-token context budget, but regex and WordPiece tokenization differ. These are single-seed baseline runs on a largely synthetic corpus, with upstream labels retained. Harmfulness detection is not a complete test of context-dependent prompt injection. The official evaluation is mostly harmful, so benign acceptance must be read alongside F1.

The accompanying uncertainty.json gives paired 95% bootstrap intervals over underlying test request groups (1,000 resamples). These measure test sampling uncertainty, not training-seed variation or annotation quality.

Hardware differs between local and Kaggle training; training times describe these runs and should not be treated as an architecture speed ranking. Weights, prompt data, and individual predictions remain outside Git. See the JSON reports for metrics, validation histories, code hashes, and partition hashes.
