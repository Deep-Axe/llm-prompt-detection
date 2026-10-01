# LLM prompt detection

Data acquisition for a comparison of MLP, CNN, BiLSTM, and frozen DistilBERT prompt classifiers.

```bash
python3 src/acquire_data.py
```

The downloader obtains JailbreakBench harmful and benign behaviors, the December 2023 DAN jailbreak prompts, and Stanford Alpaca. It reuses cached files in `data/raw/`. Raw data are excluded from Git. The URLs follow upstream `main`, so subsequent preprocessing will record checksums to identify the actual files used.

Source code is MIT-licensed. Third-party datasets retain their respective licenses; Alpaca data are subject to noncommercial terms. Model-training work is outside this initial acquisition step. Local reports are excluded through `report/` in `.gitignore`.

## Prepare the corpus

```bash
python3 src/prepare_corpus.py
```

Text is normalized and exact duplicates are removed. The pipeline retains source and family tags, samples Alpaca, and creates fixed 70/15/15 partitions with seed 4442. The main file has 1,764 training, 378 validation, 378 test, and 196 held-out DAN rows. Another 100 matched JBB benign requests are kept separately. Counts and raw checksums are recorded in `data/processed/stats.json`; prompt JSONL files remain local and ignored.

Alpaca is assumed benign. The overlap filter is not a semantic safety review. The positive class mixes direct harmful requests and jailbreak templates; those sources should retain separate evaluation slices.
