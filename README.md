# LLM prompt detection

Data acquisition for a comparison of MLP, CNN, BiLSTM, and frozen DistilBERT prompt classifiers.

```bash
python3 src/acquire_data.py
```

The downloader obtains JailbreakBench harmful and benign behaviors, the December 2023 DAN jailbreak prompts, and Stanford Alpaca. It reuses cached files in `data/raw/`. Raw data are excluded from Git. The URLs follow upstream `main`, so subsequent preprocessing will record checksums to identify the actual files used.

Source code is MIT-licensed. Third-party datasets retain their respective licenses; Alpaca data are subject to noncommercial terms. Model-training work is outside this initial acquisition step. Local reports are excluded through `report/` in `.gitignore`.
