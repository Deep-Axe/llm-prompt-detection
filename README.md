# LLM prompt detection

Dataset preparation and a controlled comparison of TF-IDF MLP, 1D CNN, BiLSTM, and frozen DistilBERT classifiers for adversarial LLM inputs. The corpus combines DAN jailbreak templates, JailbreakBench harmful goals, and Alpaca instructions, with separate held-out communities and matched benign requests.

## Prepare the data

```bash
python3 src/acquire_data.py
python3 src/prepare_corpus.py
pip install -r requirements.txt
python3 src/audit_data.py
python3 src/length_baseline.py
```

Existing raw files are reused. The processed corpus has 1,764 training, 378 validation, 378 test, and 196 held-out positive rows. A separate file contains 100 matched benign JBB requests. The raw checksums and fixed seed are recorded in `data/processed/stats.json`. Prompt text stays local and ignored by Git.

Alpaca labels are assumed benign; overlap checks are not a safety review. The corpus has length and template-similarity shortcuts. [DATA_STRATEGY.md](DATA_STRATEGY.md) documents these limitations and additional data conditions.

## Train locally

Python 3.10 and an RTX 3060 with 6 GB VRAM are used for the recorded GPU runs. [TRAINING_PLAN.md](TRAINING_PLAN.md) describes the hardware decision, configurations, and evaluation protocol. The CUDA 12.1 wheel is compatible with the installed driver; see the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-training.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/run_experiments.py --device cuda
```

The runner audits the existing corpus, reproduces the length diagnostic, trains the MLP, downloads an immutable DistilBERT snapshot, and trains the CNN, BiLSTM, and frozen-encoder head. Set `--device cpu` to run sequence models without CUDA. Checkpoints are written to `artifacts/`, which is ignored. CPU and GPU runs will have different timing characteristics.

Individual commands:

```bash
python src/train_mlp.py
python src/train_neural.py cnn --device cuda
python src/train_neural.py bilstm --device cuda
python src/download_encoder.py
python src/train_neural.py distilbert --device cuda
python src/summarize_results.py
python -m unittest discover -s tests -v
```

The optional `src/train_cnn_numpy.py` is an earlier educational CPU baseline and writes a separate `cnn_numpy.json`; it is not the CNN row in the four-model comparison.

## Results and reproducibility

[results/README.md](results/README.md) summarizes the recorded results. JSON and CSV retain the classification metrics, calibration scores, source slices, parameter counts, timings, errors by identifier, and dataset hashes. Single-class slices use recall or benign acceptance, not F1/AUC. PR-AUC and average precision are reported separately.

No hyperparameter search is performed. Models use identical corpus partitions, with documented differences in representations, truncation, budgets, and hardware. Strong overall scores should be interpreted alongside the length baseline, short harmful requests, matched benign requests, and held-out communities.

`src/download_extensions.py` supports optional candidate datasets using `requirements-acquisition.txt`; it does not merge them into the current corpus. The report is kept in `report/` and remains ignored for now. To update its result section, run `python src/export_results.py` and `make -C report`.

## License

Source code is MIT-licensed. Datasets and pretrained models retain their respective licenses. Alpaca data have noncommercial terms; the repository license does not replace them. Attribution is retained in the report bibliography and source metadata.
