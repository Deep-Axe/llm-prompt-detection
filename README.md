# LLM prompt detection

Dataset preparation and training code for TF-IDF MLP, 1D CNN, BiLSTM, and DistilBERT prompt classifiers. The expanded experiment uses 100,000 unique WildJailbreak training prompts and fine-tunes the full DistilBERT encoder. The earlier DAN/JailbreakBench/Alpaca experiment remains available as a separate small-corpus baseline.

## Expanded 100,000-row experiment

[WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) provides benign and harmful vanilla requests and adversarial variants. Access requires accepting its upstream terms. The task here is prompt harmfulness classification; a safe refusal in the response column does not make the input benign. Completions are excluded from the classifier input.

Preparation normalizes and deduplicates prompt text, removes conflicting labels, groups variants by their underlying vanilla request, and caps the training partition at exactly 100,000 unique prompts. Validation has 39,502 prompts, test has 38,911, and the official evaluation has 2,210. Exact prompts never repeat across partitions. The main partitions share no underlying requests; the official evaluation does not expose those identifiers, so request overlap there cannot be checked.

```bash
hf download allenai/wildjailbreak train/train.tsv eval/eval.tsv README.md \
  --repo-type dataset --revision 5ddc12a7894f842b0619b8e1c7ee496b198af009 \
  --local-dir data/raw/large/wildjailbreak --cache-dir data/.hf-cache
python src/prepare_large_corpus.py
python src/audit_large_corpus.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/train_large_mlp.py
python src/train_large_neural.py cnn
python src/train_large_neural.py bilstm
```

MLP trains on the local CPU; CNN and BiLSTM use the local RTX 3060. Full DistilBERT fine-tuning runs on Kaggle because of its larger memory requirements and the limited free system RAM here. The private Kaggle input contains the same checksummed partitions and pinned pretrained encoder as the local run:

```bash
python src/download_encoder.py
python src/package_kaggle.py --username YOUR_KAGGLE_USERNAME
kaggle datasets create -p kaggle/input --keep-tabular
# Wait until `kaggle datasets status USER/wildjailbreak-100k-grouped` says ready.
kaggle kernels push -p kaggle/distilbert --accelerator NvidiaTeslaT4
kaggle kernels status USER/prompt-detection-distilbert-100k
```

All models train for three epochs and select their checkpoint using validation F1. Test data are used only after selection. DistilBERT uses AdamW at 2e-5, mixed precision, gradient checkpointing, and updates every encoder layer. Runtime checks verify encoder gradients and changed weights. MLP uses hashed word/character TF-IDF with training-only IDF to keep RAM bounded. The 256-token context budget uses regex tokens for MLP/CNN/BiLSTM and WordPiece for DistilBERT.

Expanded results and provenance are written to `results/large/`; weights and individual predictions go to ignored `artifacts/large/`. Retrieve the Kaggle result JSON into `results/large/distilbert.json`, then run `python src/compare_large_results.py`. The comparison rejects mismatched partitions or a frozen DistilBERT run. Training time includes validation and differs by hardware; it is not an architecture latency comparison.

## Earlier small-corpus baseline

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

## Model assignments

The assignments follow the project synopsis; the shared runner does not change model ownership.

| Model | Assigned member | Entry command |
| --- | --- | --- |
| TF-IDF + MLP | M. Ashlesh Mallya | `python src/train_mlp.py` |
| 1D CNN | S Aditya | `python src/train_cnn.py --device cuda` |
| BiLSTM | Deepam Ahuja | `python src/train_bilstm.py --device cuda` |
| Frozen DistilBERT + linear head | Deepam Ahuja | `python src/train_distilbert.py --device cuda` |

Ashlesh also owns acquisition, Aditya the shared preprocessing, and Deepam the evaluation harness and held-out analysis. This repository update was implemented and run with coding-agent assistance. Assignments describe the team's responsibilities; they do not claim that each member authored these commits. Each member should review, explain, and develop the model assigned to them.

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
