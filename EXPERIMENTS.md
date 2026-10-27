# Training and repeated-seed evaluation

The completed 100,000-row baseline remains in `results/large/` and `artifacts/large/`. New runs use separate directories under `results/experiments/` and `artifacts/experiments/`.

## Frozen protocol

[configs/training_protocol.json](configs/training_protocol.json) records:

- Baseline results commit, current repository commit, complete baseline reports, and baseline file hashes.
- Dataset revision, partition hashes, and pretrained encoder sources.
- Seeds 4442, 4443, and 4444; threshold 0.5; validation F1 checkpoint selection.
- Epoch caps of ten for MLP/CNN/BiLSTM and five for pretrained encoders, patience two, and minimum improvement zero.
- Model dimensions, batch sizes, learning rates, and the 256-token limit.

MLP uses a 256-row feature batch per `partial_fit` call in the new runner. The earlier runner used 1,000-row feature chunks with internal 256-row optimizer batches. The extended runner's code hashes identify this change in batching and its per-epoch shuffle rule. Keep these implementation differences visible when comparing with the historical baseline.

To freeze another protocol version, use a new filename:

```bash
python src/freeze_experiment_protocol.py --output configs/training_protocol_v2.json
```

The existing protocol file is never silently replaced.

## Run or resume an experiment

```bash
python src/train_experiment.py cnn --seed 4442 --device cuda
python src/train_experiment.py cnn --seed 4442 --device cuda --resume
```

Available architectures are `mlp`, `cnn`, `bilstm`, `distilbert`, and `deberta`. MLP runs on CPU; select `--device cpu`. Pretrained encoders require a local encoder snapshot and a `source.json` that matches the pinned repository and revision. The default snapshot is DistilBERT; supply `--encoder` for DeBERTa. The larger-model download and training remain a later experiment.

Each run has paths of the form:

```text
results/experiments/wildjailbreak-100k/cnn/seed_4442/
    config.json
    status.json
    result.json
artifacts/experiments/wildjailbreak-100k/cnn/seed_4442/
    cache/
    last.pt
    best.pt
    *_predictions.npz
```

MLP uses `last.joblib` and `best.joblib`. The last checkpoint holds the current model and optimizer, training cursor, validation history, early-stopping state, and random state. Neural checkpoints also hold scheduler and mixed-precision scaler state. Checkpoints are written atomically at optimizer-update boundaries and after validation. Resume restores the next batch of the interrupted epoch and checks the protocol, data, code, and configuration fingerprint.

A completed run evaluates the best validation checkpoint and writes metrics and predictions. The new runner supports full encoder training for both encoder architectures. Gradient checks cover every encoder layer and verify a changed encoder weight.

## Short save/resume verification

```bash
python src/verify_experiment_resume.py --device cuda
```

For each existing architecture, this command runs an uninterrupted reference, interrupts a separate run after its first successful optimizer update, and resumes it in a new process. It compares final weights, predictions, training loss, selected epoch, update counts, and row counts. It also checks the baseline inventory before and after verification.

These checks use 512 training prompts and up to 128 examples per evaluation partition for two epochs. Every record has `smoke_only: true`; the check is for recovery behavior, and its metrics are excluded from experiment rankings. Three-seed full training has not been launched by this verification command.

## Kaggle verification

The verification notebook uses the existing private WildJailbreak input and a separate private code/protocol dataset. It has its own notebook ID and does not replace the completed DistilBERT notebook.

```bash
python src/package_protocol_kaggle.py --username YOUR_USERNAME
kaggle datasets create -p kaggle/input/protocol_runners --keep-tabular
kaggle datasets status YOUR_USERNAME/prompt-detection-protocol-runners-v1
# Submit after the code dataset is ready.
kaggle kernels push -p kaggle/protocol_smoke --accelerator NvidiaTeslaT4
python src/collect_protocol_verification.py \
  --kernel YOUR_USERNAME/prompt-detection-protocol-resume-check --watch
```

The notebook metadata pins the Docker image used by the original encoder run so its Python version supports the recorded library pins. The collector imports only the small audit reports and confirms that the local baseline files are unchanged. Training checkpoints stay in Kaggle output and ignored local folders.

## Aggregate repeated runs

```bash
python src/aggregate_experiments.py --root results/experiments
```

Aggregation groups complete runs by protocol, condition, and architecture. It reports mean and sample standard deviation across available seeds and indicates whether all three scheduled seeds are present. Smoke records are excluded. Duplicate seeds, inconsistent partitions or model settings, and unverified encoder updates are rejected. A single-seed standard deviation is left empty.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover RNG restoration, early-stopping state across checkpoints, aggregation of additional architectures, duplicate seeds, changed partitions, and smoke exclusion. The GPU smoke notebook separately checks optimizer, scheduler, mixed-precision, and encoder state during interrupted training.
