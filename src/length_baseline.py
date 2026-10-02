"""Character-length baseline for the shared corpus.

Fits one threshold on the train split by F1, then scores validation, test,
JailbreakBench goals, and the held-out DAN families. Writes counts only.

  python3 src/length_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
OUT = ROOT / "data" / "processed" / "length_baseline.json"


def load() -> list[dict]:
    rows = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["n_chars"] = len(row["text"])
        rows.append(row)
    return rows


def scores(rows: list[dict], threshold: int) -> dict:
    tp = fp = fn = tn = 0
    by_source = {}
    for row in rows:
        pred = 1 if row["n_chars"] >= threshold else 0
        gold = int(row["label"])
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1
        else:
            tn += 1
        bucket = by_source.setdefault(row["source"], {"caught": 0, "missed": 0, "n": 0})
        if gold == 1:
            bucket["n"] += 1
            bucket["caught" if pred == 1 else "missed"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(rows),
        "accuracy": round((tp + tn) / len(rows), 3) if rows else None,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "adversarial_caught_by_source": by_source,
    }


def best_threshold(train: list[dict]) -> int:
    lengths = sorted({row["n_chars"] for row in train})
    step = max(1, len(lengths) // 400)
    best_t, best_f = lengths[0], -1.0
    for threshold in lengths[::step]:
        f1 = scores(train, threshold)["f1"]
        if f1 > best_f:
            best_t, best_f = threshold, f1
    return best_t


def main() -> int:
    if not CORPUS.exists():
        raise SystemExit("missing corpus; run python3 src/prepare_corpus.py")
    rows = load()
    groups = {}
    for row in rows:
        groups.setdefault(row["split"], []).append(row)
    hard_path = ROOT / "data" / "processed" / "hard_negatives.jsonl"
    hard = [json.loads(line) for line in hard_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in hard:
        row["n_chars"] = len(row["text"])
    groups["hard_negative"] = hard
    groups["test_jbb_harmful"] = [row for row in groups["test"] if row["source"] == "jbb_harmful"]
    groups["test_dan"] = [row for row in groups["test"] if row["source"] == "dan"]
    threshold = best_threshold(groups["train"])
    report = {
        "rule": "predict adversarial if character length >= threshold",
        "threshold_fit_on": "train",
        "threshold": threshold,
        "splits": {name: scores(group, threshold) for name, group in sorted(groups.items())},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"threshold": threshold, "test_f1": report["splits"]["test"]["f1"],
                      "heldout_f1": report["splits"].get("heldout", {}).get("f1")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
