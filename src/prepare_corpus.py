"""Acquire and preprocess the shared prompt-classification corpus.

Downloads, if the files are not already present:

  JailbreakBench behaviors, harmful and benign splits
  Shen et al. in-the-wild jailbreak prompts, 2023-12-25 release
  Stanford Alpaca instructions

The 107,250-question harm set and the JailbreakBench judge-comparison
artifacts are not part of this corpus. Run from the repository root:

  python3 src/prepare_corpus.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 4442
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
MIN_COMMUNITY = 10
# Modulo 5 holds out nothing in the 2023-12-25 community ids. Modulo 3
# holds out about 15% of the DAN prompts among communities large enough
# to score on their own.
HOLDOUT_MOD = 3
MIN_GOAL_CHARS = 40
MIN_ALPACA_CHARS = 20

from acquire_data import SOURCES, download

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
WS = re.compile(r"\s+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return WS.sub(" ", text).strip().casefold()


def display(text: str) -> str:
    return WS.sub(" ", text).strip()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stable_id(source: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{source}-{digest}"


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    kept = []
    dropped = 0
    for row in rows:
        if row["key"] in seen:
            dropped += 1
            continue
        seen.add(row["key"])
        kept.append(row)
    return kept, dropped


def load_jbb(path: Path, source: str, label: int) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        text = display(raw.get("Goal") or "")
        key = norm(text)
        if not key:
            continue
        rows.append({
            "id": stable_id(source, key),
            "text": text,
            "key": key,
            "label": label,
            "source": source,
            "family": f"jbb:{(raw.get('Category') or 'unknown').strip()}",
            "jbb_source": (raw.get("Source") or "").strip(),
        })
    return rows


def load_dan(path: Path) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        flag = str(raw.get("jailbreak", "")).strip().lower()
        if flag not in {"true", "1", "yes"}:
            continue
        text = display(raw.get("prompt") or "")
        key = norm(text)
        if not key:
            continue
        community = str(raw.get("community_id") or "").strip()
        if community.endswith(".0"):
            community = community[:-2]
        family = f"dan:{community}" if community else "dan:unknown"
        rows.append({
            "id": stable_id("dan", key),
            "text": text,
            "key": key,
            "label": 1,
            "source": "dan",
            "family": family,
            "platform": (raw.get("platform") or "").strip(),
        })
    return rows


def load_alpaca(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for raw in data:
        instruction = display(raw.get("instruction") or "")
        context = display(raw.get("input") or "")
        text = f"{instruction}\n{context}" if context else instruction
        key = norm(text)
        if len(key) < MIN_ALPACA_CHARS:
            continue
        rows.append({
            "id": stable_id("alpaca", key),
            "text": text,
            "key": key,
            "label": 0,
            "source": "alpaca",
            "family": "alpaca",
        })
    return rows


def drop_overlapping_alpaca(alpaca: list[dict], harmful_keys: set[str], goals: list[str]) -> tuple[list[dict], dict]:
    """Drop benign rows that repeat a harmful string.

    Full DAN templates are compared by exact normalized text. JailbreakBench
    goals are also treated as contained when the whole goal appears inside a
    longer Alpaca instruction. Short goals are ignored so a few ordinary words
    do not wipe out unrelated instructions.
    """
    goals = [g for g in goals if len(g) >= MIN_GOAL_CHARS]
    kept = []
    exact = contained = 0
    for row in alpaca:
        if row["key"] in harmful_keys:
            exact += 1
            continue
        if any(goal in row["key"] for goal in goals):
            contained += 1
            continue
        kept.append(row)
    return kept, {"exact": exact, "contained_goal": contained}


def heldout_families(dan_rows: list[dict]) -> set[str]:
    counts = Counter(row["family"] for row in dan_rows)
    chosen = set()
    for family, count in counts.items():
        if family == "dan:unknown" or count < MIN_COMMUNITY:
            continue
        community = family.split(":", 1)[1]
        if community.isdigit() and int(community) % HOLDOUT_MOD == 0:
            chosen.add(family)
    return chosen


def assign_splits(rows: list[dict], rng: random.Random) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["source"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["id"])
        rng.shuffle(group)
        n = len(group)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
        for i, row in enumerate(group):
            if i < train_end:
                row["split"] = "train"
            elif i < val_end:
                row["split"] = "val"
            else:
                row["split"] = "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    fields = ("id", "text", "label", "source", "family", "split")
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({key: row[key] for key in fields}, ensure_ascii=False) + "\n")


def counts_by(rows: list[dict], *keys: str) -> dict:
    counter = Counter(tuple(row[key] for key in keys) for row in rows)
    return {"|".join(map(str, key)): value for key, value in sorted(counter.items())}


def main() -> int:
    paths = {name: download(name) for name in SOURCES}
    jbb_harm = load_jbb(paths["jbb_harmful"], "jbb_harmful", 1)
    jbb_benign = load_jbb(paths["jbb_benign"], "jbb_benign", 0)
    dan = load_dan(paths["dan"])
    alpaca = load_alpaca(paths["alpaca"])
    loaded = {
        "jbb_harmful": len(jbb_harm),
        "jbb_benign": len(jbb_benign),
        "dan_flagged_jailbreak": len(dan),
        "alpaca_long_enough": len(alpaca),
    }

    jbb_harm, jbb_harm_dup = dedupe(jbb_harm)
    jbb_benign, jbb_benign_dup = dedupe(jbb_benign)
    dan, dan_dup = dedupe(dan)
    alpaca, alpaca_dup = dedupe(alpaca)

    # A string that appears in both releases keeps the DAN row, because that
    # row carries the community id used for the held-out family split.
    dan_keys = {row["key"] for row in dan}
    jbb_goals = [row["key"] for row in jbb_harm]
    overlap_jbb_dan = sum(1 for row in jbb_harm if row["key"] in dan_keys)
    jbb_harm = [row for row in jbb_harm if row["key"] not in dan_keys]
    harmful_keys = dan_keys | {row["key"] for row in jbb_harm}
    benign_keys = {row["key"] for row in jbb_benign}
    alpaca, alpaca_filter = drop_overlapping_alpaca(alpaca, harmful_keys | benign_keys, jbb_goals)
    alpaca_after_filter = len(alpaca)

    hold = heldout_families(dan)
    heldout = []
    ind_adv = []
    for row in dan + jbb_harm:
        if row["family"] in hold:
            row["split"] = "heldout"
            heldout.append(row)
        else:
            ind_adv.append(row)

    rng = random.Random(SEED)
    alpaca.sort(key=lambda row: row["id"])
    rng.shuffle(alpaca)
    alpaca = alpaca[:len(ind_adv)]

    assign_splits(ind_adv + alpaca, random.Random(SEED))
    for row in jbb_benign:
        row["split"] = "hard_negative"

    corpus = ind_adv + alpaca + heldout
    corpus.sort(key=lambda row: (row["split"], row["id"]))
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "corpus.jsonl", corpus)
    write_jsonl(OUT / "hard_negatives.jsonl", jbb_benign)

    stats = {
        "seed": SEED,
        "split": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
        "holdout_rule": f"DAN community_id % {HOLDOUT_MOD} == 0 and count >= {MIN_COMMUNITY}",
        "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "loaded": loaded,
        "alpaca_after_filter": alpaca_after_filter,
        "duplicates_removed": {
            "jbb_harmful": jbb_harm_dup,
            "jbb_benign": jbb_benign_dup,
            "dan": dan_dup,
            "alpaca": alpaca_dup,
            "jbb_harmful_already_in_dan": overlap_jbb_dan,
        },
        "alpaca_removed_as_harmful_or_hard_negative": alpaca_filter,
        "alpaca_sampled": len(alpaca),
        "heldout_families": sorted(hold),
        "heldout_rows": len(heldout),
        "dan_without_community_id": sum(1 for row in dan if row["family"] == "dan:unknown"),
        "in_distribution": counts_by(ind_adv + alpaca, "split", "label", "source"),
        "heldout_by_family": counts_by(heldout, "family"),
        "hard_negatives": len(jbb_benign),
        "corpus_rows": len(corpus),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("corpus_rows", "heldout_rows", "hard_negatives", "alpaca_sampled", "in_distribution")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
