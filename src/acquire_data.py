"""Download the source releases; cached files are reused.

Run from the repository root: python3 src/acquire_data.py
"""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES = {
    "jbb_harmful": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
    "jbb_benign": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/benign-behaviors.csv",
    "dan": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv",
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
}
FILENAMES = {"jbb_harmful": "jbb_harmful.csv", "jbb_benign": "jbb_benign.csv",
             "dan": "jailbreak_prompts_2023_12_25.csv", "alpaca": "alpaca_data.json"}


def download(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / FILENAMES[name]
    if dest.exists() and dest.stat().st_size > 0:
        print(f"cached {dest.name}", flush=True)
        return dest
    print(f"downloading {dest.name}", flush=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(SOURCES[name], partial)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    for name in SOURCES:
        download(name)
