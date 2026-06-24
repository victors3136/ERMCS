"""
inspect_data.py
================
Run this FIRST, the moment the four dataset files are in place inside ./data/

    dataset.35past.Linux10k.json
    dataset.pass.json
    nft-121.json
    nft-123.json

Purpose
-------
Before trusting any numbers out of reproduce.py, we need to confirm the real
column names, dtypes, and label encoding actually match what the paper's
text and the authors' notebook imply. Real replication packages frequently
have small undocumented quirks (renamed columns, different label encodings,
extra/missing fields) -- finding and writing these down IS part of a
replication study, not a failure of it. Capture whatever this script prints
into your report's "Setup / Deviations" section.

Usage
-----
    python inspect_data.py
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "dataLinux": "dataset.35past.Linux10k.json",
    "dataPass": "dataset.pass.json",
}
NFT_FILES = {
    "nft121": "nft-121.json",
    "nft123": "nft-123.json",
}


def inspect_dataframe(name: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("\ndtypes:")
    print(df.dtypes)
    if "label" in df.columns:
        print("\nlabel value counts:")
        print(df["label"].value_counts())
    if "buildId" in df.columns:
        print("\nbuildId range:", df["buildId"].min(), "to", df["buildId"].max())
        print("unique buildIds:", df["buildId"].nunique())
    if "testSource" in df.columns:
        lengths = df["testSource"].astype(str).str.len()
        print("\ntestSource char length: min/median/max =",
              lengths.min(), lengths.median(), lengths.max())
        empty = (df["testSource"].astype(str).str.strip() == "").sum()
        print(f"rows with empty testSource: {empty} ({empty / len(df):.1%})")
    print("\nfirst row sample:")
    print(df.iloc[0].to_dict() if len(df) else "EMPTY DATAFRAME")


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist. Create it and place the "
              f"four dataset files inside, then re-run this script.")
        return

    for name, fname in FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"MISSING: {path} -- download it from the figshare link "
                  f"in the replication package README and place it here.")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\nLoading {fname} ({size_mb:.1f} MB) ...")
        df = pd.read_json(path)
        inspect_dataframe(name, df)

    for name, fname in NFT_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(f"type: {type(data)}")
        if isinstance(data, list):
            print(f"length: {len(data)}")
            print(f"first 5 entries: {data[:5]}")
        elif isinstance(data, dict):
            print(f"keys (first 5): {list(data.keys())[:5]}")


if __name__ == "__main__":
    main()
