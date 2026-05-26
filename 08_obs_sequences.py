#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 08 — Retrieve observation sequence names
----------------------------------------------
Queries OPUS for the CASSINIobsname metadata field for every image that
the detection pipeline (step 07) processed successfully. Extracts the
flyby/sequence tag (e.g. '106EN') from each observation name.

Output → data/pipeline_output/obs_sequences.csv
  image_id         e.g. N1669796540
  CASSINIobsname   e.g. ISS_106EN_EQLBANDPF001_PRIME
  obs_sequence     e.g. 106EN

Usage
-----
  python 08_obs_sequences.py
  python 08_obs_sequences.py --workers 8
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from pyiss import query

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR        = Path(__file__).parent / "data"
PIPELINE_DIR    = DATA_DIR / "pipeline_output"
SUMMARY_FILE    = PIPELINE_DIR / "summary.parquet"
OUTPUT_CSV      = PIPELINE_DIR / "obs_sequences.csv"

# ─────────────────────────── helpers ──────────────────────────────────────────

def fetch_obsname(opusid: str, retries: int = 3, backoff: float = 2.0) -> str | None:
    for attempt in range(retries):
        result = query().param("opusid", opusid).fetch("opusid", "CASSINIobsname")
        if result.size > 0:
            return result.df["CASSINIobsname"].iloc[0]
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    return None


def extract_sequence(obsname: str | None) -> str | None:
    if obsname is None:
        return None
    parts = obsname.split("_")
    return parts[1] if len(parts) >= 2 else parts[0]

# ─────────────────────────── main ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve observation sequence names (step 08)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_parquet(SUMMARY_FILE)
    ok = df[df["status"] == "ok"][["opusid", "image_id"]].reset_index(drop=True)
    print(f"Successful detections: {len(ok)}")

    obsnames: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_obsname, row.opusid): row for row in ok.itertuples()}
        with tqdm(total=len(futures), desc="Querying OPUS") as pbar:
            for future in as_completed(futures):
                row = futures[future]
                obsnames[row.opusid] = future.result()
                pbar.update(1)

    records = [
        {
            "image_id":       row.image_id,
            "CASSINIobsname": obsnames.get(row.opusid),
            "obs_sequence":   extract_sequence(obsnames.get(row.opusid)),
        }
        for row in ok.itertuples()
    ]
    out_df = pd.DataFrame(records)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    n_missing = out_df["CASSINIobsname"].isna().sum()
    print(f"\nSaved → {OUTPUT_CSV}  ({len(out_df)} rows,  {n_missing} missing)")

    seq_counts = out_df["obs_sequence"].value_counts()
    print(f"\nUnique obs_sequences: {seq_counts.nunique()}")
    print(seq_counts.to_string())


if __name__ == "__main__":
    main()
