#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 02 — Infer observation sets and build CLEAR–science pairs
---------------------------------------------------------------
For each CLEAR image from step 01, calls PyISS infer_set() to find all
images taken in the same observation sequence. Groups the results into
(CLEAR, science_filter) pairs — the units for CLEAR subtraction in step 05.

Optimisation: once a CLEAR image appears in any inferred set, its sequence
is already covered; it is skipped as a future seed. This reduces ~473
API round-trips to ~N unique sequences (typically far fewer).

Outputs → data/inferred_sets/
  pairs.parquet      one row per (CLEAR, science) pair, sorted by clear_time1
  sets_raw.parquet   every image seen across all inferred sets
  no_match.csv       CLEAR images whose sequence contained no science filter
  metadata.json      run stats, filter breakdown, timestamp
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from pyiss import infer_set

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR  = Path(__file__).parent / "data"
QUERY_DIR = DATA_DIR / "query_results"
OUT_DIR   = DATA_DIR / "inferred_sets"

# ─────────────────────────── load CLEAR images ────────────────────────────────

clear_df = pd.read_parquet(QUERY_DIR / "query_results.parquet")
opusids  = clear_df["opusid"].tolist()
print(f"Input: {len(opusids)} CLEAR images")

pairs, sets_raw, no_match = [], [], []
seen_clears = set()

# ─────────────────────────── infer sets ───────────────────────────────────────

for clear_id in tqdm(opusids, unit="img", desc="Inferring sets"):
    if clear_id in seen_clears:
        continue

    s = infer_set(clear_id)
    set_df = s.df.copy()
    set_df["seed_opusid"] = clear_id
    sets_raw.append(set_df)

    clears_in_set  = set_df[set_df["COISSfilter"] == "CLEAR"]
    science_in_set = set_df[set_df["COISSfilter"] != "CLEAR"]

    seen_clears.update(clears_in_set["opusid"].tolist())

    if science_in_set.empty:
        no_match.append(clear_id)
        continue

    clear_times = pd.to_datetime(clears_in_set["time1"], utc=True)
    for _, sci in science_in_set.iterrows():
        sci_t = pd.to_datetime(sci["time1"], utc=True)
        dt    = (clear_times - sci_t).abs().dt.total_seconds()
        best  = clears_in_set.loc[dt.idxmin()]
        pairs.append({
            "clear_opusid":   best["opusid"],
            "science_opusid": sci["opusid"],
            "science_filter": sci["COISSfilter"],
            "clear_time1":    best["time1"],
            "science_time1":  sci["time1"],
            "dt_s":           float(dt.min()),
        })

# ─────────────────────────── assemble & save ──────────────────────────────────

pairs_df = (
    pd.DataFrame(pairs)
    .drop_duplicates(subset=["clear_opusid", "science_opusid"])
    .sort_values(["clear_time1", "science_time1"])
    .reset_index(drop=True)
) if pairs else pd.DataFrame(columns=["clear_opusid", "science_opusid",
                                       "science_filter", "clear_time1",
                                       "science_time1", "dt_s"])

sets_df = (
    pd.concat(sets_raw, ignore_index=True)
    .drop_duplicates(subset=["opusid"])
    .sort_values("time1")
    .reset_index(drop=True)
) if sets_raw else pd.DataFrame()

OUT_DIR.mkdir(parents=True, exist_ok=True)
pairs_df.to_parquet(OUT_DIR / "pairs.parquet", index=False)
sets_df.to_parquet(OUT_DIR / "sets_raw.parquet", index=False)
pd.DataFrame({"opusid": no_match}).to_csv(OUT_DIR / "no_match.csv", index=False)

filter_counts = pairs_df["science_filter"].value_counts().to_dict() if not pairs_df.empty else {}
(OUT_DIR / "metadata.json").write_text(json.dumps({
    "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
    "clear_images_in": len(opusids),
    "unique_sets":     len(sets_raw),
    "unique_pairs":    len(pairs_df),
    "no_match_clears": len(no_match),
    "science_filters": filter_counts,
}, indent=2))

print(f"\n  {len(pairs_df):,} pairs  |  {len(sets_raw)} unique sets  |  {len(no_match)} unmatched")
print(f"  → {OUT_DIR}")
if filter_counts:
    print("\n  Science filters:")
    for filt, n in sorted(filter_counts.items(), key=lambda x: -x[1]):
        print(f"    {filt:25s}  {n:4d} pairs")
