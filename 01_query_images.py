#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 01 — Query OPUS for Cassini ISS CLEAR images
---------------------------------------------------
Queries the OPUS API for all Cassini ISS CLEAR images of Enceladus that
satisfy the E-ring forward-scattering geometry.

Parameters are read from the QUERY_PARAMS_JSON environment variable when
called by 00_run_pipeline.py, or fall back to the defaults defined below.

Outputs → data/query_results/
  query_results.parquet   full table (reload with pd.read_parquet)
  query_manifest.csv      lightweight ID list
  query_metadata.json     params, timestamp, row count
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyiss import query

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = DATA_DIR / "query_results"

# ─────────────────────────── parameters ───────────────────────────────────────
# When called via 00_run_pipeline.py these are injected through the environment.
# Edit the defaults here when running this script standalone.

_raw = os.environ.get("QUERY_PARAMS_JSON")
if _raw:
    _cfg   = json.loads(_raw)
    PARAMS = _cfg["params"]
    LIMIT  = int(_cfg["limit"])
else:
    PARAMS = {
        "target":                          "ENCELADUS",
        "COISSfilter":                     "CLEAR",
        "phase1":                          150,
        "phase2":                          180,
        "RINGGEOringradius1":              180_000,
        "RINGGEOringradius2":              480_000,
        "RINGGEOobserverringelevation1":   -5,
        "RINGGEOobserverringelevation2":   5,
        "RINGGEOsolarringopeningangle1":   -15,
        "RINGGEOsolarringopeningangle2":   15,
    }
    LIMIT = 100_000

# ─────────────────────────── query ────────────────────────────────────────────

q = query()
for name, value in PARAMS.items():
    q = q.param(name, value)

print(f"Querying OPUS ({len(PARAMS)} parameters, limit={LIMIT}) …")
result = (
    q.limit(LIMIT)
     .fetch("opusid", "time1", "time2", "target", "COISSfilter", "COISScamera",
            "duration",
            "RINGGEOringradius1", "RINGGEOringradius2",
            "RINGGEOobserverringelevation1", "RINGGEOobserverringelevation2",
            "RINGGEOsolarringopeningangle1", "RINGGEOsolarringopeningangle2",
            "RINGGEOphaseangle1",
            "CASSINIdistance1", "CASSINIdistance2")
)
df = result.df.copy()
print(f"  {len(df)} images returned")

# ─────────────────────────── save ─────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_DIR / "query_results.parquet", index=False)
df[["opusid", "time1", "COISSfilter", "COISScamera"]].to_csv(
    OUT_DIR / "query_manifest.csv", index=False)
(OUT_DIR / "query_metadata.json").write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "n_results":     len(df),
    "params":        PARAMS,
    "limit":         LIMIT,
}, indent=2))

print(f"  → {OUT_DIR}")
