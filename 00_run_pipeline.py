#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 00 — Pipeline orchestrator
---------------------------------
Single entry point that runs steps 01–06 in sequence.

Edit the QUERY_PARAMS block below to define your observation search,
then run this script. Individual steps can also be run standalone.

Usage
-----
  python 00_run_pipeline.py                  # full run (steps 01–06)
  python 00_run_pipeline.py --download       # enable image download in step 04
  python 00_run_pipeline.py --review         # enable interactive review in step 06
  python 00_run_pipeline.py --from 3 --to 5  # run only steps 03–05
"""

import json
import os
import sys
import subprocess
from pathlib import Path

# ─────────────────────────── query parameters ─────────────────────────────────
# OPUS parameter names: https://opus.pds-rings.seti.org/opus/api
# Range pairs use the convention: param1 = min, param2 = max.

QUERY_PARAMS = {
    "target":                          "ENCELADUS",
    "COISSfilter":                     "CLEAR",
    "phase1":                          150,      # phase angle min (°)
    "phase2":                          180,      # phase angle max (°)
    "RINGGEOringradius1":              180_000,  # ring radius min (km)
    "RINGGEOringradius2":              480_000,  # ring radius max (km)
    "RINGGEOobserverringelevation1":   -5,       # ring-plane elevation min (°)
    "RINGGEOobserverringelevation2":   5,        # ring-plane elevation max (°)
    "RINGGEOsolarringopeningangle1":   -15,      # solar opening angle min (°)
    "RINGGEOsolarringopeningangle2":   15,       # solar opening angle max (°)
}
QUERY_LIMIT = 100_000

# ─────────────────────────── options ──────────────────────────────────────────

DO_DOWNLOAD = "--download" in sys.argv
DO_REVIEW   = "--review"   in sys.argv
FROM_STEP   = 1
TO_STEP     = 6

for i, arg in enumerate(sys.argv[1:]):
    if arg == "--from" and i + 1 < len(sys.argv[1:]):
        FROM_STEP = int(sys.argv[i + 2])
    elif arg == "--to" and i + 1 < len(sys.argv[1:]):
        TO_STEP = int(sys.argv[i + 2])

# ─────────────────────────── helpers ──────────────────────────────────────────

HERE = Path(__file__).parent

QUERY_ENV = {
    **os.environ,
    "QUERY_PARAMS_JSON": json.dumps({"params": QUERY_PARAMS, "limit": QUERY_LIMIT}),
}


def run_step(n: int, script: str, extra_args: list[str] = (), env: dict = os.environ):
    print(f"\n{'═' * 70}")
    print(f"  Step {n}: {script}")
    print(f"{'═' * 70}\n")
    result = subprocess.run([sys.executable, str(HERE / script), *extra_args], env=env)
    if result.returncode != 0:
        print(f"\n✗ Step {n} failed — fix the error above, then re-run with --from {n}.")
        sys.exit(result.returncode)
    print(f"\n✓ Step {n} done.")


# ─────────────────────────── summary ──────────────────────────────────────────

print("Pipeline configuration")
print(f"  Steps:    {FROM_STEP} → {TO_STEP}")
print(f"  Download: {'yes' if DO_DOWNLOAD else 'no (dry-run)'}")
print(f"  Review:   {'yes (interactive)' if DO_REVIEW else 'skip'}")
print(f"  Query ({len(QUERY_PARAMS)} parameters):")
for k, v in QUERY_PARAMS.items():
    print(f"    {k:<40s} = {v}")

# ─────────────────────────── run ──────────────────────────────────────────────

steps = [
    (1, "01_query_images.py",       [],                               QUERY_ENV),
    (2, "02_infer_sets.py",         [],                               os.environ),
    (3, "03_check_kernels.py",      [],                               os.environ),
    (4, "04_download_images.py",    ["--download"] if DO_DOWNLOAD else [], os.environ),
    (5, "05_clear_subtraction.py",  [],                               os.environ),
    (6, "06_flag_review.py",        [],                               os.environ),
]

for n, script, args, env in steps:
    if n < FROM_STEP or n > TO_STEP:
        continue
    if n == 6 and not DO_REVIEW:
        print(f"\n{'═' * 70}")
        print(f"  Step 6: 06_flag_review.py  [skipped — pass --review to enable]")
        print(f"{'═' * 70}")
        continue
    run_step(n, script, args, env)

print(f"\n{'═' * 70}")
print(f"  Pipeline complete  (steps {FROM_STEP}–{TO_STEP})")
print(f"{'═' * 70}\n")
