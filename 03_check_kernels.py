#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 03 — Check and download SPICE kernels
------------------------------------------
Checks that SPICE CK (attitude) kernels cover every observation time in
pairs.parquet. Downloads any missing reconstruction kernels from NAIF and
writes an updated metakernel.

Expected kernel archive layout
-------------------------------
  data/kernels/
    data/
      ck/     ← CK files (*.bc)
      spk/    ← SPK files (*.bsp)
      lsk/    ← LSK files (*.tls)
      pck/    ← PCK files (*.tpc)
      sclk/   ← SCLK files (*.tsc)
      fk/     ← FK files (*.tf)
      ik/     ← IK files (*.ti)
      iak/    ← IAK files (*.ti)

Populate data/kernels/data/ with the CASSINI kernel archive from NAIF:
  https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/

Outputs → data/kernels/
  cassini_iss.tm          metakernel (load this with spiceypy.furnsh)
  kernels_metadata.json   coverage summary
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR       = Path(__file__).parent / "data"
KERNEL_ROOT    = DATA_DIR / "kernels"
KERNEL_DIR     = KERNEL_ROOT / "data"
PAIRS_FILE     = DATA_DIR / "inferred_sets" / "pairs.parquet"
NAIF_CK_URL    = "https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ck/"

# ─────────────────────────── 1. unique observation times ──────────────────────

pairs = pd.read_parquet(PAIRS_FILE)
times = pd.concat([
    pd.to_datetime(pairs["clear_time1"],   utc=True),
    pd.to_datetime(pairs["science_time1"], utc=True),
]).drop_duplicates().sort_values().reset_index(drop=True)
print(f"Input: {len(times)} unique obs times  ({times.min().date()} → {times.max().date()})")


def _ydoy(t) -> int:
    return (t.year % 100) * 1000 + t.timetuple().tm_yday


# ─────────────────────────── 2. check local CK coverage ───────────────────────

_CK_RE = re.compile(r'^(\d{5})_(\d{5})(ra|pa_gapfill_v\d+|py_as_flown|pc_psiv2)\.bc$')


def ck_range(path: Path):
    m = _CK_RE.match(path.name)
    return (int(m[1]), int(m[2])) if m else None


local_cks = [(p, ck_range(p)) for p in sorted((KERNEL_DIR / "ck").glob("*.bc"))]
local_cks = [(p, r) for p, r in local_cks if r is not None]


def is_covered(t) -> bool:
    yd = _ydoy(t)
    return any(lo <= yd <= hi for _, (lo, hi) in local_cks)


uncovered = [t for t in tqdm(times, desc="Checking CK coverage", unit="obs") if not is_covered(t)]
print(f"  Covered: {len(times) - len(uncovered)}  |  Uncovered: {len(uncovered)}")

# ─────────────────────────── 3. download missing CK files ─────────────────────

to_download = set()
if uncovered:
    print("Fetching NAIF CK index …")
    r = requests.get(NAIF_CK_URL, timeout=30)
    r.raise_for_status()
    available = re.findall(r'href="(\d{5}_\d{5}ra\.bc)"', r.text)
    available_ranges = [(name, int(name[:5]), int(name[6:11])) for name in available]

    needed_ydoys = {_ydoy(t) for t in uncovered}
    for yd in tqdm(sorted(needed_ydoys), desc="Matching NAIF kernels", unit="ydoy"):
        for name, lo, hi in available_ranges:
            if lo <= yd <= hi:
                if not (KERNEL_DIR / "ck" / name).exists():
                    to_download.add(name)
                break

    print(f"  Downloading {len(to_download)} CK file(s) …")
    for name in tqdm(sorted(to_download), desc="Downloading", unit="file"):
        dest = KERNEL_DIR / "ck" / name
        with requests.get(NAIF_CK_URL + name, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            dest.write_bytes(resp.content)

# ─────────────────────────── 4. write metakernel ──────────────────────────────

kernel_groups = {
    "lsk":  ("*.tls", "lsk"),
    "pck":  ("*.tpc", "pck"),
    "sclk": ("*.tsc", "sclk"),
    "fk":   ("*.tf",  "fk"),
    "ik":   ("*.ti",  "ik"),
    "iak":  ("*.ti",  "iak"),
    "spk":  ("*.bsp", "spk"),
    "ck":   ("*.bc",  "ck"),
}
available = {}
for key, (pattern, subdir) in kernel_groups.items():
    available[key] = sorted(f.name for f in (KERNEL_DIR / subdir).glob(pattern))

mk_path = KERNEL_ROOT / "cassini_iss.tm"
lines = [
    "KPL/MK\n\n",
    "\\begindata\n\n",
    f"PATH_VALUES     = ( '{KERNEL_DIR}' )\n",
    "PATH_SYMBOLS    = ( 'KERNELS' )\n\n",
    "KERNELS_TO_LOAD = (\n",
]
for key in ["lsk", "pck", "sclk", "fk", "ik", "iak", "spk", "ck"]:
    for name in available[key]:
        lines.append(f"   '$KERNELS/{key}/{name}'\n")
lines += [")\n\n\\begintext\n"]
mk_path.write_text("".join(lines))

# ─────────────────────────── 5. report ────────────────────────────────────────

still_missing = sum(1 for t in times if not is_covered(t))
(KERNEL_ROOT / "kernels_metadata.json").write_text(json.dumps({
    "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
    "obs_times_total":     len(times),
    "ck_files_available":  len(available["ck"]),
    "spk_files_available": len(available["spk"]),
    "ck_downloaded":       len(to_download),
    "obs_still_uncovered": still_missing,
    "metakernel":          str(mk_path),
}, indent=2))

print(f"\n  {len(available['ck'])} CK  |  {len(available['spk'])} SPK  "
      f"|  {len(to_download)} downloaded  |  {still_missing} still uncovered")
print(f"  → {mk_path}")
if still_missing:
    print(f"\n  ⚠  {still_missing} observation times have no matching 'ra' CK on NAIF.")
    print("     Check for 'predict' or gapfill kernels manually if needed.")
