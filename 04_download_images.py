#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 04 — Download calibrated images from OPUS
-----------------------------------------------
Downloads the calibrated (_CALIB.IMG + _CALIB.LBL) files for every unique
opusid in pairs.parquet (both CLEAR and science images).

Dry-run by default: prints the image count and estimated download size.
Pass --download to fetch files. Already-present files are skipped.

Usage
-----
  python 04_download_images.py             # dry-run
  python 04_download_images.py --download  # full download

Outputs → data/raw_images/
  {opusid}_CALIB.IMG
  {opusid}_CALIB.LBL
  download_manifest.parquet
  download_metadata.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
PAIRS_FILE = DATA_DIR / "inferred_sets" / "pairs.parquet"
OUT_DIR    = DATA_DIR / "raw_images"
OPUS_BASE  = "https://opus.pds-rings.seti.org"
DRY_RUN    = "--download" not in sys.argv

# ─────────────────────────── unique opusids ───────────────────────────────────

pairs   = pd.read_parquet(PAIRS_FILE)
opusids = (
    pd.concat([pairs["clear_opusid"], pairs["science_opusid"]])
    .drop_duplicates().sort_values().reset_index(drop=True).tolist()
)
print(f"Unique images to download: {len(opusids)}")

# ─────────────────────────── resolve calibrated URLs ──────────────────────────

def resolve_urls(opusid: str) -> dict:
    r = requests.get(f"{OPUS_BASE}/opus/api/files/{opusid}.json", timeout=30)
    r.raise_for_status()
    calib   = r.json().get("data", {}).get(opusid, {}).get("coiss_calib", [])
    img_url = next((u for u in calib if u.upper().endswith("_CALIB.IMG")), None)
    lbl_url = next((u for u in calib if u.upper().endswith("_CALIB.LBL")), None)
    size    = 0
    if img_url:
        h    = requests.head(img_url, timeout=10)
        size = int(h.headers.get("Content-Length", 0)) or 4_000_000
    return {"img_url": img_url, "lbl_url": lbl_url, "size_bytes": size}


print("Resolving calibrated URLs from OPUS …")
records = []
for oid in tqdm(opusids, desc="URL lookup", unit="img"):
    info = {"opusid": oid, "status": "pending"}
    info.update(resolve_urls(oid))
    info["status"] = "resolved" if info["img_url"] else "no_calib"
    records.append(info)

manifest    = pd.DataFrame(records)
resolvable  = manifest[manifest["status"] == "resolved"]
total_bytes = resolvable["size_bytes"].sum()

print(f"\n  Resolvable:  {len(resolvable)}")
print(f"  No calib:    {len(manifest[manifest['status'] == 'no_calib'])}")
print(f"  Est. size:   {total_bytes / 1e9:.2f} GB")

if DRY_RUN:
    print("\n  Dry-run — no files downloaded.")
    print("  Re-run with --download to start.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(OUT_DIR / "download_manifest.parquet", index=False)
    sys.exit(0)

# ─────────────────────────── download ─────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with (
            open(dest, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                 desc=dest.name, leave=False) as bar,
        ):
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))


downloaded = 0
for _, row in tqdm(resolvable.iterrows(), total=len(resolvable), desc="Downloading", unit="img"):
    oid = row["opusid"]
    download_file(row["img_url"], OUT_DIR / f"{oid}_CALIB.IMG")
    if row["lbl_url"]:
        download_file(row["lbl_url"], OUT_DIR / f"{oid}_CALIB.LBL")
    manifest.loc[manifest["opusid"] == oid, "status"] = "done"
    downloaded += 1

# ─────────────────────────── save manifest ────────────────────────────────────

manifest.to_parquet(OUT_DIR / "download_manifest.parquet", index=False)
manifest[["opusid", "size_bytes", "status"]].to_csv(
    OUT_DIR / "download_manifest.csv", index=False)
(OUT_DIR / "download_metadata.json").write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "total_images":  len(opusids),
    "downloaded":    downloaded,
    "total_bytes":   int(resolvable["size_bytes"].sum()),
}, indent=2))

print(f"\n  ✓ {downloaded} downloaded  →  {OUT_DIR}")
