#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 05 — CLEAR subtraction
----------------------------
Subtracts each CLEAR image from its matched science image to produce
cleaned I/F residuals ready for band detection.

For each (CLEAR, science) pair in pairs.parquet:
  diff     = science − CLEAR
  cleaned  = 3-sigma outlier removal of diff
  vmin/vmax = campaign MAD scale (5 × MAD, estimated from a 50-pair sample)

Outputs per pair → data/cisscal_output/{science_opusid}/
  residual.npz     residual (float32) + valid mask (bool)
  grayscale.png    linear-stretch uint8 image
  redblue.png      diverging RdBu_r map, black background

Usage
-----
  python 05_clear_subtraction.py          # full run (all pairs)
  python 05_clear_subtraction.py --test   # single test pair
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import pandas as pd
import requests
import re
from tqdm import tqdm

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
PAIRS_FILE = DATA_DIR / "inferred_sets" / "pairs.parquet"
RAW_DIR    = DATA_DIR / "raw_images"
OUT_DIR    = DATA_DIR / "cisscal_output"
OPUS_BASE  = "https://opus.pds-rings.seti.org"

# ─────────────────────────── parameters ───────────────────────────────────────

SCALE_SAMPLE_N = 50
OUTLIER_SIGMA  = 3.0
MAD_FACTOR     = 5.0
TEST_OPUSID    = "co-iss-n1616348695"

# ─────────────────────────── I/O helpers ──────────────────────────────────────

def read_calib_img(opusid: str) -> np.ndarray:
    """
    Read a VICAR _CALIB.IMG from RAW_DIR into a float32 (1024, 1024) array.

    OPUS calibrated files use VICAR format: the first LBLSIZE bytes are an
    ASCII header, followed by NLB binary-label records, then the pixel data
    (little-endian float32, PC_RIEEE). The correct image offset is
    LBLSIZE + NLB × RECSIZE (parsed from the header).
    """
    img_path = RAW_DIR / f"{opusid}_CALIB.IMG"
    with open(img_path, "rb") as fh:
        raw_header = fh.read(512).decode("ascii", errors="replace")

    def _vget(key: str, default: int) -> int:
        m = re.search(rf"\b{key}=(\d+)", raw_header)
        return int(m.group(1)) if m else default

    lblsize = _vget("LBLSIZE", 4096)
    recsize = _vget("RECSIZE", 4096)
    nlb     = _vget("NLB",     0)
    nl      = _vget("NL",      1024)
    ns      = _vget("NS",      1024)

    offset = lblsize + nlb * recsize
    arr    = np.fromfile(str(img_path), dtype="<f4", count=nl * ns, offset=offset)
    return arr.reshape(nl, ns).astype(np.float32)


def download_for_test(opusid: str) -> None:
    """Download _CALIB.IMG + _CALIB.LBL from OPUS into RAW_DIR if missing."""
    img_dest = RAW_DIR / f"{opusid}_CALIB.IMG"
    lbl_dest = RAW_DIR / f"{opusid}_CALIB.LBL"
    if img_dest.exists() and lbl_dest.exists():
        return
    r     = requests.get(f"{OPUS_BASE}/opus/api/files/{opusid}.json", timeout=30)
    r.raise_for_status()
    calib   = r.json().get("data", {}).get(opusid, {}).get("coiss_calib", [])
    img_url = next((u for u in calib if u.upper().endswith("_CALIB.IMG")), None)
    lbl_url = next((u for u in calib if u.upper().endswith("_CALIB.LBL")), None)
    if not img_url:
        raise RuntimeError(f"No coiss_calib product found for {opusid}")
    for url, dest in [(img_url, img_dest), (lbl_url, lbl_dest)]:
        if dest and url and not dest.exists():
            print(f"  Downloading {dest.name} …")
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                dest.write_bytes(resp.content)

# ─────────────────────────── processing helpers ────────────────────────────────

def clean_residual(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median      = np.median(data)
    std         = np.std(data)
    outlier     = np.abs(data - median) > OUTLIER_SIGMA * std
    valid_mask  = ~outlier & np.isfinite(data)
    cleaned     = data.copy()
    cleaned[outlier] = median
    return cleaned, valid_mask


def campaign_scale(residuals: list) -> tuple[float, float, float]:
    all_vals = np.concatenate([r.flatten() for r in residuals])
    all_vals = all_vals[np.isfinite(all_vals)]
    median   = np.median(all_vals)
    mad      = np.median(np.abs(all_vals - median))
    vmax     = MAD_FACTOR * mad
    return -vmax, vmax, mad


def save_grayscale(residual: np.ndarray, mask: np.ndarray,
                   path: Path, vmin: float, vmax: float) -> None:
    clipped    = np.clip(residual, vmin, vmax)
    normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    normalized[~mask] = 128
    Image.fromarray(normalized, mode="L").save(path)


def save_redblue(residual: np.ndarray, mask: np.ndarray,
                 path: Path, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    norm    = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    ax.imshow(residual, cmap="RdBu_r", norm=norm, origin="upper", aspect="auto")
    overlay = np.zeros((*residual.shape, 4))
    overlay[~mask, 3] = 1.0
    ax.imshow(overlay, origin="upper", aspect="auto")
    ax.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)


def compute_diff(row: pd.Series) -> np.ndarray:
    clear_arr   = read_calib_img(row["clear_opusid"])
    science_arr = read_calib_img(row["science_opusid"])
    return science_arr - clear_arr


def process_pair(row: pd.Series, vmin: float, vmax: float) -> str:
    out = OUT_DIR / row["science_opusid"]
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "residual.npz"
    if npz.exists():
        return "skipped"
    diff           = compute_diff(row)
    cleaned, vmask = clean_residual(diff)
    np.savez_compressed(npz, residual=cleaned.astype(np.float32), mask=vmask)
    save_grayscale(cleaned, vmask, out / "grayscale.png", vmin, vmax)
    save_redblue  (cleaned, vmask, out / "redblue.png",   vmin, vmax)
    return "ok"

# ─────────────────────────── main ─────────────────────────────────────────────

pairs   = pd.read_parquet(PAIRS_FILE)
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if "--test" in sys.argv:
    row = pairs[pairs["science_opusid"] == TEST_OPUSID].iloc[0]
    print(f"Test pair:\n  CLEAR:   {row['clear_opusid']}\n  Science: {row['science_opusid']}")
    download_for_test(row["clear_opusid"])
    download_for_test(row["science_opusid"])
    diff           = compute_diff(row)
    cleaned, vmask = clean_residual(diff)
    vmin, vmax, _  = campaign_scale([cleaned])
    out = OUT_DIR / TEST_OPUSID
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "residual.npz", residual=cleaned.astype(np.float32), mask=vmask)
    save_grayscale(cleaned, vmask, out / "grayscale.png", vmin, vmax)
    save_redblue  (cleaned, vmask, out / "redblue.png",   vmin, vmax)
    print(f"\n  OK → {out}")
    sys.exit(0)

# Campaign scale — load from file or estimate from sample
scale_path = OUT_DIR / "campaign_scale.json"
if scale_path.exists():
    sc   = json.loads(scale_path.read_text())
    vmin, vmax = sc["vmin"], sc["vmax"]
    print(f"Loaded campaign scale: vmin={vmin:.2e}  vmax={vmax:.2e}")
else:
    n_sample  = min(SCALE_SAMPLE_N, len(pairs))
    sample    = pairs.sample(n_sample, random_state=42)
    print(f"Estimating campaign scale from {n_sample} random pairs …")
    sample_residuals = []
    for _, srow in tqdm(sample.iterrows(), total=n_sample, desc="Scale sample"):
        diff, _ = clean_residual(compute_diff(srow))
        sample_residuals.append(diff)
    vmin, vmax, mad = campaign_scale(sample_residuals)
    scale_path.write_text(json.dumps({
        "vmin": float(vmin), "vmax": float(vmax),
        "mad":  float(mad),  "mad_factor": MAD_FACTOR,
        "sample_n": len(sample_residuals),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"  vmin={vmin:.2e}  vmax={vmax:.2e}  MAD={mad:.2e}")

ok, skipped = 0, 0
for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="CLEAR subtraction", unit="pair"):
    s = process_pair(row, vmin, vmax)
    if s == "skipped":
        skipped += 1
    else:
        ok += 1

(OUT_DIR / "subtraction_metadata.json").write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "total_pairs":   len(pairs),
    "ok":            ok,
    "skipped":       skipped,
    "vmin":          float(vmin),
    "vmax":          float(vmax),
}, indent=2))
print(f"\n  ✓ {ok} done  |  {skipped} skipped  →  {OUT_DIR}")
