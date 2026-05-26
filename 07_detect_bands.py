#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 07 — Luminous band detection
-----------------------------------
Runs the FFT angular spectrum + random-phase null baseline on every
residual.npz file produced by step 05.

Algorithm
---------
  1. Load residual.npz and preprocess: high-pass filter (σ=80 px),
     Enceladus disc mask (SPICE), row/col median detrend, arcsinh stretch.
  2. Compute the standard FFT angular spectrum E(θ).
  3. Compute the zero-padded FFT angular spectrum E_zp(θ).
  4. Build a random-phase null ensemble (N_NULL surrogates) → E_null_zp.
  5. Smooth E_diff_zp (w=9) then again (w=64); detect FWHM peak.
  6. Save bundle_A.png (8-panel diagnostic), spectra.npz, metrics.json.

SPICE kernels
-------------
  Set up the kernel archive first (step 03). The metakernel path is
  data/kernels/cassini_iss.tm. Without SPICE the disc mask falls back
  to a centred circle; detections still run but may be noisier.

Usage
-----
  python 07_detect_bands.py                    # all images in cisscal_output
  python 07_detect_bands.py --test             # single test image
  python 07_detect_bands.py --opusid co-iss-n1711553432
  python 07_detect_bands.py --workers 4        # parallel (default: 4)
  python 07_detect_bands.py --n-null 100       # null surrogates (default: 100)
  python 07_detect_bands.py --skip-done        # skip completed images
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from tqdm import tqdm

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent / "data"
CISSCAL_DIR  = DATA_DIR / "cisscal_output"
PIPELINE_DIR = DATA_DIR / "pipeline_output"
PAIRS_FILE   = DATA_DIR / "inferred_sets" / "pairs.parquet"
METAKERNEL   = DATA_DIR / "kernels" / "cassini_iss.tm"

TEST_OPUSID  = "co-iss-n1616348949"

# ─────────────────────────── SPICE constants ──────────────────────────────────

CASSINI_ID    = -82
NAC_ID        = -82360
ENCELADUS_ID  = "ENCELADUS"
ENC_RADIUS_KM = 252.1

# ─────────────────────────── processing parameters ────────────────────────────

HP_SIGMA           = 80
ARCSINH_MULT       = 0.05
MASK_RADIUS_FACTOR = 1.5
MASK_TAPER_PX      = 60

DC_RADIUS          = 8.0
EXCLUDE_AXIS_DEG   = 5.0
WEDGE_RADIUS       = 50.0
N_THETA_BINS       = 720

ZP_PAD_FACTOR_TARGET = 16
ZP_DC_RADIUS         = 1.0
ZP_R_MAX             = 6.0
ZP_PEAK_MIN_SEP_DEG  = 2.0
ZP_PROM_SCALE        = 0.25
MAX_ZP_BYTES         = 800_000_000

SMOOTH_WIN_NULL  = 9
SMOOTH_WIN_FINAL = 64
NULL_SEED        = 42

COL_ZP   = "#8e24aa"
COL_MAIN = "#00e5ff"

# ─────────────────────────── helpers ──────────────────────────────────────────

def opusid_to_image_id(opusid: str) -> str:
    stem = opusid.split("-")[-1]
    return stem[0].upper() + stem[1:]


def load_spice() -> bool:
    """Load SPICE kernels from the metakernel. Returns True on success."""
    import re
    import spiceypy as spice
    if not METAKERNEL.exists():
        return False
    text    = METAKERNEL.read_text()
    m       = re.search(r"\\begindata(.*?)(?:\\begintext|$)", text, re.DOTALL)
    if not m:
        return False
    kernels = re.findall(r"'\$KERNELS/([^']+)'", m.group(1))
    kernel_data_dir = METAKERNEL.parent / "data"
    spice.kclear()
    for rel in sorted(kernels):
        full = kernel_data_dir / rel
        if full.exists():
            spice.furnsh(str(full))
    return spice.ktotal("ALL") > 0


def enceladus_disc_px(image_id: str) -> tuple[float, float, float]:
    import spiceypy as spice
    et      = spice.scs2e(CASSINI_ID, "1/" + image_id[1:])
    pos, _  = spice.spkpos(ENCELADUS_ID, et, "CASSINI_ISS_NAC", "LT+S", "CASSINI")
    _, _, _, _, bounds = spice.getfov(NAC_ID, 10)
    plate   = math.atan(abs(bounds[0][0]) / abs(bounds[0][2])) / 512.0
    col     = 512.0 - math.atan2(pos[0], pos[2]) / plate
    row     = 512.0 - math.atan2(pos[1], pos[2]) / plate
    disc_r  = (ENC_RADIUS_KM / float(np.linalg.norm(pos))) / plate
    return float(col), float(row), float(disc_r)


def make_disc_mask(H: int, W: int, col: float, row: float, disc_r: float) -> np.ndarray:
    yy, xx  = np.mgrid[0:H, 0:W]
    dist    = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)
    r_inner = disc_r * MASK_RADIUS_FACTOR
    t       = np.clip((dist - r_inner) / MASK_TAPER_PX, 0.0, 1.0)
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)


def choose_pad_factor(h: int, w: int, target: int, max_bytes: int) -> int:
    pad = target
    while pad > 1:
        if h * pad * w * pad * 8 <= max_bytes:
            return pad
        pad //= 2
    return 1


def angular_spectrum(mag, R, theta, *, dc_radius, exclude_axis_deg, wedge_radius, n_bins):
    power   = mag.astype(np.float64) ** 2
    d_to_0  = np.minimum(theta, 180.0 - theta)
    d_to_90 = np.abs(theta - 90.0)
    mask    = (
        (R >= dc_radius) & (R <= wedge_radius)
        & (d_to_0 > exclude_axis_deg) & (d_to_90 > exclude_axis_deg)
    )
    th_m = theta[mask];  pw_m = power[mask];  r_m = R[mask].astype(np.float64)
    w    = np.where(r_m > 0, 1.0 / r_m, 0.0)
    bins = np.linspace(0.0, 180.0, n_bins + 1)
    idx  = np.clip(np.digitize(th_m, bins) - 1, 0, n_bins - 1)
    E    = np.bincount(idx, weights=pw_m * w, minlength=n_bins).astype(np.float64)
    return 0.5 * (bins[:-1] + bins[1:]), E


def detection_metrics(E: np.ndarray) -> tuple[float, float]:
    med  = float(np.median(E))
    peak = float(E.max())
    mad  = float(np.median(np.abs(E - med)))
    pmr  = peak / med if med > 0 else 0.0
    snr  = (peak - med) / mad if mad > 0 else 0.0
    return pmr, snr


def zp_angular_from_image(x, *, pad_factor):
    h0, w0  = x.shape
    ph, pw  = h0 * pad_factor, w0 * pad_factor
    padded  = np.zeros((ph, pw), dtype=np.float32)
    padded[:h0, :w0] = x - float(np.mean(x))
    Fz      = sfft.fftshift(sfft.fft2(padded))
    cy_zp, cx_zp  = (ph - 1) / 2.0, (pw - 1) / 2.0
    crop_r_px     = int(round(ZP_R_MAX * pad_factor))
    dc_r_px       = float(ZP_DC_RADIUS * pad_factor)
    icy, icx      = int(round(cy_zp)), int(round(cx_zp))
    Fz_crop       = Fz[max(0, icy - crop_r_px):icy + crop_r_px + 1,
                       max(0, icx - crop_r_px):icx + crop_r_px + 1]
    zp_crop       = (np.abs(Fz_crop).astype(np.float32) ** 2).astype(np.float64)
    del Fz, Fz_crop, padded
    ch, cw   = zp_crop.shape
    yyc, xxc = np.indices((ch, cw), dtype=np.float32)
    ccy, ccx = (ch - 1) / 2.0, (cw - 1) / 2.0
    rrc_px   = np.sqrt((xxc - ccx) ** 2 + (yyc - ccy) ** 2)
    zp_crop[rrc_px > crop_r_px] = np.nan
    zp_crop[rrc_px < dc_r_px]   = np.nan
    R_crop  = rrc_px / float(pad_factor)
    TH_crop = (np.rad2deg(np.arctan2(yyc - ccy, xxc - ccx)) + 180.0) % 180.0
    d_to_0  = np.minimum(TH_crop, 180.0 - TH_crop)
    d_to_90 = np.abs(TH_crop - 90.0)
    valid   = (
        np.isfinite(zp_crop)
        & (R_crop >= ZP_DC_RADIUS) & (R_crop <= ZP_R_MAX)
        & (d_to_0 > EXCLUDE_AXIS_DEG) & (d_to_90 > EXCLUDE_AXIS_DEG)
    )
    th_m = TH_crop[valid];  pw_m = zp_crop[valid].astype(np.float64)
    r_m  = R_crop[valid].astype(np.float64)
    w_m  = np.where(r_m > 0, 1.0 / r_m, 0.0)
    bins = np.linspace(0.0, 180.0, N_THETA_BINS + 1)
    idx  = np.clip(np.digitize(th_m, bins) - 1, 0, N_THETA_BINS - 1)
    E    = np.bincount(idx, weights=pw_m * w_m, minlength=N_THETA_BINS).astype(np.float64)
    return 0.5 * (bins[:-1] + bins[1:]), E


def smooth_1d(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.asarray(y, dtype=np.float64)
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(np.asarray(y, dtype=np.float64), k, mode="same")


def fwhm_peak(theta_vals, spectrum):
    peak_idx   = int(np.argmax(spectrum))
    peak_value = float(spectrum[peak_idx])
    half_max   = peak_value / 2.0
    left_theta = float(theta_vals[0])
    for i in range(peak_idx - 1, -1, -1):
        if spectrum[i] < half_max:
            t0, t1 = theta_vals[i], theta_vals[i + 1]
            v0, v1 = spectrum[i], spectrum[i + 1]
            left_theta = float(t0 + (half_max - v0) * (t1 - t0) / (v1 - v0))
            break
    right_theta = float(theta_vals[-1])
    for i in range(peak_idx + 1, len(spectrum)):
        if spectrum[i] < half_max:
            t0, t1 = theta_vals[i - 1], theta_vals[i]
            v0, v1 = spectrum[i - 1], spectrum[i]
            right_theta = float(t0 + (half_max - v0) * (t1 - t0) / (v1 - v0))
            break
    peak_theta = (left_theta + right_theta) / 2.0
    fwhm_width = right_theta - left_theta
    return float(peak_theta), float(fwhm_width), peak_value, left_theta, right_theta

# ─────────────────────────── bundle A figure ──────────────────────────────────

def save_bundle_A(out_path, *, opusid, raw_work, img_proc, log_power_full,
                  theta_deg, E_theta, peak_theta, pmr, snr, png_img,
                  peak_ref_64, fwhm_ref_64, l_ref_64, r_ref_64,
                  theta_zp, E_diff_zp_ref, E_diff_zp_ref_64, peak_ref,
                  zp_db, zp_peak_theta, E_theta_zp, topk, pmr_zp, snr_zp,
                  ZP_PAD_FACTOR, have_spice, col, row, disc_r, supp,
                  img_h, img_w):
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))

    ax = axes[0, 0]
    ax.imshow(raw_work, cmap="gray")
    if have_spice and disc_r > 0:
        tt     = np.linspace(0.0, 2.0 * np.pi, 360)
        mask_r = disc_r * MASK_RADIUS_FACTOR
        ax.plot(col + disc_r * np.cos(tt), row + disc_r * np.sin(tt),
                color="lime", lw=1.3, label=f"disc {disc_r:.0f}px")
        ax.plot(col + mask_r * np.cos(tt), row + mask_r * np.sin(tt),
                color="yellow", lw=1.3, label=f"mask {mask_r:.0f}px")
        ax.plot(col, row, "+", color="lime", ms=12, mew=1.5)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Top-1: Raw residual + Enceladus disc/mask")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(img_proc, cmap="gray")
    ax.set_title(f"Top-2: Preprocessed image (suppressed={supp:.1f}%)")
    ax.axis("off")

    ax = axes[0, 2]
    im3 = ax.imshow(log_power_full, cmap="magma")
    ax.set_title("Top-3: Full log|FFT| map")
    ax.axis("off")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="log10 power")

    ax = axes[0, 3]
    ax.plot(theta_deg, E_theta, lw=1.5, color="#00bcd4")
    ax.axvline(peak_theta, ls="--", lw=1.3, color="r", label=f"θ={peak_theta:.2f}°")
    ax.set_xlim(0, 180);  ax.set_xlabel("θ (deg)");  ax.set_ylabel("E(θ)")
    ax.set_title(f"Top-4: Angular spectrum  (PMR={pmr:.2f}, SNR={snr:.2f})")
    ax.grid(True, alpha=0.2);  ax.legend(loc="best")

    theta_band_64 = float((peak_ref_64 + 90.0) % 180.0)
    ax = axes[1, 0]
    ax.imshow(png_img, cmap="gray" if np.ndim(png_img) == 2 else None, origin="upper")
    hp2, wp2 = png_img.shape[:2]
    seg = 0.055 * min(hp2, wp2)
    if np.isfinite(peak_ref_64):
        a  = math.radians(theta_band_64)
        dx, dy = math.cos(a), math.sin(a)
        for iy in range(7):
            for ix in range(7):
                cx = (ix + 0.5) * wp2 / 7;  cy = (iy + 0.5) * hp2 / 7
                ax.plot([cx - seg*dx, cx + seg*dx], [cy - seg*dy, cy + seg*dy],
                        color=COL_MAIN, lw=1.0, alpha=0.6, solid_capstyle="round")
    ax.set_title("Bottom-1: PNG + w=64 7×7 band direction mosaic")
    ax.axis("off")

    ax = axes[1, 1]
    im_zp = ax.imshow(zp_db, cmap="inferno", vmin=-40, vmax=0,
                      extent=[-ZP_R_MAX, ZP_R_MAX, ZP_R_MAX, -ZP_R_MAX])
    if np.isfinite(zp_peak_theta):
        a = math.radians(zp_peak_theta);  L = 0.92 * ZP_R_MAX
        ax.plot([-L*math.cos(a), L*math.cos(a)], [-L*math.sin(a), L*math.sin(a)],
                "-", color=COL_ZP, lw=2.0, label=f"θ={zp_peak_theta:.2f}°")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Bottom-2: ZP log|FFT| (pad={ZP_PAD_FACTOR})")
    ax.set_xlabel("kx (cyc/img)");  ax.set_ylabel("ky (cyc/img)")
    fig.colorbar(im_zp, ax=ax, fraction=0.046, pad=0.04, label="dB")

    ax = axes[1, 2]
    ax.plot(theta_zp, E_theta_zp, lw=1.3, color=COL_ZP, label="E_zp(θ)")
    if topk.size > 0:
        ax.scatter(theta_zp[topk], E_theta_zp[topk], s=34, color="#ffca28",
                   edgecolor="black", lw=0.4, zorder=4, label="peaks")
    ax.set_xlim(0, 180);  ax.set_xlabel("θ (deg)");  ax.set_ylabel("E_zp(θ)")
    ax.set_title(f"Bottom-3: ZP spectrum  (PMR={pmr_zp:.2f}, SNR={snr_zp:.2f})")
    ax.grid(True, alpha=0.2);  ax.legend(loc="best", fontsize=8)

    ax = axes[1, 3]
    ax.plot(theta_zp, E_diff_zp_ref,    color="#7e57c2", lw=1.8,
            label=f"ΔE_zp (w={SMOOTH_WIN_NULL})")
    ax.plot(theta_zp, E_diff_zp_ref_64, color=COL_MAIN,  lw=2.0,
            label=f"w={SMOOTH_WIN_FINAL}")
    ax.axvline(peak_ref,    color="#7e57c2", ls="--", lw=1.2, label=f"ref={peak_ref:.2f}°")
    ax.axvline(peak_ref_64, color=COL_MAIN,  ls="--", lw=1.2, label=f"w=64={peak_ref_64:.2f}°")
    ax.axvline(l_ref_64, color="green", ls="-", lw=0.9, alpha=0.5)
    ax.axvline(r_ref_64, color="green", ls="-", lw=0.9, alpha=0.5,
               label=f"FWHM={fwhm_ref_64:.1f}°")
    ax.set_xlim(0, 180);  ax.set_xlabel("θ (deg)");  ax.set_ylabel("ΔE_zp(θ)")
    ax.set_title(f"Bottom-4: ΔE_zp  w={SMOOTH_WIN_NULL} vs w={SMOOTH_WIN_FINAL}")
    ax.grid(True, alpha=0.2);  ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Band detection — {opusid}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────── per-image worker ─────────────────────────────────

def process_one(args: tuple) -> dict:
    opusid, science_filter, science_time, n_null = args

    out_dir = PIPELINE_DIR / opusid
    out_dir.mkdir(parents=True, exist_ok=True)

    image_id = opusid_to_image_id(opusid)
    npz_path = CISSCAL_DIR / opusid / "residual.npz"
    png_path = CISSCAL_DIR / opusid / "grayscale.png"

    d       = np.load(npz_path)
    raw0    = d["residual"].astype(np.float32)
    vmask   = d["mask"].astype(bool) if "mask" in d else np.ones(raw0.shape, bool)

    finite_vals = raw0[np.isfinite(raw0) & vmask]
    med0        = float(np.median(finite_vals))
    lo_f = float(np.percentile(finite_vals, 0.1))
    hi_f = float(np.percentile(finite_vals, 99.9))
    raw_work = np.where(vmask, raw0, med0).astype(np.float32)
    raw_work = np.nan_to_num(raw_work, nan=med0, posinf=hi_f, neginf=lo_f)
    H, W     = raw_work.shape

    # SPICE disc mask
    have_spice = False
    col = W / 2.0;  row = H / 2.0;  disc_r = 0.0
    import spiceypy as spice
    if spice.ktotal("ALL") > 0:
        col, row, disc_r = enceladus_disc_px(image_id)
        have_spice = np.isfinite(disc_r) and disc_r > 0

    enc_mask = make_disc_mask(H, W, col, row, disc_r) if have_spice else np.ones((H, W), np.float32)

    # Preprocess
    hp   = (raw_work.astype(np.float64) - gaussian_filter(raw_work.astype(np.float64), HP_SIGMA)).astype(np.float32)
    work = hp.copy()
    work[enc_mask < 0.5] = np.nan
    with np.errstate(all="ignore"):
        rm = np.nanmedian(work, axis=1, keepdims=True)
        work -= np.where(np.isfinite(rm), rm, 0.0)
        cm = np.nanmedian(work, axis=0, keepdims=True)
        work -= np.where(np.isfinite(cm), cm, 0.0)
    work  = np.nan_to_num(work, nan=0.0).astype(np.float32) * enc_mask
    supp  = 100.0 * float((enc_mask < 0.5).mean())
    scale = max(float(np.std(work)) * ARCSINH_MULT, 1e-12)
    proc  = np.arcsinh(work / scale).astype(np.float32)
    lo2, hi2 = np.percentile(proc[np.isfinite(proc)], [0.1, 99.9])
    img_proc  = np.clip(proc, lo2, hi2)
    img_proc  = (img_proc - img_proc.min()) / (img_proc.max() - img_proc.min() + 1e-12)

    # Main FFT
    x_main  = img_proc - float(img_proc.mean())
    mag     = np.abs(np.fft.fftshift(np.fft.fft2(x_main))).astype(np.float32)
    yy, xx  = np.indices(mag.shape, dtype=np.float32)
    cy, cx  = (mag.shape[0] - 1) / 2.0, (mag.shape[1] - 1) / 2.0
    R       = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    theta   = (np.rad2deg(np.arctan2(yy - cy, xx - cx)) + 180.0) % 180.0
    theta_deg, E_theta = angular_spectrum(
        mag, R, theta,
        dc_radius=DC_RADIUS, exclude_axis_deg=EXCLUDE_AXIS_DEG,
        wedge_radius=WEDGE_RADIUS, n_bins=N_THETA_BINS,
    )
    pmr, snr   = detection_metrics(E_theta)
    peak_theta = float(theta_deg[int(np.argmax(E_theta))])
    log_power_full = np.log10((mag.astype(np.float64) ** 2) + 1e-12)
    del mag, R, theta

    # ZP FFT
    ZP_PAD  = choose_pad_factor(H, W, ZP_PAD_FACTOR_TARGET, MAX_ZP_BYTES)
    ph, pw  = H * ZP_PAD, W * ZP_PAD
    padded  = np.zeros((ph, pw), dtype=np.float32)
    padded[:H, :W] = img_proc - float(img_proc.mean())
    Fz      = sfft.fftshift(sfft.fft2(padded))
    cy_zp, cx_zp = (ph - 1) / 2.0, (pw - 1) / 2.0
    crop_r_px    = int(round(ZP_R_MAX * ZP_PAD))
    dc_r_px      = float(ZP_DC_RADIUS * ZP_PAD)
    icy, icx     = int(round(cy_zp)), int(round(cx_zp))
    Fz_crop      = Fz[max(0, icy - crop_r_px):icy + crop_r_px + 1,
                      max(0, icx - crop_r_px):icx + crop_r_px + 1]
    zp_crop = (np.abs(Fz_crop).astype(np.float32) ** 2).astype(np.float64)
    del Fz, Fz_crop, padded

    ch, cw   = zp_crop.shape
    yyc, xxc = np.indices((ch, cw), dtype=np.float32)
    ccy, ccx = (ch - 1) / 2.0, (cw - 1) / 2.0
    rrc_px   = np.sqrt((xxc - ccx) ** 2 + (yyc - ccy) ** 2)
    zp_crop[rrc_px > crop_r_px] = np.nan
    zp_crop[rrc_px < dc_r_px]   = np.nan
    zp_db = 10.0 * np.log10(zp_crop + 1e-12)
    if np.isfinite(zp_db).any():
        zp_db -= float(np.nanmax(zp_db))

    R_crop  = rrc_px / float(ZP_PAD)
    TH_crop = (np.rad2deg(np.arctan2(yyc - ccy, xxc - ccx)) + 180.0) % 180.0
    d_to_0  = np.minimum(TH_crop, 180.0 - TH_crop)
    d_to_90 = np.abs(TH_crop - 90.0)
    valid_zp = (
        np.isfinite(zp_crop)
        & (R_crop >= ZP_DC_RADIUS) & (R_crop <= ZP_R_MAX)
        & (d_to_0 > EXCLUDE_AXIS_DEG) & (d_to_90 > EXCLUDE_AXIS_DEG)
    )
    th_m = TH_crop[valid_zp];  pw_m = zp_crop[valid_zp].astype(np.float64)
    r_m  = R_crop[valid_zp].astype(np.float64)
    w_m  = np.where(r_m > 0, 1.0 / r_m, 0.0)
    bins = np.linspace(0.0, 180.0, N_THETA_BINS + 1)
    idx  = np.clip(np.digitize(th_m, bins) - 1, 0, N_THETA_BINS - 1)
    E_theta_zp = np.bincount(idx, weights=pw_m * w_m, minlength=N_THETA_BINS).astype(np.float64)
    theta_zp   = 0.5 * (bins[:-1] + bins[1:])

    pmr_zp, snr_zp = detection_metrics(E_theta_zp)
    min_sep_bins   = max(1, int(N_THETA_BINS * ZP_PEAK_MIN_SEP_DEG / 180.0))
    prom_thr       = max(float(np.nanstd(E_theta_zp)) * ZP_PROM_SCALE, 1e-12)
    pk_zp, _       = find_peaks(E_theta_zp, prominence=prom_thr, distance=min_sep_bins)
    if pk_zp.size == 0:
        pk_zp, _ = find_peaks(E_theta_zp, distance=min_sep_bins)
    topk           = pk_zp[np.argsort(E_theta_zp[pk_zp])[::-1][:5]] if pk_zp.size > 0 else np.array([], int)
    zp_peak_theta  = float(theta_zp[int(topk[0])]) if topk.size > 0 else float("nan")

    # Null ensemble
    rng = np.random.default_rng(NULL_SEED)
    x0  = img_proc.astype(np.float32) - float(img_proc.mean())
    F   = np.fft.fft2(x0)
    amp = np.abs(F).astype(np.float64)

    E_null_all = np.empty((n_null, N_THETA_BINS), np.float64)
    for i in range(n_null):
        phi    = rng.uniform(0.0, 2.0 * np.pi, size=F.shape)
        x_rand = np.fft.ifft2(amp * np.exp(1j * phi)).real.astype(np.float32)
        x_rand -= float(np.mean(x_rand))
        _, E_r = zp_angular_from_image(x_rand, pad_factor=ZP_PAD)
        E_null_all[i] = E_r
    del F, amp

    E_null_med = np.median(E_null_all, axis=0)
    E_null_lo  = np.percentile(E_null_all, 16, axis=0)
    E_null_hi  = np.percentile(E_null_all, 84, axis=0)
    del E_null_all

    # Smoothing & peak
    E_diff_zp      = E_theta_zp - E_null_med
    E_diff_zp_ref  = smooth_1d(E_diff_zp, SMOOTH_WIN_NULL)
    E_diff_zp_64   = smooth_1d(E_diff_zp_ref, SMOOTH_WIN_FINAL)

    peak_ref,   fwhm_ref,   _, l_ref,   r_ref   = fwhm_peak(theta_zp, E_diff_zp_ref)
    peak_ref_64, fwhm_ref_64, _, l_ref_64, r_ref_64 = fwhm_peak(theta_zp, E_diff_zp_64)

    # Bundle A
    png_img = plt.imread(str(png_path)) if png_path.exists() else raw_work
    save_bundle_A(
        out_dir / "bundle_A.png",
        opusid=opusid, raw_work=raw_work, img_proc=img_proc,
        log_power_full=log_power_full,
        theta_deg=theta_deg, E_theta=E_theta,
        peak_theta=peak_theta, pmr=pmr, snr=snr,
        png_img=png_img,
        peak_ref_64=peak_ref_64, fwhm_ref_64=fwhm_ref_64,
        l_ref_64=l_ref_64, r_ref_64=r_ref_64,
        theta_zp=theta_zp,
        E_diff_zp_ref=E_diff_zp_ref, E_diff_zp_ref_64=E_diff_zp_64,
        peak_ref=peak_ref,
        zp_db=zp_db, zp_peak_theta=zp_peak_theta,
        E_theta_zp=E_theta_zp, topk=topk,
        pmr_zp=pmr_zp, snr_zp=snr_zp, ZP_PAD_FACTOR=ZP_PAD,
        have_spice=have_spice, col=col, row=row, disc_r=disc_r,
        supp=supp, img_h=H, img_w=W,
    )

    # Spectra
    np.savez_compressed(
        out_dir / "spectra.npz",
        theta_deg=theta_deg, E_theta=E_theta,
        theta_zp=theta_zp, E_theta_zp=E_theta_zp,
        E_null_zp_med=E_null_med, E_null_zp_lo=E_null_lo, E_null_zp_hi=E_null_hi,
        E_diff_zp=E_diff_zp, E_diff_zp_ref=E_diff_zp_ref, E_diff_zp_ref_64=E_diff_zp_64,
    )

    # Metrics
    metrics = {
        "opusid": opusid, "image_id": image_id,
        "science_filter": science_filter, "science_time": str(science_time),
        "has_spice": bool(have_spice), "disc_r_px": float(disc_r),
        "disc_col": float(col), "disc_row": float(row),
        "suppression_pct": float(supp),
        "main_peak_theta_deg": float(peak_theta),
        "main_theta_band_deg": float((peak_theta + 90.0) % 180.0),
        "main_pmr": float(pmr), "main_snr": float(snr),
        "zp_peak_theta_deg": float(zp_peak_theta),
        "zp_pmr": float(pmr_zp), "zp_snr": float(snr_zp),
        "zp_pad_factor": int(ZP_PAD),
        "null_n": int(n_null), "null_seed": int(NULL_SEED),
        "null_peak_theta_ref_deg": float(peak_ref),
        "null_fwhm_ref_deg":       float(fwhm_ref),
        "null_peak_theta_64_deg":  float(peak_ref_64),
        "null_fwhm_64_deg":        float(fwhm_ref_64),
        "null_fwhm_left_deg":      float(l_ref_64),
        "null_fwhm_right_deg":     float(r_ref_64),
        "cfg_hp_sigma":            HP_SIGMA,
        "cfg_arcsinh_mult":        ARCSINH_MULT,
        "cfg_mask_radius_factor":  MASK_RADIUS_FACTOR,
        "cfg_dc_radius":           DC_RADIUS,
        "cfg_wedge_radius":        WEDGE_RADIUS,
        "cfg_zp_dc_radius":        ZP_DC_RADIUS,
        "cfg_zp_r_max":            ZP_R_MAX,
        "cfg_smooth_win_null":     SMOOTH_WIN_NULL,
        "cfg_smooth_win_final":    SMOOTH_WIN_FINAL,
        "cfg_n_theta_bins":        N_THETA_BINS,
        "status": "ok",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def worker_init() -> None:
    load_spice()

# ─────────────────────────── main ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Band detection (step 07)")
    parser.add_argument("--test",      action="store_true",
                        help=f"Single test image ({TEST_OPUSID})")
    parser.add_argument("--opusid",    default=None)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--n-null",    type=int, default=100,
                        help="Null surrogates per image (default: 100)")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip images with existing metrics.json")
    args = parser.parse_args()

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    if args.test or args.opusid:
        target = args.opusid or TEST_OPUSID
        npz    = CISSCAL_DIR / target / "residual.npz"
        if not npz.exists():
            raise SystemExit(f"ERROR: {npz} not found")
        jobs = [(target, "unknown", "unknown", args.n_null)]
        print(f"Test mode: {target}")
    else:
        # Build jobs from cisscal_output directory (any residual.npz present)
        pairs_meta = {}
        if PAIRS_FILE.exists():
            df = pd.read_parquet(PAIRS_FILE)
            pairs_meta = df.set_index("science_opusid")[["science_filter", "science_time1"]].to_dict("index")

        jobs, skipped = [], 0
        for d in sorted(CISSCAL_DIR.iterdir()):
            if not (d / "residual.npz").exists():
                continue
            opusid = d.name
            if args.skip_done and (PIPELINE_DIR / opusid / "metrics.json").exists():
                skipped += 1
                continue
            meta = pairs_meta.get(opusid, {})
            jobs.append((opusid,
                         meta.get("science_filter", "unknown"),
                         meta.get("science_time1", "unknown"),
                         args.n_null))
        print(f"Jobs: {len(jobs)}  |  Skipped (done): {skipped}  |  Workers: {args.workers}")

    if not jobs:
        print("Nothing to do.")
        return

    all_metrics = []
    n_ok = n_err = 0

    if args.workers == 1 or len(jobs) == 1:
        worker_init()
        for job in tqdm(jobs, desc="Detecting", unit="img"):
            m = process_one(job)
            all_metrics.append(m)
            n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as exe:
            futures = {exe.submit(process_one, job): job[0] for job in jobs}
            with tqdm(total=len(futures), desc="Detecting", unit="img") as pbar:
                for fut in as_completed(futures):
                    m = fut.result()
                    all_metrics.append(m)
                    if m.get("status") == "ok":
                        n_ok += 1
                    else:
                        n_err += 1
                    pbar.update(1)

    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_parquet(PIPELINE_DIR / "summary.parquet", index=False)
        print(f"\nSummary → {PIPELINE_DIR / 'summary.parquet'}")

    print(f"Done.  OK={n_ok}  Errors={n_err}")


if __name__ == "__main__":
    main()
