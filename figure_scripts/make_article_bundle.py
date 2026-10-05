"""
make_article_bundle.py
──────────────────────
Publication-quality 3-panel detection diagnostic for the scientific article.
Replaces the auto-generated Bundle-A (8 panels) with a clean 1×3 layout:

  (a) Preprocessed residual + 7×7 band-direction mosaic
  (b) log|G_zp(u,v)|  — 2D ZP FFT magnitude (cropped annulus)
  (c) E_diff(θ)        — w=64, FWHM span, θ★ annotation; plotted against band
                         orientation θ = (wavevector angle + 90°) mod 180°

Intended print size: full text width (double column), 7.0 × 2.8 inch, 300 DPI.

Usage:
  python make_article_bundle.py N1616349628 --label positive1
  python make_article_bundle.py N1643188892 --label null

  # Note: metrics.json for N1616349628 lists science_filter = "IR1",
  # but the article caption currently says "IR3" — verify with the raw data.

Output:
  figure_scripts/output/bundle_A_{label}.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter

# ── Storage roots to search (in order) ────────────────────────────────────────
_CISSCAL_ROOTS = [
    Path("/mnt/storage/projects/Thesis/LuminousBandsInERing/cisscal_output"),
    Path("/mnt/storage/cisscal_output"),
]
_PIPELINE_ROOTS = [
    Path("/mnt/storage/projects/Thesis/LuminousBandsInERing/pipeline_output"),
    Path("/mnt/storage/pipeline_output"),
]
OUT_DIR = Path(__file__).parent / "output"

# ── Pipeline parameters (must match 08_detect_bands.py) ──────────────────────
HP_SIGMA      = 80
ARCSINH_MULT  = 0.05
MASK_FACTOR   = 1.5
MASK_TAPER_PX = 60
ZP_PAD        = 8
ZP_DC_RADIUS  = 1.0    # native cycles / image
ZP_R_MAX      = 6.0

# ── Figure layout — double column Icarus = 7.0" wide ─────────────────────────
FIG_W    = 7.0
FIG_H    = 2.8
DPI      = 300
FS_LABEL = 9
FS_TICK  = 8
FS_ANNOT = 8
LW_MAIN  = 1.5
LW_THIN  = 1.0
LW_VLINE = 1.2

# ── Colours (consistent with make_zp_threepanel.py) ──────────────────────────
COL_DIFF9  = "#7B1FA2"
COL_DIFF64 = "#1B5E20"
COL_PEAK   = "#D32F2F"
COL_MOSAIC = "#00E5FF"   # cyan mosaic lines


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find(roots: list[Path], opusid_full: str, filename: str) -> Path:
    for root in roots:
        p = root / opusid_full / filename
        if p.exists():
            return p
    paths = [str(r / opusid_full / filename) for r in roots]
    raise FileNotFoundError(f"{filename} not found for {opusid_full}. Tried:\n" +
                            "\n".join(paths))


def _preprocess(raw: np.ndarray, disc_col: float, disc_row: float,
                disc_r: float) -> np.ndarray:
    """Return arcsinh-stretched, normalised preprocessed image (same as pipeline)."""
    H, W = raw.shape
    yy, xx   = np.mgrid[0:H, 0:W]
    dist     = np.sqrt((xx - disc_col) ** 2 + (yy - disc_row) ** 2)
    r_inner  = disc_r * MASK_FACTOR
    t        = np.clip((dist - r_inner) / MASK_TAPER_PX, 0.0, 1.0)
    mask     = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)

    hp   = (raw - gaussian_filter(raw, sigma=HP_SIGMA)).astype(np.float32)
    work = hp.copy()
    work[mask < 0.5] = np.nan
    with np.errstate(all="ignore"):
        rm = np.nanmedian(work, axis=1, keepdims=True)
        work -= np.where(np.isfinite(rm), rm, 0.0)
        cm = np.nanmedian(work, axis=0, keepdims=True)
        work -= np.where(np.isfinite(cm), cm, 0.0)
    work  = np.nan_to_num(work, nan=0.0).astype(np.float32) * mask
    scale = max(float(np.std(work)) * ARCSINH_MULT, 1e-12)
    proc  = np.arcsinh(work / scale).astype(np.float32)
    lo, hi = np.percentile(proc[np.isfinite(proc)], [0.1, 99.9])
    img   = np.clip(proc, lo, hi)
    img   = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img.astype(np.float32)


def _zp_fft_display(proc: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """
    Compute the 2D ZP FFT log-magnitude display array.

    Returns:
        log_mag   — (ch, cw) float64, NaN where masked
        ccx, ccy  — centre coordinates in the cropped array
        crop_r_px — radius of the unmasked annulus in pixels
    """
    H, W = proc.shape
    P    = ZP_PAD
    ph, pw = H * P, W * P
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:H, :W] = proc - float(np.mean(proc))
    Fz     = sfft.fftshift(sfft.fft2(padded))
    mag_zp = np.abs(Fz).astype(np.float32)
    del Fz, padded

    crop_r_px = int(round(ZP_R_MAX * P))
    cy_zp, cx_zp = (ph - 1) / 2.0, (pw - 1) / 2.0
    icy, icx = int(round(cy_zp)), int(round(cx_zp))
    mag_crop = mag_zp[max(0, icy - crop_r_px):icy + crop_r_px + 1,
                      max(0, icx - crop_r_px):icx + crop_r_px + 1]
    del mag_zp

    ch, cw   = mag_crop.shape
    yyc, xxc = np.indices((ch, cw), dtype=np.float32)
    ccy, ccx = (ch - 1) / 2.0, (cw - 1) / 2.0
    rrc_px   = np.sqrt((xxc - ccx) ** 2 + (yyc - ccy) ** 2)

    display = mag_crop.astype(np.float64)
    dc_r_px = ZP_DC_RADIUS * P
    display[rrc_px < dc_r_px]   = np.nan
    display[rrc_px > crop_r_px] = np.nan

    log_mag = np.log10(display + 1e-12)
    return log_mag, ccx, ccy, float(crop_r_px)


# ── Main figure builder ───────────────────────────────────────────────────────

def make_bundle(image_id: str, label: str) -> None:
    image_id     = image_id.upper().lstrip("N").lstrip("0")
    image_id_str = image_id.zfill(10)          # e.g. "1616349628"
    opusid_full  = f"co-iss-n{image_id_str}"

    print(f"\n{'─'*60}")
    print(f"Building bundle for {opusid_full}  →  bundle_A_{label}.png")
    print(f"{'─'*60}")

    # Load data
    residual_path = _find(_CISSCAL_ROOTS,  opusid_full, "residual.npz")
    spectra_path  = _find(_PIPELINE_ROOTS, opusid_full, "spectra.npz")
    metrics_path  = _find(_PIPELINE_ROOTS, opusid_full, "metrics.json")

    m = json.loads(metrics_path.read_text())
    disc_col        = float(m["disc_col"])
    disc_row        = float(m["disc_row"])
    disc_r          = float(m["disc_r_px"])
    peak_theta_64   = float(m["null_peak_theta_64_deg"])
    theta_band      = (peak_theta_64 + 90.0) % 180.0
    fwhm_left       = float(m["null_fwhm_left_deg"])
    fwhm_right      = float(m["null_fwhm_right_deg"])
    zp_peak_theta   = float(m["zp_peak_theta_deg"])
    sci_filter      = str(m["science_filter"])
    sci_time        = str(m["science_time"])[:10]   # YYYY-MM-DD

    print(f"  filter={sci_filter}, date={sci_time}")
    print(f"  θ★(w=64)={peak_theta_64:.1f}°  →  band={theta_band:.1f}°")
    print(f"  ZP peak={zp_peak_theta:.1f}°")

    raw_data = np.load(residual_path)
    raw      = raw_data["residual"].astype(np.float64)

    spec       = np.load(spectra_path)
    theta_zp   = spec["theta_zp"]
    E_diff9    = spec["E_diff_zp_ref"]
    E_diff64   = spec["E_diff_zp_ref_64"]

    # Re-index by band orientation: bands run perpendicular to the wavevector
    order      = np.argsort((theta_zp + 90.0) % 180.0)
    theta_zp   = ((theta_zp + 90.0) % 180.0)[order]
    E_diff9    = E_diff9[order]
    E_diff64   = E_diff64[order]
    band_left  = (fwhm_left + 90.0) % 180.0
    band_right = (fwhm_right + 90.0) % 180.0

    # Preprocessing + ZP FFT
    print("  Preprocessing …")
    proc = _preprocess(raw, disc_col, disc_row, disc_r)
    H, W = proc.shape

    print("  ZP FFT (may take ~30 s) …")
    log_mag, ccx, ccy, crop_r_px = _zp_fft_display(proc)
    ch, cw = log_mag.shape

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # ── Panel (a): preprocessed residual + 7×7 band-direction mosaic ──────────
    ax = axes[0]
    ax.imshow(proc, cmap="gray", origin="upper", interpolation="nearest")

    seg   = 0.055 * min(H, W)
    a_rad = math.radians(theta_band)
    dx, dy = math.cos(a_rad), math.sin(a_rad)
    for iy in range(7):
        for ix in range(7):
            cxx = (ix + 0.5) * W / 7
            cyy = (iy + 0.5) * H / 7
            ax.plot([cxx - seg * dx, cxx + seg * dx],
                    [cyy - seg * dy, cyy + seg * dy],
                    color=COL_MOSAIC, lw=0.8, alpha=0.70,
                    solid_capstyle="round")

    ax.text(0.03, 0.97, "(a)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FS_ANNOT, color="white", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # ── Panel (b): 2D log|G_zp| ───────────────────────────────────────────────
    ax = axes[1]
    ax.imshow(log_mag, cmap="inferno", origin="upper",
              vmin=np.nanpercentile(log_mag, 2),
              vmax=np.nanpercentile(log_mag, 99.5),
              interpolation="nearest")

    # ZP peak axis
    a_pk  = math.radians(zp_peak_theta)
    r_ln  = crop_r_px * 0.88
    ax.plot([ccx - math.cos(a_pk) * r_ln, ccx + math.cos(a_pk) * r_ln],
            [ccy - math.sin(a_pk) * r_ln, ccy + math.sin(a_pk) * r_ln],
            color="cyan", lw=LW_THIN, alpha=0.9)

    # Frequency-unit tick labels
    n_ticks  = 5
    tick_px  = np.linspace(0, cw - 1, n_ticks)
    tick_val = (tick_px - ccx) / float(ZP_PAD)
    ax.set_xticks(tick_px)
    ax.set_xticklabels([f"{v:.1f}" for v in tick_val], fontsize=FS_TICK)
    tick_py  = np.linspace(0, ch - 1, n_ticks)
    tick_vy  = (tick_py - ccy) / float(ZP_PAD)
    ax.set_yticks(tick_py)
    ax.set_yticklabels([f"{v:.1f}" for v in tick_vy], fontsize=FS_TICK)
    ax.set_xlabel(r"$u$ (cyc img$^{-1}$)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$v$ (cyc img$^{-1}$)", fontsize=FS_LABEL)
    ax.text(0.03, 0.97, "(b)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FS_ANNOT, color="white", fontweight="bold")

    # ── Panel (c): E_diff(θ) ──────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(theta_zp, E_diff64, color=COL_DIFF64, lw=LW_MAIN,
            label=r"$w=64$")
    ax.axvline(theta_band, color=COL_PEAK, ls="--", lw=LW_VLINE)
    if band_left <= band_right:
        ax.axvspan(band_left, band_right, alpha=0.13, color="green")
    else:   # FWHM wraps across 0°/180°
        ax.axvspan(band_left, 180.0, alpha=0.13, color="green")
        ax.axvspan(0.0, band_right, alpha=0.13, color="green")
    ax.axhline(0, color="black", lw=0.5, ls=":")

    ymax = float(E_diff64.max())
    right = theta_band > 120.0   # keep the label inside the axes
    ax.text(theta_band + (-6 if right else 2), ymax * 0.95,
            f"$\\theta^\\star={theta_band:.0f}\\degree$",
            color=COL_PEAK, fontsize=FS_ANNOT, va="top",
            ha="right" if right else "left",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))

    ax.set_xlim(0, 180)
    ax.set_xlabel(r"Band orientation $\theta$ (deg)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$E_{\rm diff}(\theta)$", fontsize=FS_LABEL)
    ax.text(0.03, 0.97, r"(c)  $E_{\rm diff}(\theta)$",
            transform=ax.transAxes, va="top", ha="left", fontsize=FS_ANNOT)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    fig.tight_layout(pad=0.8)

    out = OUT_DIR / f"bundle_A_{label}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image_id", help="Image ID, e.g. N1616349628 or 1616349628")
    p.add_argument("--label", default=None,
                   help="Output filename suffix (default: image_id). "
                        "Use e.g. 'positive1' or 'null'.")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    label = args.label or args.image_id.upper().lstrip("N")
    make_bundle(args.image_id, label)
