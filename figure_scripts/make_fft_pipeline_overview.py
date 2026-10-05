"""
make_fft_pipeline_overview.py
──────────────────────────────
Article-ready recreation of fft_pipeline_overview.png (Figure 7).

Three panels for image pair N1711553312 / N1711553432 (IR3, orbit 163EN):
  (a) Preprocessed CLEAR-subtracted residual with disc mask boundary overlaid
  (b) 2D non-padded FFT log-magnitude (cropped to low-frequency centre)
  (c) Non-padded angular power spectrum E(θ) against band orientation
      θ = (wavevector angle + 90°) mod 180°, red dashed line at θ★ only

No orange band-orientation line in panel (c): that angle is not a frequency-
domain quantity and belongs only in the ZP detection figure (Figure 8).

Output:
    figure_scripts/output/fft_pipeline_overview.png
    Thesis/icarus_template_used/figs/fft_pipeline_overview.png
    (Thesis/chapters/figs/fft_pipeline_overview.png keeps the earlier
    wavevector-angle axis and is no longer written.)

Usage:
    python figure_scripts/make_fft_pipeline_overview.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/mnt/storage/projects/Thesis/LuminousBandsInERing")
RESIDUAL  = DATA_ROOT / "cisscal_output"  / "co-iss-n1711553432" / "residual.npz"
SPECTRA   = DATA_ROOT / "pipeline_output" / "co-iss-n1711553432" / "spectra.npz"

OUT_DIR     = Path(__file__).parent / "output"
OUT_ARTICLE = Path(__file__).parent.parent / "Thesis/icarus_template_used/figs/fft_pipeline_overview.png"
OUT_THESIS  = Path(__file__).parent.parent / "Thesis/chapters/figs/fft_pipeline_overview.png"

# ── Disc geometry ──────────────────────────────────────────────────────────────
DISC_COL, DISC_ROW, DISC_R_PX = 515.9, 1282.8, 361.0
MASK_FACTOR   = 1.5
MASK_TAPER_PX = 60

# ── Pipeline parameters ────────────────────────────────────────────────────────
HP_SIGMA     = 80
ARCSINH_MULT = 0.05
R_MAX_DISP   = 80.0     # cycles/image, display crop for panel (b)

# ── Figure style (Icarus double-column = 7.0 inch) ────────────────────────────
FIG_W  = 7.0
FIG_H  = 2.4
DPI    = 300
FS_LB  = 7    # axis labels
FS_TK  = 6    # tick labels
FS_AN  = 6.5  # annotations

COL_SPEC = "#1565C0"   # blue for E(θ)
COL_PEAK = "#D32F2F"   # red for θ★


# ══════════════════════════════════════════════════════════════════════════════
# Data loading and preprocessing (identical to make_zp_threepanel.py)
# ══════════════════════════════════════════════════════════════════════════════

res = np.load(RESIDUAL)
raw = res["residual"].astype(np.float64)
H, W = raw.shape

# Disc mask
yy, xx  = np.mgrid[0:H, 0:W]
dist    = np.sqrt((xx - DISC_COL)**2 + (yy - DISC_ROW)**2)
r_inner = DISC_R_PX * MASK_FACTOR
t       = np.clip((dist - r_inner) / MASK_TAPER_PX, 0.0, 1.0)
enc_mask = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)

# High-pass filter
hp   = (raw - gaussian_filter(raw, sigma=HP_SIGMA)).astype(np.float32)
work = hp.copy()
work[enc_mask < 0.5] = np.nan
with np.errstate(all="ignore"):
    rm = np.nanmedian(work, axis=1, keepdims=True)
    work -= np.where(np.isfinite(rm), rm, 0.0)
    cm = np.nanmedian(work, axis=0, keepdims=True)
    work -= np.where(np.isfinite(cm), cm, 0.0)
work     = np.nan_to_num(work, nan=0.0).astype(np.float32) * enc_mask
scale    = max(float(np.std(work)) * ARCSINH_MULT, 1e-12)
proc     = np.arcsinh(work / scale).astype(np.float32)
lo2, hi2 = np.percentile(proc[np.isfinite(proc)], [0.1, 99.9])
img_proc = np.clip(proc, lo2, hi2)
img_proc = (img_proc - img_proc.min()) / (img_proc.max() - img_proc.min() + 1e-12)

# ── Non-padded 2D FFT (for panel b display only) ───────────────────────────────
F   = sfft.fftshift(sfft.fft2(img_proc - float(np.mean(img_proc))))
mag = np.abs(F).astype(np.float32)
del F

# Frequency axes in cycles/image
kx = sfft.fftshift(sfft.fftfreq(W)) * W   # cycles per image
ky = sfft.fftshift(sfft.fftfreq(H)) * H

# ── Load pipeline-computed E(θ) for panel (c) ──────────────────────────────────
spec          = np.load(SPECTRA)
theta_k       = spec["theta_deg"]     # 720 bins, 0.25° each (wavevector angle)
E             = spec["E_theta"]       # main (non-padded) angular spectrum
# Re-index by band orientation: bands run perpendicular to the wavevector
order         = np.argsort((theta_k + 90.0) % 180.0)
theta_centres = ((theta_k + 90.0) % 180.0)[order]
E             = E[order]
peak_idx      = int(np.argmax(E))
peak_theta    = float(theta_centres[peak_idx])   # band orientation at the main-FFT peak

# ── Crop 2D FFT for display ────────────────────────────────────────────────────
cy_px, cx_px = (H - 1) / 2.0, (W - 1) / 2.0
crop_px = int(round(R_MAX_DISP / (W / float(W)) ))   # 1 px = 1 cycle/image here
# simpler: R_MAX_DISP cycles/image, and kx is in cycles/image with 1px = 1 cycle
crop_px = int(R_MAX_DISP)
icy, icx = int(round(cy_px)), int(round(cx_px))
mag_crop = mag[max(0, icy - crop_px): icy + crop_px + 1,
               max(0, icx - crop_px): icx + crop_px + 1]
kx_crop  = kx[max(0, icx - crop_px): icx + crop_px + 1]
ky_crop  = ky[max(0, icy - crop_px): icy + crop_px + 1]

log_crop = np.log10(mag_crop.astype(np.float64) + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                         gridspec_kw={"wspace": 0.35})
fig.patch.set_facecolor("white")

# ── Panel (a): preprocessed residual ──────────────────────────────────────────
ax = axes[0]
ax.imshow(img_proc, cmap="gray", origin="upper",
          interpolation="nearest", vmin=0, vmax=1)

# Disc boundary (dashed white) and mask boundary (solid yellow)
theta_arc = np.linspace(0, 2 * np.pi, 360)
r_disc  = DISC_R_PX
r_mask  = DISC_R_PX * MASK_FACTOR
ax.plot(DISC_COL + r_disc * np.cos(theta_arc),
        DISC_ROW + r_disc * np.sin(theta_arc),
        color="white", lw=0.6, ls="--")
ax.plot(DISC_COL + r_mask * np.cos(theta_arc),
        DISC_ROW + r_mask * np.sin(theta_arc),
        color="yellow", lw=0.6, ls="-")

ax.set_xlim(0, W - 1)
ax.set_ylim(H - 1, 0)
ax.set_xticks([0, 256, 512, 768, 1023])
ax.set_yticks([0, 256, 512, 768, 1023])
ax.set_xticklabels(["0", "256", "512", "768", "1024"], fontsize=FS_TK)
ax.set_yticklabels(["0", "256", "512", "768", "1024"], fontsize=FS_TK)
ax.set_xlabel("Sample [px]", fontsize=FS_LB)
ax.set_ylabel("Line [px]", fontsize=FS_LB)
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
        va="top", ha="left", fontsize=FS_LB + 1, color="white", fontweight="bold")

# ── Panel (b): 2D FFT log-magnitude ───────────────────────────────────────────
ax = axes[1]
extent = [kx_crop[0], kx_crop[-1], ky_crop[-1], ky_crop[0]]
ax.imshow(log_crop, cmap="inferno", origin="upper",
          extent=extent,
          vmin=np.nanpercentile(log_crop, 2),
          vmax=np.nanpercentile(log_crop, 99.5),
          interpolation="nearest", aspect="equal")

# Same range and ticks on both frequency axes (square panel; reviewer BS79)
ticks = [-R_MAX_DISP, -R_MAX_DISP / 2, 0, R_MAX_DISP / 2, R_MAX_DISP]
ax.set_xlim(-R_MAX_DISP, R_MAX_DISP)
ax.set_ylim(R_MAX_DISP, -R_MAX_DISP)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"Frequency $x$ (cycles image$^{-1}$)", fontsize=FS_LB)
ax.set_ylabel(r"Frequency $y$ (cycles image$^{-1}$)", fontsize=FS_LB)
ax.tick_params(labelsize=FS_TK)
ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
        va="top", ha="left", fontsize=FS_LB + 1, color="white", fontweight="bold")

# ── Panel (c): non-padded E(θ) ────────────────────────────────────────────────
ax = axes[2]
E_norm = E / (E.max() + 1e-12)
ax.plot(theta_centres, E_norm, color=COL_SPEC, lw=0.6)
ax.axvline(peak_theta, color=COL_PEAK, lw=1.0, ls="--")
ax.text(peak_theta + 2, 0.88,
        fr"$\theta^\star = {peak_theta:.0f}°$",
        color=COL_PEAK, fontsize=FS_AN, va="top")

ax.set_xlim(0, 180)
ax.set_ylim(bottom=0)
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xlabel(r"Band orientation $\theta$ [°]", fontsize=FS_LB)
ax.set_ylabel(r"$E(\theta)$ (normalised)", fontsize=FS_LB)
ax.tick_params(labelsize=FS_TK)
ax.grid(True, alpha=0.2)
ax.text(0.03, 0.97, "(c)", transform=ax.transAxes,
        va="top", ha="left", fontsize=FS_LB + 1, fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
for out in [OUT_DIR / "fft_pipeline_overview.png", OUT_ARTICLE]:
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
plt.close(fig)
