"""
make_zp_threepanel.py
─────────────────────
3-panel ZP detection figure for article / thesis ch4.
  (a) log|G_zp(u,v)| — 2D ZP FFT magnitude, cropped to accepted annulus
  (b) E_zp(θ) vs null-ensemble (median + 16/84th pct)
  (c) E_diff(θ) = E_zp - E_null, w=9 and w=64 smoothing

Pair: N1711553312 / N1711553432  (IR3, orbit 163EN, 2012-03-27)
Spectra in (b) and (c) are plotted against band orientation θ = (wavevector angle + 90°) mod 180°.
Actual peak value (from data): θ* ≈ 40° (wavevector angle ≈ 130°)
"""

from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from scipy import fft as sfft

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/mnt/storage/projects/Thesis/LuminousBandsInERing")
RESIDUAL    = DATA_ROOT / "cisscal_output" / "co-iss-n1711553432" / "residual.npz"
SPECTRA     = DATA_ROOT / "pipeline_output" / "co-iss-n1711553432" / "spectra.npz"

OUT_THESIS  = Path("/home/arnaudm/projects/LuminousBandsInERing/Thesis/chapters/figs/zp_threepanel.png")
OUT_ARTICLE = Path("/home/arnaudm/projects/LuminousBandsInERing/Thesis/icarus_template_used/figs/zp_threepanel.png")

# ── Disc geometry (from ch4 figure description, Enceladus off-screen) ─────────
DISC_COL, DISC_ROW, DISC_R_PX = 515.9, 1282.8, 361.0
MASK_FACTOR, MASK_TAPER_PX    = 1.5, 60

# ── Pipeline parameters ────────────────────────────────────────────────────────
HP_SIGMA      = 80
ARCSINH_MULT  = 0.05
ZP_PAD        = 8
ZP_DC_RADIUS  = 1.0    # native cycles/image
ZP_R_MAX      = 6.0
EXCLUDE_AXIS  = 5.0

# ── Font / style sizes ─────────────────────────────────────────────────────────
FS_TITLE  = 15
FS_LABEL  = 13
FS_TICK   = 11
FS_ANNOT  = 12
LW_MAIN   = 2.2
LW_THIN   = 1.5
LW_VLINE  = 1.8

# ── Colours ────────────────────────────────────────────────────────────────────
COL_ZP    = "#1565C0"
COL_NULL  = "#757575"
COL_DIFF9 = "#7B1FA2"
COL_DIFF64= "#1B5E20"
COL_PEAK  = "#D32F2F"

# ── Load data ──────────────────────────────────────────────────────────────────
res = np.load(RESIDUAL)
raw = res["residual"].astype(np.float64)
H, W = raw.shape

spec       = np.load(SPECTRA)
theta_zp   = spec["theta_zp"]
E_zp       = spec["E_theta_zp"]
E_null_med = spec["E_null_zp_med"]
E_null_lo  = spec["E_null_zp_lo"]
E_null_hi  = spec["E_null_zp_hi"]
E_diff9    = spec["E_diff_zp_ref"]
E_diff64   = spec["E_diff_zp_ref_64"]

# ── Re-index spectra by band orientation ───────────────────────────────────────
# The pipeline bins power by wavevector angle; bands run perpendicular to it, so
# band orientation = (wavevector angle + 90°) mod 180°. Plotting against band
# orientation lets the peak read directly as the band angle in the image.
theta_k    = theta_zp                               # wavevector angle
order      = np.argsort((theta_k + 90.0) % 180.0)
theta_zp   = ((theta_k + 90.0) % 180.0)[order]      # band orientation
E_zp, E_null_med, E_null_lo, E_null_hi, E_diff9, E_diff64 = (
    a[order] for a in (E_zp, E_null_med, E_null_lo, E_null_hi, E_diff9, E_diff64))

# ── Derive peak angles from data ───────────────────────────────────────────────
peak_idx   = int(np.argmax(E_diff64))
band_theta = float(theta_zp[peak_idx])          # band orientation θ* ≈ 40°
peak_theta = (band_theta + 90.0) % 180.0        # wavevector angle ≈ 130° (panel a line)

half_max = float(E_diff64[peak_idx]) / 2.0
left_t = float(theta_zp[0])
for i in range(peak_idx - 1, -1, -1):
    if E_diff64[i] < half_max:
        left_t = float(theta_zp[i])
        break
right_t = float(theta_zp[-1])
for i in range(peak_idx + 1, len(E_diff64)):
    if E_diff64[i] < half_max:
        right_t = float(theta_zp[i])
        break

# ── Build cosine-taper disc mask ───────────────────────────────────────────────
yy, xx   = np.mgrid[0:H, 0:W]
dist     = np.sqrt((xx - DISC_COL)**2 + (yy - DISC_ROW)**2)
r_inner  = DISC_R_PX * MASK_FACTOR
t        = np.clip((dist - r_inner) / MASK_TAPER_PX, 0.0, 1.0)
enc_mask = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)

# ── Preprocessing ──────────────────────────────────────────────────────────────
hp   = (raw - gaussian_filter(raw, sigma=HP_SIGMA)).astype(np.float32)
work = hp.copy()
work[enc_mask < 0.5] = np.nan
with np.errstate(all="ignore"):
    rm = np.nanmedian(work, axis=1, keepdims=True)
    work -= np.where(np.isfinite(rm), rm, 0.0)
    cm = np.nanmedian(work, axis=0, keepdims=True)
    work -= np.where(np.isfinite(cm), cm, 0.0)
work = np.nan_to_num(work, nan=0.0).astype(np.float32) * enc_mask
scale    = max(float(np.std(work)) * ARCSINH_MULT, 1e-12)
proc     = np.arcsinh(work / scale).astype(np.float32)
lo2, hi2 = np.percentile(proc[np.isfinite(proc)], [0.1, 99.9])
img_proc = np.clip(proc, lo2, hi2)
img_proc = (img_proc - img_proc.min()) / (img_proc.max() - img_proc.min() + 1e-12)

# ── ZP FFT ─────────────────────────────────────────────────────────────────────
P      = ZP_PAD
ph, pw = H * P, W * P
padded = np.zeros((ph, pw), dtype=np.float32)
padded[:H, :W] = img_proc - float(np.mean(img_proc))
Fz     = sfft.fftshift(sfft.fft2(padded))
mag_zp = np.abs(Fz).astype(np.float32);  del Fz, padded

cy_zp, cx_zp = (ph - 1) / 2.0, (pw - 1) / 2.0
crop_r_px    = int(round(ZP_R_MAX * P))
icy, icx     = int(round(cy_zp)), int(round(cx_zp))
mag_crop = mag_zp[max(0,icy-crop_r_px):min(ph,icy+crop_r_px+1),
                  max(0,icx-crop_r_px):min(pw,icx+crop_r_px+1)];  del mag_zp

ch, cw   = mag_crop.shape
yyc, xxc = np.indices((ch, cw), dtype=np.float32)
ccy, ccx = (ch-1)/2.0, (cw-1)/2.0
rrc_px   = np.sqrt((xxc-ccx)**2 + (yyc-ccy)**2)

display  = mag_crop.astype(np.float64)
dc_r_px  = ZP_DC_RADIUS * P
display[rrc_px < dc_r_px]   = np.nan
display[rrc_px > crop_r_px] = np.nan

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6.0))
fig.patch.set_facecolor("white")

# ── Panel (a): log|G_zp| 2D ───────────────────────────────────────────────────
ax = axes[0]
log_mag = np.log10(display + 1e-12)
ax.imshow(log_mag, cmap="inferno", origin="upper",
          vmin=np.nanpercentile(log_mag, 2),
          vmax=np.nanpercentile(log_mag, 99.5),
          interpolation="nearest")

# Peak direction line through centre (both directions)
cy_d, cx_d = (ch-1)/2.0, (cw-1)/2.0
angle_rad  = math.radians(peak_theta)
r_line     = crop_r_px * 0.88
dx = math.cos(angle_rad) * r_line
dy = math.sin(angle_rad) * r_line
ax.plot([cx_d-dx, cx_d+dx], [cy_d-dy, cy_d+dy],
        color="cyan", lw=LW_THIN, alpha=0.9)

ax.set_title(r"(a) $\log|G_\mathrm{zp}(u,v)|$", fontsize=FS_TITLE)
ax.set_xlabel(r"$u$ (cycles image$^{-1}$)", fontsize=FS_LABEL)
ax.set_ylabel(r"$v$ (cycles image$^{-1}$)", fontsize=FS_LABEL)

n_ticks  = 5
tick_px  = np.linspace(0, cw-1, n_ticks)
tick_val = (tick_px - cx_d) / float(P)
ax.set_xticks(tick_px)
ax.set_xticklabels([f"{v:.1f}" for v in tick_val], fontsize=FS_TICK)
tick_py  = np.linspace(0, ch-1, n_ticks)
tick_vy  = (tick_py - cy_d) / float(P)
ax.set_yticks(tick_py)
ax.set_yticklabels([f"{v:.1f}" for v in tick_vy], fontsize=FS_TICK)

# ── Panel (b): E_zp vs null ────────────────────────────────────────────────────
ax = axes[1]
norm_max = E_zp.max() + 1e-12
E_n   = E_zp       / norm_max
n_med = E_null_med / norm_max
n_lo  = E_null_lo  / norm_max
n_hi  = E_null_hi  / norm_max

ax.fill_between(theta_zp, n_lo, n_hi, color=COL_NULL, alpha=0.30)
ax.plot(theta_zp, n_med, color=COL_NULL, lw=LW_THIN, ls="--")
ax.plot(theta_zp, E_n,   color=COL_ZP,  lw=LW_MAIN)

# Annotate lines with text instead of legend
ax.text(100, 0.14, r"$E_\mathrm{zp}(\theta)$", color=COL_ZP,
        fontsize=FS_ANNOT-1, va="bottom")
ax.text(100, 0.07, "null median", color=COL_NULL,
        fontsize=FS_ANNOT-1, va="bottom")

ax.set_xlim(0, 180)
ax.set_xlabel(r"Band orientation $\theta$ (deg)", fontsize=FS_LABEL)
ax.set_ylabel("Normalised angular power", fontsize=FS_LABEL)
ax.set_title(r"(b) $E_\mathrm{zp}(\theta)$ vs null ensemble", fontsize=FS_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

# ── Panel (c): E_diff ──────────────────────────────────────────────────────────
ax = axes[2]
ax.plot(theta_zp, E_diff64, color=COL_DIFF64, lw=LW_MAIN)
ax.axvline(band_theta, color=COL_PEAK,  ls="--", lw=LW_VLINE)
ax.axvspan(left_t, right_t, alpha=0.13, color="green")
ax.axhline(0, color="black", lw=0.7, ls=":")

# Text labels
ymax = float(E_diff64.max())
ax.text(band_theta + 2, ymax * 0.95,
        f"$\\theta^\\star={band_theta:.0f}°$",
        color=COL_PEAK, fontsize=FS_ANNOT, va="top")

# Half-maximum bar between the interpolated crossings reported by the pipeline
met  = json.loads((SPECTRA.parent / "metrics.json").read_text())
hm_l = (met["null_fwhm_left_deg"] + 90.0) % 180.0     # wavevector -> band axis
hm_r = (met["null_fwhm_right_deg"] + 90.0) % 180.0
fwhm = met["null_fwhm_64_deg"]
ax.annotate("", xy=(hm_l, ymax / 2), xytext=(hm_r, ymax / 2),
            arrowprops=dict(arrowstyle="|-|", color="black", lw=1.2,
                            shrinkA=0, shrinkB=0, mutation_scale=6))
ax.text(hm_r + 2, ymax / 2,
        f"FWHM $= {fwhm:.0f}°$ ($\\pm{fwhm / 2:.1f}°$)",
        color="black", fontsize=FS_ANNOT, va="center", ha="left")

ax.set_xlim(0, 180)
ax.set_xlabel(r"Band orientation $\theta$ (deg)", fontsize=FS_LABEL)
ax.set_ylabel(r"$E_\mathrm{zp} - E_\mathrm{null}$ (excess power)", fontsize=FS_LABEL)
ax.set_title(r"(c) $E_\mathrm{diff}(\theta)$", fontsize=FS_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

fig.tight_layout(pad=1.5)
# Article only: the thesis copy (OUT_THESIS) keeps the earlier wavevector-angle axis
fig.savefig(OUT_ARTICLE, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_ARTICLE}")
plt.close(fig)
