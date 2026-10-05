"""
make_profile_example.py
────────────────────────
Two-panel profile-extraction figure for article Figure 6.
Example image: co-iss-n1711553432 (IR3, orbit 163EN, 2012-03-27)
  delta = 2.94e-5 I/F,  SNR = 7.0

Panel (a): preprocessed display residual with 15-slice bundle overlay
Panel (b): smoothed stacked mean profile (raw I/F) with max/min picks
"""

from pathlib import Path
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter, map_coordinates

# ── Paths ──────────────────────────────────────────────────────────────────────
OPUSID      = "co-iss-n1711553432"
DATA_ROOT   = Path("/home/arnaudm/projects/LuminousBandsInERing/mnt-storage")
RESIDUAL    = DATA_ROOT / "cisscal_output" / OPUSID / "residual.npz"
PICKS_JSON  = DATA_ROOT / "post_pipeline" / "13_profile_slices" / OPUSID / "picks.json"

OUT_THESIS  = Path("/home/arnaudm/projects/LuminousBandsInERing/Thesis/chapters/figs/profile_example.png")
OUT_ARTICLE = Path("/home/arnaudm/projects/LuminousBandsInERing/Thesis/icarus_template_used/figs/profile_example.png")

# ── Preprocessing parameters (same as notebook 13 / pipeline) ─────────────────
HP_SIGMA     = 80.0
ARCSINH_MULT = 0.05
SLICE_SAMPLES = 2400
SMOOTH_WINDOW = 101

# ── Style ──────────────────────────────────────────────────────────────────────
FS_TITLE = 14
FS_LABEL = 12
FS_TICK  = 10
FS_ANNOT = 11
LW_SLICE = 0.6
COL_MAX  = "#D32F2F"   # red
COL_MIN  = "#1565C0"   # blue
COL_PROF = "#37474F"   # dark grey profile
COL_SMTH = "#1B5E20"   # dark green smoothed mean
COL_SPAN = "#FFF9C4"   # pale yellow band span

# ── Load ───────────────────────────────────────────────────────────────────────
arr     = np.load(RESIDUAL)
raw_res = np.asarray(arr["residual"], dtype=np.float64)
mask    = np.asarray(arr["mask"],     dtype=bool) if "mask" in arr else np.isfinite(raw_res)
raw_res = np.nan_to_num(raw_res, nan=0.0)
H, W    = raw_res.shape

picks_data  = json.loads(PICKS_JSON.read_text())
sl          = picks_data["slicer"]
cx, cy      = sl["cx_px"], sl["cy_px"]
theta_slice = sl["theta_slice_deg"]
theta_band  = sl["theta_band_deg"]
n_slices    = sl.get("n_slices", 15)
spacing     = sl.get("slice_spacing_px", 8.0)
max_picks   = picks_data["picks"]["max"]
min_picks   = picks_data["picks"]["min"]

# ── Build preprocessed display image ──────────────────────────────────────────
def build_display(r, m):
    hp   = r - gaussian_filter(r, sigma=HP_SIGMA)
    work = hp.copy(); work[~m] = np.nan
    with np.errstate(all="ignore"):
        rm = np.nanmedian(work, axis=1, keepdims=True)
        work -= np.where(np.isfinite(rm), rm, 0.0)
        cm = np.nanmedian(work, axis=0, keepdims=True)
        work -= np.where(np.isfinite(cm), cm, 0.0)
    work = np.nan_to_num(work, nan=0.0) * m.astype(np.float64)
    scale = max(float(np.std(work)) * ARCSINH_MULT, 1e-12)
    proc  = np.arcsinh(work / scale)
    fin   = np.isfinite(proc)
    lo, hi = np.percentile(proc[fin], [0.1, 99.9])
    img = np.clip(proc, lo, hi)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img.astype(np.float32)

disp = build_display(raw_res, mask)

# ── Multi-slice profile ────────────────────────────────────────────────────────
def sample_line(r, m, cx0, cy0, ang_deg, n_samp):
    a = math.radians(ang_deg)
    dx, dy = math.cos(a), math.sin(a)
    L = float(math.hypot(W, H))
    t = np.linspace(-L, L, n_samp)
    x = cx0 + t * dx; y = cy0 + t * dy
    inb = (x >= 0) & (x <= W-1) & (y >= 0) & (y <= H-1)
    vals = np.full_like(t, np.nan)
    if inb.any():
        idx = np.where(inb)[0]
        rv  = map_coordinates(r, [y[inb], x[inb]], order=1, mode="nearest")
        mv  = map_coordinates(m.astype(np.float32), [y[inb], x[inb]], order=0, mode="nearest") > 0.5
        vals[idx[mv]] = rv[mv]
    return t, vals

a_rad   = math.radians(theta_slice)
bx, by  = -math.sin(a_rad), math.cos(a_rad)  # band direction (perp to slice)
offs    = (np.arange(n_slices) - (n_slices - 1) / 2.0) * spacing

all_vals = []
t_ref    = None
for o in offs:
    t, v = sample_line(raw_res, mask, cx + o*bx, cy + o*by, theta_slice, SLICE_SAMPLES)
    if t_ref is None: t_ref = t
    all_vals.append(v)

stack = np.vstack(all_vals)
cnt   = np.sum(np.isfinite(stack), axis=0)
mean_raw = np.divide(np.nansum(stack, axis=0), cnt,
                     out=np.full(cnt.shape, np.nan), where=cnt > 0)

# Smooth
k    = np.ones(SMOOTH_WINDOW)
fin  = np.isfinite(mean_raw)
num  = np.convolve(np.where(fin, mean_raw, 0.0), k, mode="same")
den  = np.convolve(fin.astype(float), k, mode="same")
smoothed = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

# ── Identify the contrast picks ────────────────────────────────────────────────
def profile_at(tp):
    v = np.isfinite(smoothed)
    return float(smoothed[v][np.argmin(np.abs(t_ref[v] - tp))])

max_vals = np.array([profile_at(p["t_px"]) for p in max_picks])
min_vals = np.array([profile_at(p["t_px"]) for p in min_picks])
best_idx = int(np.argmax(max_vals))
I_max    = max_vals[best_idx]
t_max    = max_picks[best_idx]["t_px"]

left  = [(v, p) for v, p in zip(min_vals, min_picks) if p["t_px"] < t_max]
right = [(v, p) for v, p in zip(min_vals, min_picks) if p["t_px"] > t_max]
candidates = []
if left:  candidates.append(min(left,  key=lambda x: abs(x[1]["t_px"] - t_max)))
if right: candidates.append(min(right, key=lambda x: abs(x[1]["t_px"] - t_max)))
I_min = float(np.mean([v for v, _ in candidates]))
delta = I_max - I_min
flanking_picks = [p for _, p in candidates]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("white")

# ── Panel (a): display image with slice overlay ────────────────────────────────
ax_img.imshow(disp, cmap="gray", origin="upper", interpolation="nearest")

# Draw each of the 15 slices
for o in offs:
    x0, y0 = cx + o*bx, cy + o*by
    t_lo, t_hi = -math.hypot(W, H)*0.75, math.hypot(W, H)*0.75
    x1 = x0 + t_lo * math.cos(a_rad);  y1 = y0 + t_lo * math.sin(a_rad)
    x2 = x0 + t_hi * math.cos(a_rad);  y2 = y0 + t_hi * math.sin(a_rad)
    ax_img.plot([x1, x2], [y1, y2], color="orange", lw=LW_SLICE, alpha=0.7)

# Mark the band direction arrow
blen = 120
ax_img.annotate("", xy=(cx + blen*bx, cy + blen*by),
                xytext=(cx - blen*bx, cy - blen*by),
                arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.5))
ax_img.text(cx + blen*bx + 8, cy + blen*by - 8,
            rf"$\theta^\star_{{\mathrm{{band}}}}={theta_band:.0f}°$",
            color="cyan", fontsize=FS_ANNOT, va="top")

ax_img.set_xlim(0, W); ax_img.set_ylim(H, 0)
ax_img.set_title(rf"(a) {OPUSID}", fontsize=FS_TITLE)
ax_img.set_xlabel("Column (px)", fontsize=FS_LABEL)
ax_img.set_ylabel("Row (px)", fontsize=FS_LABEL)
ax_img.tick_params(labelsize=FS_TICK)

# ── Panel (b): smoothed stacked mean profile ───────────────────────────────────
# Individual raw slice profiles (faint)
for v in all_vals:
    ax_prof.plot(t_ref, v * 1e5, color=COL_PROF, lw=0.35, alpha=0.3)

# Smoothed mean
ax_prof.plot(t_ref, smoothed * 1e5, color=COL_SMTH, lw=2.0, label="Smoothed mean")

# Max picks (red circles)
for i, p in enumerate(max_picks):
    val = profile_at(p["t_px"]) * 1e5
    ax_prof.plot(p["t_px"], val, "o", color=COL_MAX, ms=8, zorder=5,
                 label="Max pick" if i == 0 else "")

# Min picks (blue circles) — all
for i, p in enumerate(min_picks):
    val = profile_at(p["t_px"]) * 1e5
    ax_prof.plot(p["t_px"], val, "s", color=COL_MIN, ms=8, zorder=5,
                 label="Min pick" if i == 0 else "")

# Flanking min markers (filled, with dashed line to I_min level)
t_flanks = [p["t_px"] for p in flanking_picks]
ax_prof.axhline(I_min * 1e5, color=COL_MIN, ls="--", lw=1.0, alpha=0.7)
ax_prof.axhline(I_max * 1e5, color=COL_MAX, ls="--", lw=1.0, alpha=0.7)

# delta annotation
t_ann = t_max + 130
ax_prof.annotate("", xy=(t_ann, I_max*1e5), xytext=(t_ann, I_min*1e5),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
ax_prof.text(t_ann + 20, (I_max + I_min) / 2 * 1e5,
             rf"$\delta={delta*1e5:.2f}\times10^{{-5}}\,I/F$",
             fontsize=FS_ANNOT, va="center")

ax_prof.set_xlabel(r"Profile coordinate $t$ (px)", fontsize=FS_LABEL)
ax_prof.set_ylabel(r"Residual $I_\mathrm{res}$ ($\times 10^{-5}$ $I/F$)", fontsize=FS_LABEL)
ax_prof.set_title(r"(b) Smoothed stacked profile", fontsize=FS_TITLE)
ax_prof.tick_params(labelsize=FS_TICK)
ax_prof.grid(True, alpha=0.2)
ax_prof.legend(fontsize=FS_ANNOT - 1, loc="upper right")

fig.tight_layout(pad=1.5)
for out in [OUT_THESIS, OUT_ARTICLE]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
