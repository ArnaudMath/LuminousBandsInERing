"""
make_synthetic_band_diagnostic.py
──────────────────────────────────
Two single-column figures demonstrating why the Fourier angular spectrum is
the natural space for luminous band detection:

  synthetic_band_diagnostic_A.png
      Pure sinusoidal stripe pattern.
      E(θ) shows a single razor-sharp peak at the stripe orientation.

  synthetic_band_diagnostic_B.png
      Same stripe + a solid half-circle (Enceladus stand-in) + Gaussian noise.
      E(θ) becomes noisy -- the stripe peak is overwhelmed.

All elements are purely geometrical: no gradients, no realistic rendering.
The full E(θ) is shown across [0°, 180°) with no axis regions excluded.

Usage:
    python make_synthetic_band_diagnostic.py

Output:
    figure_scripts/output/synthetic_band_diagnostic_A.png
    figure_scripts/output/synthetic_band_diagnostic_B.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "output"

# ── Image dimensions ──────────────────────────────────────────────────────────
H, W = 1024, 1024
_YY, _XX = np.indices((H, W), dtype=np.float32)
_CX, _CY = (W - 1) / 2.0, (H - 1) / 2.0
_XR = _XX - _CX
_YR = _YY - _CY

# ── Stripe parameters ─────────────────────────────────────────────────────────
# Band runs at PHI degrees from the +sample axis towards +line (clockwise as
# displayed with line 0 at the top), the convention of pipeline/08_detect_bands.py.
# E(θ) is binned by band orientation, so its peak appears at THETA_PEAK = PHI.
BAND_PHI_DEG = 45.0
BAND_PERIOD  = 180.0    # px -- gives ~5-6 stripes across the image
BAND_AMP     = 0.45     # pure sinusoid amplitude in [0, 1]

THETA_PEAK   = BAND_PHI_DEG                       # = 45°

# ── Figure layout (Icarus single column = 3.5 inch) ───────────────────────────
COL_W  = 3.5
IMG_H  = COL_W
SPEC_H = COL_W * 0.72
DPI    = 300
FS_AX  = 8
FS_LB  = 7


# ══════════════════════════════════════════════════════════════════════════════
# Scene builders
# ══════════════════════════════════════════════════════════════════════════════

def _stripe() -> np.ndarray:
    """
    Pure 2D sinusoidal stripe running at BAND_PHI_DEG degrees.
    The Fourier peak in E(θ) lands at θ = BAND_PHI_DEG = THETA_PEAK.
    """
    nx = np.cos(np.radians(BAND_PHI_DEG + 90.0))
    ny = np.sin(np.radians(BAND_PHI_DEG + 90.0))
    u  = nx * _XR + ny * _YR
    return (0.5 + BAND_AMP * np.cos(2.0 * np.pi * u / BAND_PERIOD)).astype(np.float32)


def scene_A() -> np.ndarray:
    """Stripe only -- completely clean."""
    return _stripe()


def scene_B(seed: int = 7) -> np.ndarray:
    """
    Stripe + solid half-disc at the bottom edge + Gaussian noise.
    The disc and noise distribute power across all angles in E(θ),
    burying the stripe peak under a noisy floor.
    """
    img = _stripe().copy()

    # Solid half-disc centred at the bottom edge (no gradient)
    cx = float(W) / 2.0
    cy = float(H) - 1.0
    r  = float(H) * 0.30
    dist = np.sqrt((_XX - cx) ** 2 + (_YY - cy) ** 2)
    img[dist <= r] = 0.90   # flat bright fill -- purely geometric

    # Gaussian noise strong enough to bury the stripe peak in E(θ)
    rng = np.random.default_rng(seed)
    img = np.clip(
        img + rng.normal(0.0, 0.35, (H, W)).astype(np.float32),
        0.0, 1.0,
    )
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Angular power spectrum  (full [0°, 180°), no axis exclusions)
# ══════════════════════════════════════════════════════════════════════════════

def angular_spectrum(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute E(θ) from the mean-subtracted image.
    No preprocessing, no axis exclusion -- full spectrum shown.
    """
    m = img.astype(np.float64) - img.mean()
    F = np.fft.fftshift(np.fft.fft2(m))
    mag = np.abs(F)

    kx = np.fft.fftshift(np.fft.fftfreq(W)).astype(np.float32)
    ky = np.fft.fftshift(np.fft.fftfreq(H)).astype(np.float32)
    KX, KY = np.meshgrid(kx, ky)
    R  = np.sqrt(KX ** 2 + KY ** 2)
    # Band orientation = wavevector angle + 90°, same axes as the pipeline
    THETA = (np.degrees(np.arctan2(KY, KX)) + 90.0) % 180.0

    active = R > 0.004   # exclude only the very DC centre

    bins = np.arange(0, 181, 1, dtype=np.float64)
    E, edges = np.histogram(
        THETA[active].ravel(),
        bins=bins,
        weights=(mag[active] ** 2).ravel(),
    )
    theta_centres = (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)
    return theta_centres, E.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Column figure builder
# ══════════════════════════════════════════════════════════════════════════════

def _save_column(
    img: np.ndarray,
    theta: np.ndarray,
    E: np.ndarray,
    label_img: str,
    label_spc: str,
    filename: str,
) -> None:
    fig_h = IMG_H + SPEC_H + 0.06
    fig, (ax_img, ax_spc) = plt.subplots(
        2, 1,
        figsize=(COL_W, fig_h),
        gridspec_kw={"hspace": 0.14, "height_ratios": [IMG_H, SPEC_H]},
    )
    fig.patch.set_facecolor("white")

    # ── Image panel ───────────────────────────────────────────────────────────
    ax_img.imshow(img, cmap="gray", origin="upper",
                  interpolation="nearest", vmin=0, vmax=1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_visible(False)
    ax_img.text(0.03, 0.97, f"({label_img})",
                transform=ax_img.transAxes, va="top", ha="left",
                fontsize=FS_AX, color="white", fontweight="bold")

    # ── Spectrum panel ─────────────────────────────────────────────────────────
    E_norm = E / (E.max() + 1e-12)
    ax_spc.plot(theta, E_norm, color="#2196F3", lw=1.2)
    ax_spc.axvline(THETA_PEAK, color="#F44336", lw=1.2, ls="--")
    ax_spc.text(
        THETA_PEAK + 2, 0.90,
        fr"$\theta^\star = {THETA_PEAK:.0f}°$",
        fontsize=FS_LB, color="#F44336", va="top",
    )
    ax_spc.set_xlim(0, 180)
    ax_spc.set_ylim(0, 1.08)
    ax_spc.set_xticks([0, 45, 90, 135, 180])
    ax_spc.set_xlabel(r"Band orientation $\theta$ (deg)", fontsize=FS_AX)
    ax_spc.set_ylabel(r"$E(\theta)$ (normalised)", fontsize=FS_AX)
    ax_spc.tick_params(labelsize=FS_LB)
    ax_spc.grid(True, alpha=0.20)
    ax_spc.text(0.03, 0.97, f"({label_spc})",
                transform=ax_spc.transAxes, va="top", ha="left",
                fontsize=FS_AX, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def make_figure() -> None:
    print("Scene A: stripe only ...")
    img_A = scene_A()
    theta, E_A = angular_spectrum(img_A)
    _save_column(img_A, theta, E_A,
                 label_img="a", label_spc="b",
                 filename="synthetic_band_diagnostic_A.png")

    print("Scene B: stripe + half-disc + noise ...")
    img_B = scene_B()
    _, E_B = angular_spectrum(img_B)
    _save_column(img_B, theta, E_B,
                 label_img="c", label_spc="d",
                 filename="synthetic_band_diagnostic_B.png")


if __name__ == "__main__":
    make_figure()
