"""
make_obs_set_mosaic.py
──────────────────────
Creates a mosaic figure showing all images in a single ISS NAC observation set.

Purpose: illustrate that an observation set consists of images taken at the same
pointing, within minutes of each other, but through different spectral filters.
This is the key concept behind CLEAR-filter subtraction for luminous band detection.

Inspired by Fig. 1 of Rubbrecht et al. (2025), which shows multiple filter images
of the same Enceladus scene.

Default seed: CLEAR frame from orbit 163EN (E17, 2012-03-27), OPUS ID
co-iss-n1711553312. This is the running example throughout the scientific article.

Layout:
  - One row of panels, one per filter in the observation set.
  - Each panel: calibrated I/F preview from OPUS, grayscale, percentile-stretched.
  - Labels: filter name + approximate wavelength, Δt from the CLEAR frame (seconds).

Usage:
    python make_obs_set_mosaic.py
    python make_obs_set_mosaic.py --seed co-iss-n1616349628 --label 106EN

Output:
    figure_scripts/output/obs_set_mosaic_{label}.png
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from pyiss import infer_set
from pyiss.opus import preview_url

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "output"

# ── ISS NAC filter metadata ───────────────────────────────────────────────────
# Approximate centre wavelengths (nm) for the science-filter component.
# Source: Porco et al. (2004), Table 1.
FILTER_CENTER_NM = {
    "UV3": 338,
    "VIO": 420,
    "BL1": 451,
    "GRN": 568,
    "RED": 617,
    "MT2": 727,
    "IR1": 752,
    "IR2": 862,
    "MT3": 890,
    "IR3": 930,
}

# ── Figure constants (Icarus double-column = 7.0 inch) ────────────────────────
FIG_W    = 7.0
DPI      = 300
FS_FILT  = 8    # filter name font size
FS_DT    = 7    # Δt label font size
FS_PANEL = 8    # panel letter font size
FS_TITLE = 8
PAD_IN   = 0.15  # gap between panels (inches)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_filter(coiss_filter: str) -> str:
    """
    Extract the science-filter component from a two-wheel COISS filter string.

    Examples:
        "CL1+CL2" -> "CLEAR"
        "IR3+CL2" -> "IR3"
        "BL1+CL2" -> "BL1"
    """
    clear_wheels = {"CL1", "CL2"}
    parts = [p.strip() for p in coiss_filter.split("+")]
    science = [p for p in parts if p not in clear_wheels]
    return "+".join(science) if science else "CLEAR"


def _filter_label(science_filter: str) -> str:
    """Build a two-line display label: filter name + wavelength info."""
    if science_filter == "CLEAR":
        return "CLEAR\n(broadband)"
    wl = FILTER_CENTER_NM.get(science_filter)
    if wl:
        return f"{science_filter}\n(~{wl} nm)"
    return science_filter


def _fetch_preview(opusid: str, *, calibrated: bool = True) -> np.ndarray:
    """
    Download an OPUS calibrated preview image and return a float32 array [0, 1].
    Falls back to uncalibrated preview on error.
    """
    try:
        url = preview_url(opusid, image_size="full", image_calibrated=calibrated)
    except Exception:
        url = preview_url(opusid, image_size="full", image_calibrated=False)

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _stretch(arr: np.ndarray, plo: float = 0.5, phi: float = 99.5) -> np.ndarray:
    """Percentile linear stretch to [0, 1]."""
    lo, hi = np.percentile(arr, [plo, phi])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


# ── Main figure builder ───────────────────────────────────────────────────────

def make_mosaic(seed: str, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"Observation-set mosaic  ->  obs_set_mosaic_{label}.png")
    print(f"Seed: {seed}")
    print(f"{'─'*60}")

    # ── Infer the full observation set ────────────────────────────────────────
    obs_set = infer_set(seed)
    df = obs_set.df.copy()
    df["time1"] = df["time1"].apply(
        lambda t: t if hasattr(t, "tzinfo")
        else __import__("pandas").to_datetime(t, utc=True)
    )
    df = df.sort_values("time1").reset_index(drop=True)
    df["science_filter"] = df["COISSfilter"].apply(_parse_filter)

    # Δt relative to the CLEAR frame (earliest frame)
    t0 = df["time1"].iloc[0]
    df["dt_s"] = (df["time1"] - t0).dt.total_seconds().round(0).astype(int)

    print(f"\nFrames in observation set ({len(df)} total):")
    for _, row in df.iterrows():
        print(f"  {row['opusid']}  {row['science_filter']:10s}  Δt={row['dt_s']:+4d} s")

    # ── Layout: one row of equal panels ───────────────────────────────────────
    n     = len(df)
    ncols = n
    nrows = 1

    if n > 6:
        ncols = (n + 1) // 2
        nrows = 2

    panel_w = (FIG_W - PAD_IN * (ncols - 1)) / ncols
    panel_h = panel_w
    fig_h   = panel_h * nrows + PAD_IN * (nrows - 1)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIG_W, fig_h),
        gridspec_kw={"wspace": PAD_IN / panel_w, "hspace": PAD_IN / panel_h},
    )
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(nrows, ncols)

    # ── Download and plot each frame ──────────────────────────────────────────
    panel_idx = 0
    for _, row in df.iterrows():
        r_idx, c_idx = divmod(panel_idx, ncols)
        ax = axes[r_idx, c_idx]

        opusid  = str(row["opusid"])
        sfilt   = row["science_filter"]
        dt_s    = int(row["dt_s"])
        flabel  = _filter_label(sfilt)
        pletter = chr(ord("a") + panel_idx)

        print(f"\n  [{pletter}] Downloading {opusid}  ({sfilt})  Δt={dt_s} s ...")
        try:
            img = _fetch_preview(opusid, calibrated=True)
            img = _stretch(img, plo=0.5, phi=99.5)
        except Exception as exc:
            print(f"      Warning: could not download ({exc}) -- blank panel")
            img = np.zeros((256, 256), dtype=np.float32)

        ax.imshow(img, cmap="gray", origin="upper",
                  interpolation="bilinear", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Panel letter -- top-left, white
        ax.text(0.04, 0.96, f"({pletter})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=FS_PANEL, color="white", fontweight="bold")

        # Filter name + wavelength -- top-right, white
        ax.text(0.96, 0.96, flabel,
                transform=ax.transAxes, va="top", ha="right",
                fontsize=FS_FILT, color="white", fontweight="bold",
                multialignment="right")

        # Δt label -- bottom-left, light grey
        dt_text = "reference" if dt_s == 0 else f"+{dt_s} s"
        ax.text(0.04, 0.04, dt_text,
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=FS_DT, color="#cccccc")

        panel_idx += 1

    # Hide unused panels
    for idx in range(panel_idx, nrows * ncols):
        r_idx, c_idx = divmod(idx, ncols)
        axes[r_idx, c_idx].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"obs_set_mosaic_{label}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--seed",
        default="co-iss-n1711553312",
        help="Seed OPUS ID for the observation set "
             "(default: CLEAR from orbit 163EN, 2012-03-27)",
    )
    p.add_argument(
        "--label",
        default="163EN",
        help="Output filename suffix (default: 163EN)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    make_mosaic(args.seed, args.label)
