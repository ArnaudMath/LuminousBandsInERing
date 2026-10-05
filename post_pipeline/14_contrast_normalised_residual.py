#!/usr/bin/env python3
"""
14_contrast_normalised_residual.py
───────────────────────────────────
For every opusid that has a saved picks.json, compute band contrast from the
smoothed multi-slice profile — the same curve the user saw when placing picks.

Method:
  1. Re-extract the multi-slice profile from the slicer parameters saved in
     picks.json (cx_px, cy_px, theta_slice_deg, n_slices, slice_spacing_px).
  2. Apply the same 101-point moving average used in notebook 13.
  3. Sample the *raw* I/F profile at each pick's t_px  (no global normalisation).

Contrast rule (all quantities in I/F residual units):
  I_max  = profile value at the highest max pick
  I_min  = mean of the two closest flanking min picks (one each side of I_max)
            — this is the local ring background estimate
  delta  = I_max − I_min  ← primary contrast metric [I/F]

  DC-offset invariance: because delta is a *difference* of profile values,
  any image-wide additive offset (caused by variable CLEAR-subtraction
  pedestal) cancels exactly. The metric is therefore directly comparable
  across all observations without additional normalisation.

  Michelson contrast (I_max−I_min)/(I_max+I_min) is intentionally avoided:
  the denominator is meaningless for residuals that can be negative, and it
  conflates contrast with the absolute brightness level.

Additional metric:
  sigma_bg = robust noise estimate from the quiet off-band profile regions
             (std of all finite profile points > BG_EXCL_PX away from any pick)
  snr      = delta / sigma_bg  [dimensionless]
             Accounts for varying per-image noise; useful for ranking
             significance of detections across exposures.

Outputs
  mnt-storage/post_pipeline/14_contrast_table.csv
  Columns: opusid, I_max, I_min, delta, sigma_bg, snr,
           n_max_picks, n_min_picks, n_flanking_mins
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_DATA_ROOT     = _PROJECT_ROOT / "mnt-storage"
_PROFILE_DIR   = _DATA_ROOT / "post_pipeline" / "13_profile_slices"
_CISSCAL_DIR   = _DATA_ROOT / "cisscal_output"
_OUT_CSV       = _DATA_ROOT / "post_pipeline" / "14_contrast_table.csv"

_SLICE_SAMPLES = 2400
_SMOOTH_WINDOW = 101

# Minimum distance in t-px from any pick to be counted as "quiet background"
# for the sigma_bg estimate.  101 px matches the smoothing kernel half-width;
# doubling it gives a conservative exclusion zone around each pick.
_BG_EXCL_PX = 202.0


# ── Residual loader ────────────────────────────────────────────────────────────
def _load_residual(opusid: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    npz_path = _CISSCAL_DIR / opusid / "residual.npz"
    if not npz_path.exists():
        return None, None
    arr  = np.load(npz_path)
    r    = np.asarray(arr["residual"], dtype=np.float64)
    mask = np.asarray(arr["mask"], dtype=bool) if "mask" in arr else np.isfinite(r)
    r    = np.nan_to_num(r, nan=0.0)
    return r, mask


# ── Profile extraction (mirrors notebook 13) ───────────────────────────────────
def _sample_line(r: np.ndarray, mask: np.ndarray,
                 cx: float, cy: float, angle_deg: float, n_samples: int):
    h, w   = r.shape
    a      = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    L      = float(math.hypot(w, h))
    t      = np.linspace(-L, L, n_samples, dtype=np.float64)
    x = cx + t * dx
    y = cy + t * dy
    inb  = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    vals = np.full_like(t, np.nan)
    if inb.any():
        idx    = np.where(inb)[0]
        raw_s  = map_coordinates(r,                        [y[inb], x[inb]], order=1, mode="nearest")
        mask_s = map_coordinates(mask.astype(np.float32),  [y[inb], x[inb]], order=0, mode="nearest") > 0.5
        vals[idx[mask_s]] = raw_s[mask_s]
    return t, vals


def _extract_profile(r: np.ndarray, mask: np.ndarray, slicer: dict) -> tuple[np.ndarray, np.ndarray]:
    cx       = float(slicer["cx_px"])
    cy       = float(slicer["cy_px"])
    angle    = float(slicer["theta_slice_deg"])
    n_slices = int(slicer.get("n_slices", 15))
    spacing  = float(slicer.get("slice_spacing_px", 8.0))

    offs = (np.arange(n_slices) - (n_slices - 1) / 2.0) * spacing
    a = math.radians(angle)
    band_px, band_py = -math.sin(a), math.cos(a)

    all_vals: list[np.ndarray] = []
    t_ref: np.ndarray | None = None
    for o in offs:
        t, vals = _sample_line(r, mask, cx + o * band_px, cy + o * band_py, angle, _SLICE_SAMPLES)
        if t_ref is None:
            t_ref = t
        all_vals.append(vals)

    stack = np.vstack(all_vals)
    cnt   = np.sum(np.isfinite(stack), axis=0)
    mean  = np.divide(np.nansum(stack, axis=0), cnt,
                      out=np.full(cnt.shape, np.nan), where=cnt > 0)

    # Moving average — same kernel as notebook 13
    w   = _SMOOTH_WINDOW
    k   = np.ones(w, dtype=np.float64)
    fin = np.isfinite(mean)
    num = np.convolve(np.where(fin, mean, 0.0), k, mode="same")
    den = np.convolve(fin.astype(np.float64),   k, mode="same")
    smoothed = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    return t_ref, smoothed  # type: ignore[return-value]


def _profile_at(t: np.ndarray, ys: np.ndarray, t_pick: float) -> float:
    valid = np.isfinite(ys)
    t_v, y_v = t[valid], ys[valid]
    return float(y_v[np.argmin(np.abs(t_v - t_pick))])


# ── Background noise estimator ─────────────────────────────────────────────────
def _estimate_bg_sigma(t: np.ndarray, ys: np.ndarray,
                       max_picks: list[dict], min_picks: list[dict]) -> float:
    """
    Estimate photometric noise from "quiet" off-band profile regions.

    A profile sample is considered quiet when it is:
      - finite, and
      - more than _BG_EXCL_PX away (in t-coordinate) from every user pick.

    Returns the standard deviation of those samples, or NaN if fewer than
    10 quiet samples are available.
    """
    all_t = np.array([p["t_px"] for p in max_picks + min_picks], dtype=np.float64)
    fin   = np.isfinite(ys)
    quiet = fin.copy()
    for tp in all_t:
        quiet &= (np.abs(t - tp) > _BG_EXCL_PX)

    vals = ys[quiet]
    if len(vals) < 10:
        return float("nan")
    return float(np.std(vals, ddof=1))


# ── Contrast ───────────────────────────────────────────────────────────────────
def _compute_contrast(t: np.ndarray, ys: np.ndarray,
                      max_picks: list[dict], min_picks: list[dict]) -> dict:
    """
    Compute band contrast directly in I/F (residual) units.

    No global normalisation is applied.  The contrast delta = I_max - I_min
    is a difference of two local profile values, so any image-wide DC offset
    cancels algebraically.  The result is cross-image comparable without
    any further rescaling.
    """
    fin = np.isfinite(ys)
    if not fin.any():
        return {"I_max": float("nan"), "I_min": float("nan"),
                "delta": float("nan"), "sigma_bg": float("nan"), "snr": float("nan"),
                "n_max_picks": len(max_picks), "n_min_picks": len(min_picks),
                "n_flanking_mins": 0}

    # Sample raw I/F profile at pick positions (no normalisation)
    max_vals = np.array([_profile_at(t, ys, p["t_px"]) for p in max_picks])
    min_vals = np.array([_profile_at(t, ys, p["t_px"]) for p in min_picks])

    best_idx = int(np.argmax(max_vals))
    I_max    = float(max_vals[best_idx])
    t_max    = float(max_picks[best_idx]["t_px"])

    # Find the closest flanking background trough on each side of I_max
    left  = [(v, p) for v, p in zip(min_vals, min_picks) if p["t_px"] < t_max]
    right = [(v, p) for v, p in zip(min_vals, min_picks) if p["t_px"] > t_max]

    candidates: list[tuple[float, dict]] = []
    if left:
        candidates.append(min(left,  key=lambda x: abs(x[1]["t_px"] - t_max)))
    if right:
        candidates.append(min(right, key=lambda x: abs(x[1]["t_px"] - t_max)))

    sigma_bg = _estimate_bg_sigma(t, ys, max_picks, min_picks)

    if not candidates:
        return {"I_max": I_max, "I_min": float("nan"),
                "delta": float("nan"), "sigma_bg": sigma_bg, "snr": float("nan"),
                "n_max_picks": len(max_picks), "n_min_picks": len(min_picks),
                "n_flanking_mins": 0}

    # Local ring background = mean of the (up to two) closest flanking troughs
    I_min = float(np.mean([v for v, _ in candidates]))
    delta = I_max - I_min
    snr   = (delta / sigma_bg) if np.isfinite(sigma_bg) and sigma_bg > 0 else float("nan")

    return {
        "I_max":           I_max,
        "I_min":           I_min,
        "delta":           delta,
        "sigma_bg":        sigma_bg,
        "snr":             snr,
        "n_max_picks":     len(max_picks),
        "n_min_picks":     len(min_picks),
        "n_flanking_mins": len(candidates),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = []

    pick_dirs = sorted(p for p in _PROFILE_DIR.iterdir() if (p / "picks.json").exists())
    print(f"[14] found {len(pick_dirs)} opusids with picks")

    for opus_dir in pick_dirs:
        opusid    = opus_dir.name
        picks     = json.loads((opus_dir / "picks.json").read_text())
        max_picks = picks["picks"].get("max") or []
        min_picks = picks["picks"].get("min") or []
        slicer    = picks.get("slicer", {})

        nan_row = {"opusid": opusid, "I_max": float("nan"), "I_min": float("nan"),
                   "delta": float("nan"), "sigma_bg": float("nan"), "snr": float("nan"),
                   "n_max_picks": len(max_picks), "n_min_picks": len(min_picks),
                   "n_flanking_mins": 0}

        if not max_picks or not min_picks:
            print(f"[14]   {opusid}: skipped (no max or min picks)")
            rows.append(nan_row)
            continue

        r, mask = _load_residual(opusid)
        if r is None:
            print(f"[14]   {opusid}: residual.npz missing — skipped")
            rows.append(nan_row)
            continue

        t, ys  = _extract_profile(r, mask, slicer)
        result = _compute_contrast(t, ys, max_picks, min_picks)
        rows.append({"opusid": opusid, **result})
        snr_str = f"{result['snr']:.1f}" if np.isfinite(result.get("snr", float("nan"))) else "n/a"
        print(f"[14]   {opusid}: I_max={result['I_max']:.5f}  I_min={result['I_min']:.5f}"
              f"  Δ={result['delta']:.5f} [I/F]  SNR={snr_str}")

    df = pd.DataFrame(rows)
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_CSV, index=False)
    print(f"\n[14] wrote {len(df)} rows → {_OUT_CSV}")
    print(df[["opusid", "I_max", "I_min", "delta", "sigma_bg", "snr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
