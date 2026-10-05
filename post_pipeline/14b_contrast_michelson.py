#!/usr/bin/env python3
"""
14b_contrast_michelson.py
─────────────────────────
Copy of 14_contrast_normalised_residual.py that adds a normalised
(Michelson-type) contrast. 14 and its table are left untouched.

Method (profile extraction and picks identical to 14):
  1. Re-extract the multi-slice profile from the slicer parameters saved in
     picks.json, on the CLEAR-subtracted residual (as in 14) AND, with the
     same mask, slices and smoothing, on the calibrated science-filter I/F
     image (_CALIB.IMG).
  2. Sample both profiles at the user's picks.

Contrast rule:
  delta  = r_max − r_min           residual profile, exactly as in 14 [I/F]
  S_max, S_min                     science-filter I/F at the same picks
  C      = delta / (S_max + S_min) normalised contrast [dimensionless]

  Why the numerator comes from the residual: on the raw science image the
  peak-to-trough difference is dominated by the smooth E ring brightness
  gradient across the profile (28/56 bands come out negative, and the sign
  flips under small pick shifts). CLEAR subtraction removes that achromatic
  gradient, so the residual difference isolates the band. C is therefore the
  band's chromatic excess in filter X relative to CLEAR, divided by the local
  mean brightness in filter X.

  Why the denominator comes from the science image: these are real I/F
  values, always positive, unlike the residual which can be negative
  (Seignovert BS142).

Outputs
  post_pipeline/14b_contrast_table.csv
  Columns: opusid, I_max, I_min, delta, sigma_bg, snr, S_max, S_min, C,
           n_max_picks, n_min_picks, n_flanking_mins
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_PROFILE_DIR   = _PROJECT_ROOT / "post_pipeline" / "13_profile_slices"
_CISSCAL_DIR   = Path("/mnt/storage/cisscal_output")      # same residuals as 14
_RAW_DIRS      = [Path("/mnt/storage/raw_images"),
                  Path("/mnt/storage/pipeline_output/surveys/survey_B/raw_images"),
                  Path("/mnt/storage/pipeline_output/surveys/survey_C/raw_images")]
_OUT_CSV       = _PROJECT_ROOT / "post_pipeline" / "14b_contrast_table.csv"

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


def _load_science(opusid: str) -> np.ndarray | None:
    """Calibrated science-filter I/F (VICAR, same reader as pipeline step 05)."""
    for d in _RAW_DIRS:
        p = d / f"{opusid}_CALIB.IMG"
        if not p.exists():
            continue
        head = p.open("rb").read(512).decode("ascii", errors="replace")

        def _vget(key: str, default: int) -> int:
            m = re.search(rf"\b{key}=(\d+)", head)
            return int(m.group(1)) if m else default

        offset = _vget("LBLSIZE", 4096) + _vget("NLB", 0) * _vget("RECSIZE", 4096)
        nl, ns = _vget("NL", 1024), _vget("NS", 1024)
        arr = np.fromfile(str(p), dtype="<f4", count=nl * ns, offset=offset)
        return arr.reshape(nl, ns).astype(np.float64)
    return None


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
                      max_picks: list[dict], min_picks: list[dict],
                      ys_sci: np.ndarray) -> dict:
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
                "S_max": float("nan"), "S_min": float("nan"), "C": float("nan"),
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
                "S_max": float("nan"), "S_min": float("nan"), "C": float("nan"),
                "n_max_picks": len(max_picks), "n_min_picks": len(min_picks),
                "n_flanking_mins": 0}

    # Local ring background = mean of the (up to two) closest flanking troughs
    I_min = float(np.mean([v for v, _ in candidates]))
    delta = I_max - I_min
    snr   = (delta / sigma_bg) if np.isfinite(sigma_bg) and sigma_bg > 0 else float("nan")

    # Science-filter I/F at the same picks: max pick and the same flanking mins
    S_max = _profile_at(t, ys_sci, t_max)
    S_min = float(np.mean([_profile_at(t, ys_sci, p["t_px"]) for _, p in candidates]))
    C     = delta / (S_max + S_min) if (S_max + S_min) > 0 else float("nan")

    return {
        "I_max":           I_max,
        "I_min":           I_min,
        "delta":           delta,
        "sigma_bg":        sigma_bg,
        "snr":             snr,
        "S_max":           S_max,
        "S_min":           S_min,
        "C":               C,
        "n_max_picks":     len(max_picks),
        "n_min_picks":     len(min_picks),
        "n_flanking_mins": len(candidates),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = []

    pick_dirs = sorted(p for p in _PROFILE_DIR.iterdir() if (p / "picks.json").exists())
    print(f"[14b] found {len(pick_dirs)} opusids with picks")

    for opus_dir in pick_dirs:
        opusid    = opus_dir.name
        picks     = json.loads((opus_dir / "picks.json").read_text())
        max_picks = picks["picks"].get("max") or []
        min_picks = picks["picks"].get("min") or []
        slicer    = picks.get("slicer", {})

        nan_row = {"opusid": opusid, "I_max": float("nan"), "I_min": float("nan"),
                   "delta": float("nan"), "sigma_bg": float("nan"), "snr": float("nan"),
                   "S_max": float("nan"), "S_min": float("nan"), "C": float("nan"),
                   "n_max_picks": len(max_picks), "n_min_picks": len(min_picks),
                   "n_flanking_mins": 0}

        if not max_picks or not min_picks:
            print(f"[14b]   {opusid}: skipped (no max or min picks)")
            rows.append(nan_row)
            continue

        r, mask = _load_residual(opusid)
        if r is None:
            print(f"[14b]   {opusid}: residual.npz missing — skipped")
            rows.append(nan_row)
            continue

        sci = _load_science(opusid)
        if sci is None:
            print(f"[14b]   {opusid}: science _CALIB.IMG missing — skipped")
            rows.append(nan_row)
            continue

        t, ys  = _extract_profile(r, mask, slicer)
        _, ys_sci = _extract_profile(sci, mask, slicer)
        result = _compute_contrast(t, ys, max_picks, min_picks, ys_sci)
        rows.append({"opusid": opusid, **result})
        snr_str = f"{result['snr']:.1f}" if np.isfinite(result.get("snr", float("nan"))) else "n/a"
        print(f"[14b]   {opusid}: I_max={result['I_max']:.5f}  I_min={result['I_min']:.5f}"
              f"  Δ={result['delta']:.5f} [I/F]  C={result['C']:.4f}  SNR={snr_str}")

    df = pd.DataFrame(rows)
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_CSV, index=False)
    print(f"\n[14b] wrote {len(df)} rows → {_OUT_CSV}")
    print(df[["opusid", "delta", "S_max", "S_min", "C", "snr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
