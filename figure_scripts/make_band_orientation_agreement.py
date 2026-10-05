"""
make_band_orientation_agreement.py
──────────────────────────────────
Replacement for the deviation-angle histogram (figs/band_delta_phi_hist.png).

Single panel: measured band position angle against the predicted direction, both
in the image plane, for all 62 positive detections, with the 1:1 line. The
predicted direction is the projected intersection of the scattering plane with
the ring plane. The deviation distribution and the in-plane-perpendicular
control are reported numerically in the text rather than drawn.

Reads figure_scripts/output/wp2_image_plane_alignment.csv.
Writes figure_scripts/output/band_orientation_agreement.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "output"
CSV     = OUT_DIR / "wp2_image_plane_alignment.csv"

C_PRE, C_POST = "#d1495b", "#2e6f95"


def nearest_rep(obs: float, ref: float) -> float:
    """Representative of the axial angle obs (mod 180) closest to ref."""
    best, bestd = obs, abs(obs - ref)
    for cand in (obs - 180.0, obs + 180.0):
        if abs(cand - ref) < bestd:
            best, bestd = cand, abs(cand - ref)
    return best


def main() -> int:
    df = pd.read_csv(CSV)
    df["obs_plot"] = [nearest_rep(o, p) for o, p in zip(df.th_obs, df.th_gvec_img)]

    fig, ax1 = plt.subplots(figsize=(5.9, 5.6))

    lo, hi = 0.0, 180.0
    ax1.plot([lo, hi], [lo, hi], color="0.35", lw=1.0, ls="--", zorder=1,
             label="1:1")
    for grp, colour, lab in [("pre-equinox", C_PRE, "pre-equinox (n=%d)"),
                             ("post-equinox", C_POST, "post-equinox (n=%d)")]:
        sub = df[df.group == grp]
        ax1.scatter(sub.th_gvec_img, sub.obs_plot, s=34, facecolor=colour,
                    edgecolor="white", linewidth=0.6, zorder=3,
                    label=lab % len(sub))
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xticks(range(0, 181, 30))
    ax1.set_yticks(range(0, 181, 30))
    ax1.set_xlabel("predicted band position angle in the image (deg)")
    ax1.set_ylabel("measured band position angle (deg)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    out = OUT_DIR / "band_orientation_agreement.png"
    fig.savefig(out, dpi=200)
    print(f"[out] {out}")
    print(f"  n={len(df)}  median dev {df.d_to_gvec.median():.2f}  "
          f"mean {df.d_to_gvec.mean():.2f}  max {df.d_to_gvec.max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
