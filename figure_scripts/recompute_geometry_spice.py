"""
recompute_geometry_spice.py
────────────────────────────
Recomputes the Sun-relative Enceladus orbital longitude for every survey
pair using per-observation SPICE calls, then regenerates Figures 14, 15,
and 18 with correct ansa assignments.

The key fix: the geometry JSON uses a single median-epoch rotation (Sun at
+y at ~2010), which introduces up to ±30° error for observations far from
that epoch.  This script computes the Sun direction at the actual
observation epoch for each pair, giving the correct Sun-relative longitude.

Saves to figure_scripts/output/ only — do not push to article figs until
the output has been reviewed.

Usage:
    cd /home/arnaudm/projects/LuminousBandsInERing
    python figure_scripts/recompute_geometry_spice.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Wedge

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
GEO_JSON    = _ROOT / "post_pipeline" / "12_selected_bundle_A_geometry_overview.json"
OBS_SEQ     = _ROOT / "pipeline_output" / "obs_sequences.csv"
METAKERNEL  = _ROOT / "spice_kernels_staged" / "cassini_iss_pipeline.tm"
OUT_DIR     = Path(__file__).parent / "output"

RS_KM      = 60268.0
CASSINI_ID = -82

_MOON_ORBITS = {
    "F": 140220 / RS_KM,
    "G": 167500 / RS_KM,
    "E": 237948 / RS_KM,
    "T": 294619 / RS_KM,
    "D": 377396 / RS_KM,
    "R": 527108 / RS_KM,
}


# ── SPICE helpers ──────────────────────────────────────────────────────────────

def load_spice() -> None:
    import spiceypy as spice
    spice.kclear()
    os.chdir(METAKERNEL.parent)
    spice.furnsh(str(METAKERNEL))
    for extra in ["spk/200128RU_SCPSE_10360_11041.bsp",
                  "spk/200128RU_SCPSE_12016_12060.bsp"]:
        if Path(extra).exists():
            spice.furnsh(extra)
    print(f"[SPICE] loaded {METAKERNEL.name} (+supplementary SPKs)")


def et_from_image_id(spice, image_id: str) -> float | None:
    sclk = "1/" + image_id[1:]
    try:
        return float(spice.scs2e(CASSINI_ID, sclk))
    except Exception:
        return None


def sun_relative_angle(spice, et: float) -> float | None:
    """
    Angle of Enceladus relative to the Sun, in Saturn's equatorial plane.
    Returns angle in [0, 2π) such that Sun is at π/2 (top in the plot).
    """
    try:
        enc, _ = spice.spkpos("ENCELADUS", et, "IAU_SATURN", "LT+S", "SATURN")
        sun, _ = spice.spkpos("SUN",       et, "IAU_SATURN", "LT+S", "SATURN")
    except Exception:
        return None

    phi_enc = math.atan2(float(enc[1]), float(enc[0]))
    phi_sun = math.atan2(float(sun[1]), float(sun[0]))

    # Rotate so Sun is at π/2 (top = +y)
    theta = phi_enc - phi_sun + math.pi / 2.0
    return theta % (2 * math.pi)


def cassini_sun_relative_xy(spice, et: float, phi_sun: float) -> tuple[float, float] | None:
    """Cassini position in Saturn radii, rotated so Sun is at top."""
    try:
        pos, _ = spice.spkpos("CASSINI", et, "IAU_SATURN", "NONE", "SATURN")
    except Exception:
        return None
    x = float(pos[0]) / RS_KM
    y = float(pos[1]) / RS_KM
    # Rotate by -(phi_sun - π/2) so Sun is at top
    rot = math.pi / 2.0 - phi_sun
    xr = x * math.cos(rot) - y * math.sin(rot)
    yr = x * math.sin(rot) + y * math.cos(rot)
    return xr, yr


def enc_sun_relative_xy(spice, et: float, phi_sun: float) -> tuple[float, float] | None:
    """Enceladus position in Saturn radii, rotated so Sun is at top."""
    try:
        pos, _ = spice.spkpos("ENCELADUS", et, "IAU_SATURN", "LT+S", "SATURN")
        sun, _ = spice.spkpos("SUN",       et, "IAU_SATURN", "LT+S", "SATURN")
    except Exception:
        return None
    phi_sun_actual = math.atan2(float(sun[1]), float(sun[0]))
    x = float(pos[0]) / RS_KM
    y = float(pos[1]) / RS_KM
    rot = math.pi / 2.0 - phi_sun_actual
    xr = x * math.cos(rot) - y * math.sin(rot)
    yr = x * math.sin(rot) + y * math.cos(rot)
    return xr, yr


def phi_sun_at(spice, et: float) -> float | None:
    try:
        sun, _ = spice.spkpos("SUN", et, "IAU_SATURN", "LT+S", "SATURN")
        return math.atan2(float(sun[1]), float(sun[0]))
    except Exception:
        return None


# ── Saturn background plate ────────────────────────────────────────────────────

def _draw_backplate(ax, r_max: float, fontsize: int = 11) -> None:
    ax.set_facecolor("#eaf4f7")
    ax.add_patch(Circle((0, 0), 1.0, facecolor="white", edgecolor="grey", lw=1.5, zorder=3))
    ax.add_patch(Wedge((0, 0), 1.0, 180, 360, facecolor="#cfcfcf", edgecolor="none", zorder=3))
    for lbl, r in _MOON_ORBITS.items():
        is_e = lbl == "E"
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="grey",
                             lw=1.4 if is_e else 1.0,
                             linestyle="-" if is_e else "--",
                             alpha=0.9 if is_e else 0.6, zorder=2))
        ax.text(r + 0.15, 0.0, lbl, fontsize=fontsize - 1, color="grey", va="center")
    sun_r = _MOON_ORBITS["R"] + 0.35
    ax.text(0, sun_r, "Sun ☉", ha="center", va="center", fontsize=fontsize, color="grey")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel("Distance [Saturn radii]", fontsize=fontsize)
    ax.set_ylabel("Distance [Saturn radii]", fontsize=fontsize)
    ax.grid(False)


# ── Ansa labels and separator ──────────────────────────────────────────────────

def _draw_ansa_labels(ax, enc_r: float, fontsize: int = 10) -> None:
    sep_kw = dict(color="#777777", lw=1.3, ls=(0, (5, 4)), alpha=0.85, zorder=4)
    r_max = ax.get_xlim()[1]
    ax.plot([-r_max, r_max], [-r_max,  r_max], **sep_kw)
    ax.plot([-r_max, r_max], [ r_max, -r_max], **sep_kw)
    ql = enc_r + 0.65
    kw = dict(fontsize=fontsize, color="#333333", fontstyle="italic", fontweight="bold",
              ha="center", va="center", zorder=12,
              path_effects=[pe.withStroke(linewidth=3, foreground="#eaf4f7")])
    ax.text( 0,       ql + 0.3,  "Sub-solar ansa",  **kw)
    ax.text( 0,      -ql - 0.3,  "Anti-solar ansa", **kw)
    # x-axis labels sit further out so they clear the Enceladus-orbit markers
    ax.text(-(enc_r + 2.0), 0,   "Evening ansa",    **kw)   # -x = evening
    ax.text( (enc_r + 2.0), 0,   "Morning ansa",    **kw)   # +x = morning


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import spiceypy as spice
    load_spice()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    geo_data = json.loads(GEO_JSON.read_text())
    obs_df   = pd.read_csv(OBS_SEQ)
    imgid_to_flyby = dict(zip(obs_df["image_id"].astype(str),
                              obs_df["obs_sequence"].astype(str)))

    # ── Compute per-observation Sun-relative angles for all 624 pairs ──────────
    print("\n[1/3] Computing Sun-relative Enceladus angles for all pairs...")
    all_pairs = geo_data["all_pairs_records"]
    n_ok = 0
    for ap in all_pairs:
        et = et_from_image_id(spice, str(ap["image_id"]))
        if et is None:
            ap["theta_sun"] = None
            ap["enc_xr"] = ap["enc_yr"] = None
            continue
        theta = sun_relative_angle(spice, et)
        if theta is None:
            ap["theta_sun"] = None
            ap["enc_xr"] = ap["enc_yr"] = None
            continue
        ap["theta_sun"] = theta
        phi_sun = phi_sun_at(spice, et)
        xy = enc_sun_relative_xy(spice, et, phi_sun) if phi_sun is not None else None
        ap["enc_xr"], ap["enc_yr"] = (xy if xy else (None, None))
        ap["flyby"] = imgid_to_flyby.get(str(ap["image_id"]), "?")
        n_ok += 1

    print(f"   {n_ok}/{len(all_pairs)} pairs computed")

    valid_pairs = [ap for ap in all_pairs if ap.get("theta_sun") is not None]
    theta_all   = np.array([ap["theta_sun"] for ap in valid_pairs])
    theta_deg   = np.degrees(theta_all) % 360

    # ── Positive and doubtful detections ──────────────────────────────────────
    print("\n[2/3] Computing positions for positive/doubtful detections...")
    det_records = []
    for rec in geo_data["records"]:
        if not all(k in rec for k in ("obs_x", "obs_y", "enc_x", "enc_y")):
            continue
        et = et_from_image_id(spice, str(rec.get("image_id", "")))
        if et is None:
            det_records.append({**rec, "enc_xr": rec["enc_x"],
                                 "obs_xr": rec["obs_x"], "obs_yr": rec["obs_y"],
                                 "enc_yr": rec["enc_y"]})
            continue
        phi_sun = phi_sun_at(spice, et)
        if phi_sun is None:
            det_records.append({**rec, "enc_xr": rec["enc_x"],
                                 "obs_xr": rec["obs_x"], "obs_yr": rec["obs_y"],
                                 "enc_yr": rec["enc_y"]})
            continue
        rot = math.pi / 2.0 - phi_sun
        def _rot(px, py):
            return (px * math.cos(rot) - py * math.sin(rot),
                    px * math.sin(rot) + py * math.cos(rot))
        # Try Cassini from SPICE, fall back to JSON
        cas_xy = cassini_sun_relative_xy(spice, et, phi_sun)
        if cas_xy is None:
            cas_xy = _rot(rec["obs_x"], rec["obs_y"])
        enc_xy = enc_sun_relative_xy(spice, et, phi_sun)
        if enc_xy is None:
            enc_xy = _rot(rec["enc_x"], rec["enc_y"])
        det_records.append({**rec, "obs_xr": cas_xy[0], "obs_yr": cas_xy[1],
                             "enc_xr": enc_xy[0], "enc_yr": enc_xy[1]})

    pos_recs = [r for r in det_records if r.get("label") == "positive"]
    dbt_recs = [r for r in det_records if r.get("label") == "doubtful"]
    print(f"   {len(pos_recs)} positive, {len(dbt_recs)} doubtful")

    # ════════════════════════════════════════════════════════════════════════
    # Figure 14: geometry_positives
    # ════════════════════════════════════════════════════════════════════════
    print("\n[3/3] Generating figures...")

    COL_LINE    = "#607d8b"
    COL_CASSINI = "#7b1fa2"
    COL_POS     = "dodgerblue"
    COL_DOUBTFUL = "#ff9800"

    all_x = [r["obs_xr"] for r in det_records] + [r["enc_xr"] for r in det_records]
    all_y = [r["obs_yr"] for r in det_records] + [r["enc_yr"] for r in det_records]
    r_max14 = max(max(abs(v) for v in all_x), max(abs(v) for v in all_y)) * 1.08
    r_max14 = max(r_max14, 9.5)

    fig14, ax14 = plt.subplots(figsize=(3.46, 3.2), dpi=300)
    fig14.patch.set_facecolor("white")
    _draw_backplate(ax14, r_max14, fontsize=7)

    for rec in dbt_recs:
        ax14.plot([rec["obs_xr"], rec["enc_xr"]], [rec["obs_yr"], rec["enc_yr"]],
                  color=COL_LINE, lw=0.8, alpha=0.14, zorder=5)
        ax14.scatter([rec["obs_xr"]], [rec["obs_yr"]], s=8, c=COL_DOUBTFUL,
                     edgecolors="white", linewidths=0.2, alpha=0.5, zorder=7)
        ax14.scatter([rec["enc_xr"]], [rec["enc_yr"]], s=12, c=COL_DOUBTFUL,
                     edgecolors="white", linewidths=0.2, alpha=0.8, zorder=8)

    for rec in pos_recs:
        ax14.plot([rec["obs_xr"], rec["enc_xr"]], [rec["obs_yr"], rec["enc_yr"]],
                  color=COL_LINE, lw=0.8, alpha=0.18, zorder=9)
        ax14.scatter([rec["obs_xr"]], [rec["obs_yr"]], s=8, c=COL_CASSINI,
                     edgecolors="white", linewidths=0.2, alpha=0.6, zorder=10)
        ax14.scatter([rec["enc_xr"]], [rec["enc_yr"]], s=12, c=COL_POS,
                     edgecolors="white", linewidths=0.2, alpha=0.9, zorder=11)

    handles = [
        plt.Line2D([], [], color=COL_LINE, lw=1.0, alpha=0.5, label="Line of sight"),
        plt.scatter([], [], s=8,  c=COL_CASSINI,  edgecolors="white", linewidths=0.2, label="Cassini (positive)"),
        plt.scatter([], [], s=12, c=COL_POS,       edgecolors="white", linewidths=0.2, label="Enceladus (positive)"),
        plt.scatter([], [], s=8,  c=COL_DOUBTFUL,  edgecolors="white", linewidths=0.2, label="Cassini (doubtful)"),
        plt.scatter([], [], s=12, c=COL_DOUBTFUL,  edgecolors="white", linewidths=0.2, label="Enceladus (doubtful)"),
    ]
    ax14.legend(handles=handles, loc="upper right", fontsize=5.5,
                frameon=True, framealpha=0.85, edgecolor="none")
    fig14.tight_layout(pad=0.5)
    out14 = OUT_DIR / "geometry_positives_spice.png"
    fig14.savefig(out14, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig14)
    print(f"   Saved: {out14}")

    # ════════════════════════════════════════════════════════════════════════
    # Figure 15: ansa coverage histogram
    # ════════════════════════════════════════════════════════════════════════
    START = 45
    def _shift(ang):
        a = np.asarray(ang, dtype=float) % 360
        return np.where(a < START, a + 360, a)

    pos_thetas = np.array([
        math.degrees(ap["theta_sun"]) % 360
        for ap in valid_pairs
        if imgid_to_flyby.get(str(ap.get("image_id", "")), "?") != "?"
        or True   # include all
    ])
    # Positive detection angles
    pos_det_thetas = []
    for rec in pos_recs:
        et = et_from_image_id(spice, str(rec.get("image_id", "")))
        if et is not None:
            th = sun_relative_angle(spice, et)
            if th is not None:
                pos_det_thetas.append(math.degrees(th) % 360)
    pos_det_thetas = np.array(pos_det_thetas)

    all_shifted = _shift(theta_deg)
    pos_shifted = _shift(pos_det_thetas)

    BIN_DEG = 5
    bins = np.arange(START, START + 361, BIN_DEG)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    all_counts, _ = np.histogram(all_shifted, bins=bins)
    pos_counts, _ = np.histogram(pos_shifted, bins=bins)

    ANSA_BOUNDS  = [45, 135, 225, 315, 405]
    ANSA_CENTERS = [
        ( 90, "Sub-solar",    "#e3f2fd", 0.18),
        (180, "Evening ansa", "#f3e5f5", 0.18),
        (270, "Anti-solar",   "#e8f5e9", 0.18),
        (360, "Morning ansa", "#ffebee", 0.30),
    ]

    fig15, ax15 = plt.subplots(figsize=(7.0, 2.6))
    fig15.patch.set_facecolor("white")

    for i, (cx, label, color, alpha) in enumerate(ANSA_CENTERS):
        ax15.axvspan(ANSA_BOUNDS[i], ANSA_BOUNDS[i + 1], color=color, alpha=alpha, zorder=0)

    ax15.bar(bin_centres, all_counts, width=BIN_DEG * 0.95,
             color="#b0bec5", edgecolor="none", alpha=0.85,
             label=f"All processed pairs (n={len(valid_pairs)})")
    ax15.bar(bin_centres, pos_counts, width=BIN_DEG * 0.95,
             color="#1565C0", edgecolor="none", alpha=0.9,
             label=f"Positive detections (n={len(pos_det_thetas)})")

    for b in ANSA_BOUNDS[1:-1]:
        ax15.axvline(b, color="#555555", lw=0.7, ls="--", zorder=5, alpha=0.6)
    ax15.axvline(90, color="#ff8f00", lw=1.0, ls=":", zorder=6, alpha=0.8)

    ax15.set_xlim(START, START + 360)
    ax15.set_ylim(0, None)
    ax15.set_xticks(ANSA_BOUNDS)
    ax15.set_xticklabels(["45°", "135°", "225°", "315°", "45°"], fontsize=6)
    ax15.set_xlabel("Enceladus orbital longitude (Sun at 90°)", fontsize=7)
    ax15.set_ylabel("Image pairs", fontsize=7)
    ax15.tick_params(labelsize=6, length=2)
    ax15.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    ax15.yaxis.get_major_locator().set_params(integer=True)
    ax15.legend(fontsize=6, loc="upper right", framealpha=0.9)
    fig15.tight_layout(pad=0.6)

    y_top = ax15.get_ylim()[1]
    for cx, label, color, alpha in ANSA_CENTERS:
        ax15.text(cx, 1.01, label, ha="center", va="bottom", fontsize=6,
                  color="#333333", fontstyle="italic",
                  transform=ax15.get_xaxis_transform(), clip_on=False)

    out15 = OUT_DIR / "ansa_coverage_histogram_spice.png"
    fig15.savefig(out15, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig15)
    print(f"   Saved: {out15}")

    # ════════════════════════════════════════════════════════════════════════
    # Figure 18: all 23 flybys ansa geometry
    # ════════════════════════════════════════════════════════════════════════
    flyby_order = sorted(obs_df["obs_sequence"].dropna().unique())
    n_fb = len(flyby_order)
    colors23 = [plt.cm.tab20(i) for i in range(20)] + [plt.cm.tab20b(i) for i in range(3)]
    flyby_col = {fb: colors23[i] for i, fb in enumerate(flyby_order)}
    flyby_col["?"] = (0.55, 0.55, 0.55, 1.0)

    enc_xr_all = [ap["enc_xr"] for ap in valid_pairs if ap.get("enc_xr") is not None]
    enc_yr_all = [ap["enc_yr"] for ap in valid_pairs if ap.get("enc_yr") is not None]
    r_max18 = max(max(abs(v) for v in enc_xr_all), max(abs(v) for v in enc_yr_all)) * 1.08
    r_max18 = max(r_max18, 9.5)

    fig18, ax18 = plt.subplots(figsize=(13, 13), dpi=150)
    fig18.patch.set_facecolor("white")
    _draw_backplate(ax18, r_max18, fontsize=11)
    _draw_ansa_labels(ax18, _MOON_ORBITS["E"], fontsize=10)

    for ap in valid_pairs:
        if ap.get("enc_xr") is None:
            continue
        col = flyby_col.get(ap.get("flyby", "?"), flyby_col["?"])
        ax18.scatter([ap["enc_xr"]], [ap["enc_yr"]], s=18, color=col,
                     edgecolors="white", linewidths=0.35, zorder=9, alpha=0.88)

    # Flyby labels at outermost point of each cluster
    n_unknown = sum(1 for ap in valid_pairs if ap.get("flyby", "?") == "?")
    for fb in flyby_order + (["?"] if n_unknown else []):
        pts = np.array([(ap["enc_xr"], ap["enc_yr"])
                        for ap in valid_pairs
                        if ap.get("flyby", "?") == fb and ap.get("enc_xr") is not None])
        if len(pts) == 0:
            continue
        radii = np.linalg.norm(pts, axis=1)
        mx, my = pts[np.argmax(radii)]
        norm = math.hypot(mx, my)
        ox = mx + 0.4 * mx / norm if norm > 0 else mx
        oy = my + 0.4 * my / norm if norm > 0 else my
        lbl = f"unassigned ({n_unknown})" if fb == "?" else fb
        ax18.text(ox, oy, lbl, fontsize=8, color=flyby_col[fb], fontweight="bold",
                  ha="center", va="center", zorder=13,
                  path_effects=[pe.withStroke(linewidth=2, foreground="#eaf4f7")])

    ax18.set_title(
        f"Viewing geometry — {len(valid_pairs)} pairs across all {n_fb} flyby sequences\n"
        "Enceladus positions colour-coded by ISS flyby  "
        "(Sun at top; diagonal dotted lines separate ansae)",
        fontsize=12, pad=12,
    )
    fig18.tight_layout()
    out18 = OUT_DIR / "all_23flybys_in_their_ansa_spice.png"
    fig18.savefig(out18, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig18)
    print(f"   Saved: {out18}")

    print("\nDone. Check figure_scripts/output/ before pushing to article figs.")


if __name__ == "__main__":
    main()
