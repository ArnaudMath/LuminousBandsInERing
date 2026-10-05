#!/usr/bin/env python3
"""
12_selected_bundle_A_geometry_overview.py
────────────────────────────────────────
Create post-review figures from the labels saved by post_pipeline/11_review_bundle_A.py.

Outputs:
  1) Geometry plot (positive only)
  2) Monthly timeline histogram (positive only)
  3) Geometry plot with doubtful overlays (positive + doubtful)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "mnt-storage"
_DEFAULT_SELECTION_JSON = _DEFAULT_DATA_ROOT / "post_pipeline" / "11_bundle_A_review_labels.json"
_DEFAULT_OUT_PNG = _DEFAULT_DATA_ROOT / "post_pipeline" / "12_selected_bundle_A_geometry_overview.png"
_DEFAULT_OUT_TIMELINE_PNG = _DEFAULT_DATA_ROOT / "post_pipeline" / "12_selected_bundle_A_timeline.png"
_DEFAULT_OUT_WITH_DOUBTFUL_PNG = _DEFAULT_DATA_ROOT / "post_pipeline" / "12_selected_bundle_A_geometry_with_doubtful.png"
_DEFAULT_OUT_ALL_PNG = _DEFAULT_DATA_ROOT / "post_pipeline" / "12_selected_bundle_A_geometry_all_pairs.png"
_DEFAULT_OUT_JSON = _DEFAULT_DATA_ROOT / "post_pipeline" / "12_selected_bundle_A_geometry_overview.json"


def _make_paths(root: Path) -> None:
    global KERNEL_ARCHIVE, KERNELS_DIR, METAKERNEL, METAKERNELS
    KERNEL_ARCHIVE = root / "cosp_1000_070930_120502"
    KERNELS_DIR = KERNEL_ARCHIVE / "cosp_1000" / "data"
    METAKERNEL = KERNEL_ARCHIVE / "cas_2012_v15_070930_120502.tm"
    METAKERNELS = sorted(KERNEL_ARCHIVE.glob("cas_*_070930_120502.tm"))


def _load_spice_kernels() -> bool:
    import re
    import spiceypy as spice

    meta_paths = [p for p in METAKERNELS if p.exists()] if METAKERNELS else []
    if not meta_paths and METAKERNEL.exists():
        meta_paths = [METAKERNEL]
    if not meta_paths:
        return False

    rel_paths = set()
    for mk in meta_paths:
        text = mk.read_text()
        m = re.search(r"\\begindata(.*?)(?:\\begintext|$)", text, re.DOTALL)
        if not m:
            continue
        data_block = m.group(1)
        rel_paths.update(re.findall(r"'\$KERNELS/([^']+)'", data_block))

    if not rel_paths:
        return False

    spice.kclear()
    loaded = 0
    for rel in sorted(rel_paths):
        full = KERNELS_DIR / rel
        if full.exists():
            try:
                spice.furnsh(str(full))
                loaded += 1
            except Exception:
                pass
    return loaded > 0


RS_KM = 60268.0
SC = "CASSINI"
ENCELADUS_ID = "ENCELADUS"
CASSINI_ID = -82

# Updated colors: orange reserved for doubtful
COL_LINE = "#607d8b"
COL_CASSINI = "#7b1fa2"  # purple (not orange)
COL_POS = "dodgerblue"
COL_DOUBTFUL = "#ff9800"

_MOON_ORBITS = {
    "F": 140220 / 60268,
    "G": 167500 / 60268,
    "E": 237948 / 60268,
    "T": 294619 / 60268,
    "D": 377396 / 60268,
    "R": 527108 / 60268,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--selection-json", type=Path, default=_DEFAULT_SELECTION_JSON)
    parser.add_argument("--out-png", type=Path, default=_DEFAULT_OUT_PNG)
    parser.add_argument("--out-timeline-png", type=Path, default=_DEFAULT_OUT_TIMELINE_PNG)
    parser.add_argument("--out-with-doubtful-png", type=Path, default=_DEFAULT_OUT_WITH_DOUBTFUL_PNG)
    parser.add_argument("--out-all-png", type=Path, default=_DEFAULT_OUT_ALL_PNG)
    parser.add_argument("--out-json", type=Path, default=_DEFAULT_OUT_JSON)
    return parser.parse_args()


def _load_spice():
    try:
        import spiceypy as spice  # type: ignore
    except Exception as exc:
        raise SystemExit(f"ERROR: spiceypy is required for this script: {exc}")
    return spice


def _plot_saturn_backplate(ax, r_max: float = 9.5, sun_direction: np.ndarray | None = None,
                           fontsize: int = 12) -> None:
    from matplotlib.patches import Circle, Wedge

    if sun_direction is None:
        sun_direction = np.array([0.0, 1.0])

    ax.set_facecolor("#eaf4f7")
    saturn = Circle((0, 0), 1.0, facecolor="white", edgecolor="grey", lw=1.5, zorder=3)
    ax.add_patch(saturn)
    shadow = Wedge((0, 0), 1.0, 180, 360, facecolor="#cfcfcf", edgecolor="none", zorder=3)
    ax.add_patch(shadow)

    for label, r in _MOON_ORBITS.items():
        is_enc = label == "E"
        orb = Circle((0, 0), r, fill=False, edgecolor="grey", lw=1.4 if is_enc else 1.0,
                     linestyle="-" if is_enc else "--", alpha=0.9 if is_enc else 0.6, zorder=2)
        ax.add_patch(orb)
        ax.text(r + 0.15, 0.0, label, fontsize=fontsize, color="grey", va="center")

    sun_dir = sun_direction / np.linalg.norm(sun_direction)
    rhea_r = _MOON_ORBITS.get("R", 8.745)
    lpos = sun_dir * (rhea_r + 0.35)
    ax.text(lpos[0], lpos[1], "☉", ha="center", va="center", fontsize=fontsize, color="grey")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel("Distance [Saturn radii]", fontsize=fontsize)
    ax.set_ylabel("Distance [Saturn radii]", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(False)


def _load_selection(selection_json: Path) -> tuple[list[str], list[str]]:
    if not selection_json.exists():
        raise SystemExit(f"ERROR: selection JSON not found: {selection_json}")

    data = json.loads(selection_json.read_text())
    selected = data.get("selected_opusids", data.get("flagged_opusids", []))
    doubtful = data.get("doubtful_opusids", [])
    if not isinstance(selected, list) or not isinstance(doubtful, list):
        raise SystemExit(f"ERROR: unexpected JSON structure in {selection_json}")

    positive = [str(x) for x in selected]
    doubtful_clean = [str(x) for x in doubtful if str(x) not in set(positive)]
    return positive, doubtful_clean


def _load_metrics(pipeline_dir: Path, opusid: str) -> dict[str, object] | None:
    metrics_path = pipeline_dir / opusid / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return None


def _make_transform(spice, et_ref: float):
    try:
        rot_ref = spice.pxform("J2000", "IAU_SATURN", float(et_ref))
    except Exception as exc:
        raise SystemExit(f"ERROR: could not build Saturn frame transform: {exc}")

    angle_to_rotate = 0.0
    sun_direction_2d = np.array([0.0, 1.0])
    try:
        pos_sun, _ = spice.spkpos("SUN", float(et_ref), "J2000", "LT+S", "SATURN")
        sun_sat = rot_ref.T @ np.array(pos_sun, dtype=float)
        x_sun, y_sun = float(sun_sat[0]), float(sun_sat[1])
        angle_to_rotate = math.atan2(y_sun, x_sun) - math.pi / 2.0
        sun_r2d = math.sqrt(x_sun**2 + y_sun**2)
        if sun_r2d > 0:
            ca, sa = math.cos(-angle_to_rotate), math.sin(-angle_to_rotate)
            sx = ca * x_sun - sa * y_sun
            sy = sa * x_sun + ca * y_sun
            n = math.sqrt(sx**2 + sy**2)
            if n > 0:
                sun_direction_2d = np.array([sx, sy]) / n
    except Exception:
        pass

    def _rot2d(x: float, y: float) -> tuple[float, float]:
        ca = math.cos(-angle_to_rotate)
        sa = math.sin(-angle_to_rotate)
        return x * ca - y * sa, x * sa + y * ca

    def _to_rs(pos_j2000) -> tuple[float, float]:
        p = rot_ref.T @ np.asarray(pos_j2000, dtype=float)
        x, y = _rot2d(float(p[0]), float(p[1]))
        return x / RS_KM, y / RS_KM

    return _to_rs, sun_direction_2d


def _plot_geometry(
    out_path: Path,
    records: list[dict[str, object]],
    *,
    sun_direction_2d: np.ndarray,
    r_max: float,
    title: str,
    include_doubtful_overlay: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=170)
    _plot_saturn_backplate(ax=ax, sun_direction=sun_direction_2d, r_max=r_max)

    ax.plot([], [], color=COL_LINE, lw=1.2, alpha=0.35, label="Line of sight")
    ax.scatter([], [], s=20, c=COL_CASSINI, edgecolors="white", linewidths=0.5, label="Cassini (positive)")
    ax.scatter([], [], s=30, c=COL_POS, edgecolors="white", linewidths=0.5, label="Enceladus (positive)")
    if include_doubtful_overlay:
        ax.scatter([], [], s=20, c=COL_DOUBTFUL, edgecolors="white", linewidths=0.5, label="Cassini (doubtful)")
        ax.scatter([], [], s=30, c=COL_DOUBTFUL, edgecolors="white", linewidths=0.5, label="Enceladus (doubtful)")

    for rec in records:
        obs_x = float(rec["obs_x"])
        obs_y = float(rec["obs_y"])
        enc_x = float(rec["enc_x"])
        enc_y = float(rec["enc_y"])
        label = str(rec.get("label", "positive"))
        cassini_col = COL_DOUBTFUL if (include_doubtful_overlay and label == "doubtful") else COL_CASSINI
        enc_col = COL_DOUBTFUL if (include_doubtful_overlay and label == "doubtful") else COL_POS
        ax.plot([obs_x, enc_x], [obs_y, enc_y], color=COL_LINE, lw=1.0, alpha=0.18, zorder=5)
        ax.scatter([obs_x], [obs_y], s=11, c=cassini_col, edgecolors="white", linewidths=0.2, alpha=0.55, zorder=7)
        ax.scatter([enc_x], [enc_y], s=15, c=enc_col, edgecolors="white", linewidths=0.25, alpha=0.85, zorder=8)

    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.set_title(title, fontsize=12, pad=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_geometry_all_pairs(
    out_path: Path,
    records: list[dict[str, object]],
    *,
    sun_direction_2d: np.ndarray,
    r_max: float,
) -> None:
    FS = 32  # target fontsize: renders at ~11pt at thesis linewidth
    # Tight r_max from actual data extent + padding
    all_coords = [(float(r["obs_x"]), float(r["obs_y"])) for r in records] + \
                 [(float(r["enc_x"]), float(r["enc_y"])) for r in records]
    data_r_max = max(max(abs(x), abs(y)) for x, y in all_coords)
    r_max_tight = max(data_r_max + 1.5, 9.5)

    fig, (ax, ax_hm) = plt.subplots(1, 2, figsize=(18, 11), dpi=170)
    _plot_saturn_backplate(ax=ax, sun_direction=sun_direction_2d, r_max=r_max_tight, fontsize=FS)

    for rec in records:
        obs_x = float(rec["obs_x"])
        obs_y = float(rec["obs_y"])
        enc_x = float(rec["enc_x"])
        enc_y = float(rec["enc_y"])
        ax.plot([obs_x, enc_x], [obs_y, enc_y], color="#6b7280", lw=1.0, alpha=0.15, zorder=5)
        ax.scatter([obs_x], [obs_y], s=11, c="#6b7280", edgecolors="white", linewidths=0.2, alpha=0.4, zorder=7)
        ax.scatter([enc_x], [enc_y], s=15, c="#9aa0a6", edgecolors="white", linewidths=0.25, alpha=0.65, zorder=8)

    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=FS,
            va="top", ha="left", fontweight="bold")

    # Right subplot: Enceladus heat map on the E-orbit (no Cassini sight-lines)
    _plot_saturn_backplate(ax=ax_hm, sun_direction=sun_direction_2d, r_max=r_max_tight, fontsize=FS)
    enc_r = _MOON_ORBITS["E"]
    n_bins = 72  # 5° bins around the orbit
    theta = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)

    enc_x = np.array([float(r["enc_x"]) for r in records], dtype=float)
    enc_y = np.array([float(r["enc_y"]) for r in records], dtype=float)
    enc_ang = (np.arctan2(enc_y, enc_x) + 2.0 * np.pi) % (2.0 * np.pi)
    counts, _ = np.histogram(enc_ang, bins=theta)

    vmax = int(np.max(counts)) if counts.size else 1
    norm = mcolors.Normalize(vmin=0, vmax=max(1, vmax))
    cmap = plt.get_cmap("inferno")

    # Draw colored arc segments along Enceladus' orbit.
    for i in range(n_bins):
        t0, t1 = theta[i], theta[i + 1]
        tt = np.linspace(t0, t1, 12)
        xx = enc_r * np.cos(tt)
        yy = enc_r * np.sin(tt)
        ax_hm.plot(xx, yy, color=cmap(norm(counts[i])), lw=4.0, solid_capstyle="round", zorder=9)

    # Light guide points for actual Enceladus locations used in the all-pairs set.
    ax_hm.scatter(enc_x, enc_y, s=8, c="#e0e0e0", alpha=0.25, linewidths=0, zorder=8)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_hm, fraction=0.046, pad=0.03)
    cbar.set_label("Images per E-orbit sector", fontsize=FS - 4)
    cbar.ax.tick_params(labelsize=FS - 6)

    ax_hm.text(0.02, 0.98, "(b)", transform=ax_hm.transAxes, fontsize=FS,
               va="top", ha="left", fontweight="bold")

    fig.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _opusid_to_image_id(opusid: str) -> str:
    stem = opusid.split("-")[-1]
    return stem[0].upper() + stem[1:] if stem else ""


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    pipeline_dir = data_root / "pipeline_output"
    selection_json = args.selection_json.resolve()
    out_png = args.out_png.resolve()
    out_timeline_png = args.out_timeline_png.resolve()
    out_with_doubtful_png = args.out_with_doubtful_png.resolve()
    out_all_png = args.out_all_png.resolve()
    out_json = args.out_json.resolve()

    if not pipeline_dir.exists():
        raise SystemExit(f"ERROR: pipeline_output directory not found: {pipeline_dir}")

    _make_paths(data_root)

    positive_opusids, doubtful_opusids = _load_selection(selection_json)
    if not positive_opusids and not doubtful_opusids:
        raise SystemExit(f"ERROR: no labels found in {selection_json}")

    spice = _load_spice()
    if spice.ktotal("ALL") == 0 and not _load_spice_kernels():
        raise SystemExit("ERROR: no SPICE kernels are available under the data root")

    all_opusids = list(dict.fromkeys(positive_opusids + doubtful_opusids))
    label_map = {o: "positive" for o in positive_opusids}
    label_map.update({o: "doubtful" for o in doubtful_opusids})

    records: list[dict[str, object]] = []
    et_values: list[float] = []

    for opusid in all_opusids:
        metrics = _load_metrics(pipeline_dir, opusid)
        if not metrics:
            records.append({"opusid": opusid, "status": "missing_metrics", "label": label_map.get(opusid, "positive")})
            continue

        image_id = str(metrics.get("image_id", ""))
        science_time = str(metrics.get("science_time", ""))
        science_filter = str(metrics.get("science_filter", ""))
        if not image_id:
            records.append({"opusid": opusid, "status": "missing_image_id", "label": label_map.get(opusid, "positive")})
            continue

        try:
            et_img = spice.scs2e(CASSINI_ID, "1/" + image_id[1:])
        except Exception as exc:
            records.append({"opusid": opusid, "image_id": image_id, "status": f"spice_error: {exc}", "label": label_map.get(opusid, "positive")})
            continue

        et_values.append(float(et_img))
        records.append(
            {
                "opusid": opusid,
                "image_id": image_id,
                "science_time": science_time,
                "science_filter": science_filter,
                "et_img": float(et_img),
                "status": "ok",
                "label": label_map.get(opusid, "positive"),
            }
        )

    ok_records = [r for r in records if r.get("status") == "ok"]
    if not ok_records:
        raise SystemExit("ERROR: no labeled images could be converted to SPICE geometry")

    et_ref = float(np.median(np.asarray(et_values, dtype=float)))
    to_rs, sun_direction_2d = _make_transform(spice, et_ref)

    def _et_to_utc_iso(et: float) -> str:
        try:
            return spice.et2utc(float(et), "ISOC", 3)
        except Exception:
            return ""

    for rec in ok_records:
        et_img = float(rec["et_img"])
        try:
            pos_c, _ = spice.spkpos(SC, et_img, "J2000", "NONE", "SATURN")
            obs_x, obs_y = to_rs(pos_c)
            pos_enc, _ = spice.spkpos(ENCELADUS_ID, et_img, "J2000", "NONE", "SATURN")
            enc_x, enc_y = to_rs(pos_enc)
        except Exception as exc:
            rec["status"] = f"geometry_error: {exc}"
            continue

        rec["obs_x"] = float(obs_x)
        rec["obs_y"] = float(obs_y)
        rec["enc_x"] = float(enc_x)
        rec["enc_y"] = float(enc_y)
        rec["line_len_rs"] = float(math.hypot(enc_x - obs_x, enc_y - obs_y))
        rec["utc_iso"] = _et_to_utc_iso(et_img)

    valid_records = [r for r in ok_records if r.get("status") == "ok" and all(k in r for k in ("obs_x", "obs_y", "enc_x", "enc_y"))]
    if not valid_records:
        raise SystemExit("ERROR: no labeled images produced valid Cassini/Enceladus positions")

    positive_records = [r for r in valid_records if r.get("label") == "positive"]
    doubtful_records = [r for r in valid_records if r.get("label") == "doubtful"]

    max_r = 0.0
    for rec in valid_records:
        max_r = max(max_r, math.hypot(float(rec["obs_x"]), float(rec["obs_y"])))
        max_r = max(max_r, math.hypot(float(rec["enc_x"]), float(rec["enc_y"])))
    r_max = max(max_r * 1.08, 9.5)

    # Geometry 1: positives only (article title, no diagnostic stats)
    _plot_geometry(
        out_png,
        positive_records,
        sun_direction_2d=sun_direction_2d,
        r_max=r_max,
        title="Luminous-band candidate viewing geometry",
        include_doubtful_overlay=False,
    )

    # Timeline: positives only
    time_values = pd.to_datetime([r["utc_iso"] for r in positive_records if r.get("utc_iso")], utc=True, errors="coerce")
    time_values = time_values[~time_values.isna()]

    timeline_summary: dict[str, object] = {
        "n_points": int(len(time_values)),
        "min_utc": None,
        "max_utc": None,
        "bins": [],
        "counts": [],
    }

    if len(time_values) > 0:
        time_values_naive = pd.DatetimeIndex(time_values.tz_convert(None))
        monthly_counts = pd.Series(1, index=time_values_naive).resample("MS").sum().astype(int)
        bin_centres = monthly_counts.index
        counts = monthly_counts.to_numpy(dtype=int)
        timeline_summary.update(
            {
                "min_utc": str(time_values.min()),
                "max_utc": str(time_values.max()),
                "bins": [str(t) for t in bin_centres],
                "counts": [int(x) for x in counts.tolist()],
                "bin_frequency": "monthly",
            }
        )
    else:
        bin_centres = pd.DatetimeIndex([])
        counts = np.array([], dtype=int)

    fig_t, ax_time = plt.subplots(figsize=(12.5, 4.8), dpi=170)
    ax_time.set_facecolor("#fcfcfc")
    if len(time_values) > 0 and len(counts) > 0:
        centers = bin_centres.to_pydatetime()
        ax_time.bar(
            centers,
            counts,
            width=20,
            align="center",
            color="#90caf9",
            edgecolor="#1e88e5",
            alpha=0.75,
            label="Selections per month",
        )
        ax_time.set_xlim(time_values.min().to_pydatetime(), time_values.max().to_pydatetime())
    else:
        ax_time.text(0.5, 0.5, "No timeline data available", ha="center", va="center")

    ax_time.set_title("Observation timeline", fontsize=12)
    ax_time.set_ylabel("Count")
    ax_time.set_xlabel("UTC time")
    ax_time.grid(True, alpha=0.2)
    ax_time.legend(loc="upper left", fontsize=9)
    fig_t.autofmt_xdate(rotation=30)
    fig_t.tight_layout()
    out_timeline_png.parent.mkdir(parents=True, exist_ok=True)
    fig_t.savefig(out_timeline_png, dpi=180, bbox_inches="tight")
    plt.close(fig_t)

    # Geometry 2: positives + doubtful overlay (third plot)
    _plot_geometry(
        out_with_doubtful_png,
        positive_records + doubtful_records,
        sun_direction_2d=sun_direction_2d,
        r_max=r_max,
        title="Luminous-band viewing geometry (including doubtful cases)",
        include_doubtful_overlay=True,
    )

    # Geometry 3: all image Cassini-Enceladus pairs from pipeline_output
    all_pair_records: list[dict[str, object]] = []
    all_pair_skipped: list[dict[str, object]] = []
    for d in sorted(pipeline_dir.iterdir()):
        if not d.is_dir():
            continue
        opusid = d.name

        metrics = _load_metrics(pipeline_dir, opusid)
        image_id = str(metrics.get("image_id", "")) if metrics else ""
        if not image_id:
            image_id = _opusid_to_image_id(opusid)
        if not image_id:
            all_pair_skipped.append({"opusid": opusid, "status": "missing_image_id"})
            continue

        try:
            et_img = spice.scs2e(CASSINI_ID, "1/" + image_id[1:])
            pos_c, _ = spice.spkpos(SC, et_img, "J2000", "NONE", "SATURN")
            obs_x, obs_y = to_rs(pos_c)
            pos_enc, _ = spice.spkpos(ENCELADUS_ID, et_img, "J2000", "NONE", "SATURN")
            enc_x, enc_y = to_rs(pos_enc)
        except Exception as exc:
            all_pair_skipped.append({"opusid": opusid, "image_id": image_id, "status": f"geometry_error: {exc}"})
            continue

        all_pair_records.append(
            {
                "opusid": opusid,
                "image_id": image_id,
                "obs_x": float(obs_x),
                "obs_y": float(obs_y),
                "enc_x": float(enc_x),
                "enc_y": float(enc_y),
                "line_len_rs": float(math.hypot(enc_x - obs_x, enc_y - obs_y)),
            }
        )

    r_max_all = r_max
    for rec in all_pair_records:
        r_max_all = max(r_max_all, math.hypot(float(rec["obs_x"]), float(rec["obs_y"])))
        r_max_all = max(r_max_all, math.hypot(float(rec["enc_x"]), float(rec["enc_y"])))
    r_max_all = max(r_max_all * 1.08, 9.5)

    _plot_geometry_all_pairs(
        out_all_png,
        all_pair_records,
        sun_direction_2d=sun_direction_2d,
        r_max=r_max_all,
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_json": str(selection_json),
        "selection_count": len(positive_opusids),
        "doubtful_count": len(doubtful_opusids),
        "plotted_positive_count": len(positive_records),
        "plotted_doubtful_count": len(doubtful_records),
        "all_pairs_total_in_pipeline_output": len([d for d in pipeline_dir.iterdir() if d.is_dir()]),
        "all_pairs_plotted_count": len(all_pair_records),
        "all_pairs_skipped_count": len(all_pair_skipped),
        "reference_et": float(et_ref),
        "reference_mode": "median_labeled_image_epoch",
        "output_png": str(out_png),
        "timeline_png": str(out_timeline_png),
        "output_with_doubtful_png": str(out_with_doubtful_png),
        "output_all_png": str(out_all_png),
        "timeline": timeline_summary,
        "records": valid_records,
        "all_pairs_records": all_pair_records,
        "all_pairs_skipped": all_pair_skipped,
        "skipped": [r for r in records if r.get("status") != "ok"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"Saved geometry overview          → {out_png}")
    print(f"Saved timeline plot             → {out_timeline_png}")
    print(f"Saved geometry with doubtful    → {out_with_doubtful_png}")
    print(f"Saved all-pairs geometry        → {out_all_png}")
    print(f"Saved companion JSON            → {out_json}")


if __name__ == "__main__":
    main()
