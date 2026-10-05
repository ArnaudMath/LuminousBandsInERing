"""
wp2_image_plane_alignment.py
────────────────────────────
Which model direction do the measured bands actually align with?

The ring-plane deviation angle Delta-phi is geometry-locked at the near-grazing
viewing of this survey and carries almost no information about the bands. The
informative comparison is made in the image plane, where the angular spectrum
operates natively. This script projects BOTH candidate model directions into
the image and measures the folded angle to the observed band direction:

  b_pred     = n_ring x v_grating   (the grating LINES, in the ring plane)
  v_grating  = n_scat x n_ring      (the grating VECTOR, i.e. the direction of
                                     the periodic spacing, perpendicular to the
                                     lines within the ring plane)

plus two geometric reference directions at the ring intercept point:

  azimuthal  = n_ring x r_hat       (the local ring line)
  radial     = r_hat                (in-plane, away from Saturn)

Projection: a line through the ring intercept point P with 3D direction d maps,
under the gnomonic camera projection, to the image direction

    dU = -(d_x p_z - p_x d_z)/p_z^2 ,  dV = -(d_y p_z - p_y d_z)/p_z^2

with p = P - r_cassini expressed in the NAC frame, U along image column and V
along image row (matching pipeline/08_detect_bands.py and
compute_band_orientations.py: image_col = -NAC_X, image_row = -NAC_Y).

Observed band angle: theta_band = (null_peak_theta_64_deg + 90) mod 180, since
the spectral peak is a wavevector angle and the bands run perpendicular to it.
This convention is taken unchanged from the detection pipeline.

Output (review only, overwrites nothing used by the article):
  figure_scripts/output/wp2_image_plane_alignment.csv
  figure_scripts/output/wp2_image_plane_alignment_summary.txt
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as spice

_ROOT        = Path(__file__).resolve().parent.parent
LABELS_JSON  = _ROOT / "post_pipeline" / "11_bundle_A_review_labels.json"
PIPELINE_DIR = _ROOT / "pipeline_output"
KERNEL_DIR   = _ROOT / "spice_kernels_staged"
METAKERNEL   = KERNEL_DIR / "cassini_iss_pipeline.tm"
OUT_DIR      = Path(__file__).resolve().parent / "output"

SC_ID  = -82
RS_KM  = 60268.0

EXTRA_SPKS = [
    "spk/200128RU_SCPSE_10360_11041.bsp",
    "spk/200128RU_SCPSE_12016_12060.bsp",
]


def load_spice() -> None:
    os.chdir(KERNEL_DIR)
    spice.kclear()
    spice.furnsh(METAKERNEL.name)
    for extra in EXTRA_SPKS:
        if Path(extra).exists():
            spice.furnsh(extra)


def et_for(opusid: str, science_time: str) -> float | None:
    if science_time and science_time.lower() != "unknown":
        utc = science_time.replace("+00:00", "").replace(" ", "T")
        try:
            return float(spice.str2et(utc))
        except Exception:
            pass
    try:
        return float(spice.scs2e(SC_ID, opusid.split("-n")[-1]))
    except Exception:
        return None


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def project_line_angle(d_j2000, p_nac, M) -> float:
    """Image-plane position angle (deg, mod 180) of a 3D line direction."""
    d = M.T @ unit(d_j2000)          # into NAC frame
    px, py, pz = p_nac
    if abs(pz) < 1e-12:
        return float("nan")
    dU = -(d[0] * pz - px * d[2]) / pz**2
    dV = -(d[1] * pz - py * d[2]) / pz**2
    if abs(dU) < 1e-15 and abs(dV) < 1e-15:
        return float("nan")
    return math.degrees(math.atan2(dV, dU)) % 180.0


def fold(a: float, b: float) -> float:
    """Undirected angle between two image position angles, in [0, 90]."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    d = abs(a - b) % 180.0
    return d if d <= 90.0 else 180.0 - d


def one(opusid: str, met: dict, force_fallback: bool = False) -> dict | None:
    theta_zp = met.get("null_peak_theta_64_deg")
    if theta_zp is None:
        return None
    th_obs = (float(theta_zp) + 90.0) % 180.0

    et = et_for(opusid, str(met.get("science_time")))
    if et is None:
        return None

    try:
        M       = np.array(spice.pxform("CASSINI_ISS_NAC", "J2000", et))
        cas_pos = np.asarray(spice.spkpos("CASSINI", et, "J2000", "NONE", "SATURN")[0], float)
        sun_pos = np.asarray(spice.spkpos("SUN", et, "J2000", "LT+S", "SATURN")[0], float)
        enc_pos = np.asarray(spice.spkpos("ENCELADUS", et, "J2000", "NONE", "SATURN")[0], float)
        M_pole  = np.array(spice.pxform("IAU_SATURN", "J2000", et))
    except Exception:
        return None

    n_ring = M_pole[:, 2]
    bs     = M @ np.array([0.0, 0.0, 1.0])

    denom = float(np.dot(bs, n_ring))
    ring_open = math.degrees(math.asin(min(abs(denom), 1.0)))

    # Reference point in the ring plane along the line of sight.
    #
    # Primary: the boresight ray intersected with the ring plane.
    #
    # Fallback: at the most grazing geometries (ring opening angle below about
    # 0.2 deg) the ray does not cross the ring plane ahead of the spacecraft at
    # all, so no intercept exists. There the point on the line of sight closest
    # to Enceladus is used, projected into the ring plane. Cassini is itself
    # within a few hundred km of the ring plane at these epochs, so the
    # projection is small. The two methods agree closely wherever both are
    # defined; see ref_method in the output table.
    P = None
    method = "intercept"
    if abs(denom) > 1e-9 and not force_fallback:
        t = -float(np.dot(cas_pos, n_ring)) / denom
        if t > 0:
            P = cas_pos + t * bs
    if P is None:
        t_enc = float(np.dot(enc_pos - cas_pos, bs))
        if t_enc <= 0:
            return None
        P_los = cas_pos + t_enc * bs
        P = P_los - float(np.dot(P_los, n_ring)) * n_ring
        method = "los_closest_to_enceladus"

    k_cas = unit(cas_pos - P)
    k_sun = unit(sun_pos - P)
    n_scat = np.cross(k_sun, k_cas)
    if np.linalg.norm(n_scat) < 1e-12:
        return None
    n_scat = unit(n_scat)

    v_grating = unit(np.cross(n_scat, n_ring))
    b_pred    = unit(np.cross(n_ring, v_grating))

    r_vec  = P - np.dot(P, n_ring) * n_ring   # in-plane radial at intercept
    r_hat  = unit(r_vec)
    azim   = unit(np.cross(n_ring, r_hat))

    # Control: intersection line of the focal plane with the ring plane. This is
    # the direction that the ring-plane projection collapses onto at grazing
    # viewing, so any alignment that merely reflects viewing geometry shows up
    # here. A genuine orientation result must sit well away from it.
    fp_line = np.cross(bs, n_ring)

    p_nac = M.T @ (P - cas_pos)

    th_bpred = project_line_angle(b_pred,    p_nac, M)
    th_gvec  = project_line_angle(v_grating, p_nac, M)
    th_azim  = project_line_angle(azim,      p_nac, M)
    th_rad   = project_line_angle(r_hat,     p_nac, M)
    th_fp    = project_line_angle(fp_line,   p_nac, M)

    return {
        "opusid": opusid,
        "science_time": str(met.get("science_time")),
        "ref_method": method,
        "ring_open_deg": round(ring_open, 3),
        "r_intercept_rs": round(float(np.linalg.norm(P)) / RS_KM, 3),
        "th_obs": round(th_obs, 2),
        "th_bpred_img": round(th_bpred, 2),
        "th_gvec_img": round(th_gvec, 2),
        "th_azim_img": round(th_azim, 2),
        "th_radial_img": round(th_rad, 2),
        "th_fpline_img": round(th_fp, 2),
        "d_to_bpred": round(fold(th_obs, th_bpred), 2),
        "d_to_gvec": round(fold(th_obs, th_gvec), 2),
        "d_to_azim": round(fold(th_obs, th_azim), 2),
        "d_to_radial": round(fold(th_obs, th_rad), 2),
        "d_to_fpline": round(fold(th_obs, th_fp), 2),
        "bpred_gvec_sep_img": round(fold(th_bpred, th_gvec), 2),
    }


def circ_corr(obs_deg, pred_deg) -> float:
    """Correlation for axial (mod 180) angles: double them, then correlate."""
    a = np.deg2rad(2.0 * np.asarray(obs_deg, float))
    b = np.deg2rad(2.0 * np.asarray(pred_deg, float))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    sa, sb = np.sin(a - a.mean()), np.sin(b - b.mean())
    return float(np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2)))


def main() -> int:
    load_spice()
    labels = json.loads(LABELS_JSON.read_text())
    rows = []
    for opusid in sorted(labels["selected_opusids"]):
        mpath = PIPELINE_DIR / opusid / "metrics.json"
        if not mpath.exists():
            continue
        r = one(opusid, json.loads(mpath.read_text()))
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    df["group"] = df["opusid"].apply(
        lambda o: "pre-equinox" if o.startswith(("co-iss-n160", "co-iss-n161")) else "post-equinox")

    # Cross-validate the fallback reference point against the exact intercept
    # wherever the intercept exists.
    cross = []
    for opusid in df[df.ref_method == "intercept"].opusid:
        met = json.loads((PIPELINE_DIR / opusid / "metrics.json").read_text())
        alt = one(opusid, met, force_fallback=True)
        if alt is not None:
            ref = df[df.opusid == opusid].iloc[0]
            cross.append(fold(float(ref.th_gvec_img), float(alt["th_gvec_img"])))
    cross = np.array([c for c in cross if np.isfinite(c)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "wp2_image_plane_alignment.csv", index=False)

    lines = [f"n = {len(df)} positive detections with a valid ring-plane intercept", ""]
    for grp in ["pre-equinox", "post-equinox", "ALL"]:
        sub = df if grp == "ALL" else df[df.group == grp]
        lines.append(f"── {grp}  (n={len(sub)}) " + "─" * 34)
        lines.append(f"  ring opening angle: {sub.ring_open_deg.min():.2f} to "
                     f"{sub.ring_open_deg.max():.2f} deg")
        lines.append(f"  observed band angle spans {sub.th_obs.min():.1f} to "
                     f"{sub.th_obs.max():.1f} deg")
        for col in ["d_to_gvec", "d_to_bpred", "d_to_azim", "d_to_radial",
                    "d_to_fpline", "bpred_gvec_sep_img"]:
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            lines.append(f"  {col:20s} mean={v.mean():6.2f}  median={v.median():6.2f}"
                         f"  std={v.std():5.2f}  min={v.min():6.2f}  max={v.max():6.2f}")
        lines.append("")

    lines.append("── axial correlation of each candidate direction with the "
                 "observed band angle (all detections) " + "─" * 3)
    for col in ["th_gvec_img", "th_bpred_img", "th_azim_img", "th_radial_img",
                "th_fpline_img"]:
        lines.append(f"  {col:16s} r = {circ_corr(df.th_obs, df[col]): .3f}")
    lines.append("")
    lines.append("d_to_fpline is the control: the focal-plane / ring-plane "
                 "intersection is the geometry-locked direction, so a large")
    lines.append("value there means the agreement with the grating vector is "
                 "not an artefact of the near-grazing viewing geometry.")
    lines.append("")

    n_fb = int((df.ref_method != "intercept").sum())
    lines.append("── reference point ".ljust(52, "─"))
    lines.append(f"  exact ring-plane intercept: {len(df) - n_fb}")
    lines.append(f"  fallback (no intercept ahead of the spacecraft): {n_fb}")
    if len(cross):
        lines.append(f"  the two methods differ by at most {cross.max():.3f} deg "
                     f"(median {np.median(cross):.3f}) where both are defined")
    lines.append("")

    lines.append("── does the residual to the grating vector trend with geometry? "
                 + "─" * 3)
    for col in ["ring_open_deg", "r_intercept_rs"]:
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df["d_to_gvec"], errors="coerce")
        ok = x.notna() & y.notna()
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        lines.append(f"  d_to_gvec vs {col:16s} Pearson r = {r: .3f}")
    lines.append("  (a clear trend would point to a systematic effect such as an "
                 "out-of-plane grating inclination;")
    lines.append("   no trend is consistent with the residual being measurement "
                 "scatter)")

    summary = "\n".join(lines)
    print(summary)
    (OUT_DIR / "wp2_image_plane_alignment_summary.txt").write_text(summary + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
