#!/usr/bin/env python3
"""
12b_geometry_per_epoch.py
─────────────────────────
Corrected copy of the geometry overview of 12_selected_bundle_A_geometry_overview.py.
12 and its JSON are left untouched.

What changes
  12 projects every image with ONE transform built at the median epoch:
  the Sun direction of that epoch is put at +y for all images. Over 2007--2012
  the Sun direction moves by about 55 deg in Saturn's frame (58 to 112 deg in
  that fixed frame), so the ansa of early and late images is off by up to ~30 deg.

  12b uses, for each image, the Saturn equatorial frame (IAU_SATURN) AT THAT
  IMAGE'S EPOCH and rotates it about the pole so that the Sun of that epoch is
  at +y. Positions are therefore Sun-relative at the moment of exposure.

Convention (checked with SPICE): viewed from Saturn's north pole, x right,
y up (Sun), Enceladus moves counterclockwise and Saturn's shadow is at 270 deg,
so ring particles leave the shadow towards 0 deg: +x (0 deg) is the MORNING
(dawn) ansa, -x (180 deg) the EVENING ansa. This is the convention of
make_ansa_coverage_histogram.py.

Input   post_pipeline/12_selected_bundle_A_geometry_overview.json  (record list)
Output  post_pipeline/12b_geometry_per_epoch.json  (same layout, new positions)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT    = Path(__file__).resolve().parents[1]
IN_JSON  = _ROOT / "post_pipeline" / "12_selected_bundle_A_geometry_overview.json"
OUT_JSON = _ROOT / "post_pipeline" / "12b_geometry_per_epoch.json"
RS_KM    = 60268.0
SC_ID    = -82

import importlib.util                   # noqa: E402
import spiceypy as spice                 # noqa: E402

# Same kernel set as script 12 (full cosp_1000 archive), via its own loader
_spec = importlib.util.spec_from_file_location(
    "geo12", Path(__file__).resolve().parent / "12_selected_bundle_A_geometry_overview.py")
_geo12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo12)


def _sun_up_xy(et: float):
    """Return f(pos_J2000) -> (x, y) in R_S, Saturn equatorial plane, Sun at +y at epoch et."""
    R = np.array(spice.pxform("J2000", "IAU_SATURN", et))
    sun = R @ np.asarray(spice.spkpos("SUN", et, "J2000", "LT+S", "SATURN")[0], float)
    a = math.pi / 2.0 - math.atan2(sun[1], sun[0])
    ca, sa = math.cos(a), math.sin(a)

    def f(pos):
        p = R @ np.asarray(pos, float)
        return (ca * p[0] - sa * p[1]) / RS_KM, (sa * p[0] + ca * p[1]) / RS_KM
    return f


def _recompute(rec: dict) -> dict | None:
    out = dict(rec)
    try:
        et = float(rec["et_img"]) if "et_img" in rec else \
             float(spice.scs2e(SC_ID, "1/" + rec["image_id"][1:]))
        f = _sun_up_xy(et)
        ox, oy = f(spice.spkpos("CASSINI", et, "J2000", "NONE", "SATURN")[0])
        ex, ey = f(spice.spkpos("ENCELADUS", et, "J2000", "NONE", "SATURN")[0])
    except Exception as exc:
        print(f"[12b] {rec.get('opusid')}: {exc}")
        return None
    out.update(obs_x=float(ox), obs_y=float(oy), enc_x=float(ex), enc_y=float(ey),
               line_len_rs=float(math.hypot(ex - ox, ey - oy)), et_img=et)
    return out


def main() -> int:
    data = json.loads(IN_JSON.read_text())
    _geo12._make_paths(_ROOT)
    if not _geo12._load_spice_kernels():
        raise SystemExit("[12b] could not load the cosp_1000 kernels")

    out = {k: v for k, v in data.items() if not isinstance(v, list)}
    out.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
               reference_mode="per_image_epoch (Sun at +y at each exposure)",
               source_json=str(IN_JSON))
    out.pop("reference_et", None)
    for key in ("records", "all_pairs_records"):
        new = [r for r in (_recompute(rec) for rec in data[key]) if r is not None]
        out[key] = new
        print(f"[12b] {key}: {len(new)} / {len(data[key])} recomputed")
    out["skipped"], out["all_pairs_skipped"] = data.get("skipped", []), data.get("all_pairs_skipped", [])

    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"[12b] wrote {OUT_JSON}")

    def ansa(r):
        a = math.degrees(math.atan2(r["enc_y"], r["enc_x"])) % 360
        return ("evening" if 135 <= a < 225 else "anti-solar" if 225 <= a < 315
                else "sub-solar" if 45 <= a < 135 else "morning")
    for name, recs in [("all pairs", out["all_pairs_records"]),
                       ("positives", [r for r in out["records"] if r.get("label") == "positive"]),
                       ("doubtful", [r for r in out["records"] if r.get("label") == "doubtful"])]:
        c = {a: 0 for a in ("morning", "sub-solar", "evening", "anti-solar")}
        for r in recs:
            c[ansa(r)] += 1
        print(f"[12b] {name:10s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
