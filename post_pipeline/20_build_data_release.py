#!/usr/bin/env python3
"""
20_build_data_release.py
────────────────────────
Builds the output-data deposit for Zenodo (reviewer comments BS24, BS195):
the list of every image pair considered in the survey, with acquisition
metadata, geometry, selection flags and results, plus the tables behind the
article figures.

Inputs (all existing, read-only)
  inferred_sets/pairs.parquet                    main run, 631 (CLEAR, science) pairs
  .../survey_B/inferred_sets/pairs.parquet       CLEAR partner for the 45 early-run pairs
                                                 (later reconstruction; not verifiable)
  .../survey_B/inferred_sets/pairs_infer_set.parquet   (fallback, 156EN)
  pre_pipeline/06_flagged_pairs.json             quality-control exclusions (main run)
  pre_pipeline/06b_flagged_pairs.json            review of the 51 pairs that skipped step 06
  pre_pipeline/clean_pairs.parquet               573 pairs passing QC (main run)
  pipeline_output/<id>/metrics.json              detection results, 624 pairs
  post_pipeline/11_bundle_A_review_labels.json   positive / doubtful labels
  post_pipeline/14b_contrast_table.csv           delta, C, SNR (56 positives)
  post_pipeline/12b_geometry_per_epoch.json      Enceladus longitude, Sun at 90 deg
  figure_scripts/output/wp2_image_plane_alignment.csv   predicted orientation
  OPUS metadata API (cached in release/zenodo_data/opus_metadata_cache.json)

Output  release/zenodo_data/
  image_pairs.csv               one row per pair (main run + survey_B)
  figure_orientation_agreement.csv
  figure_contrast_trends.csv
  figure_viewing_geometry.csv
  README.md                     column descriptions
"""

from __future__ import annotations

import glob
import json
import math
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT  = pathlib.Path(__file__).resolve().parents[1]
OUT   = ROOT / "release" / "zenodo_data"
CACHE = OUT / "opus_metadata_cache.json"
SURVEY_B_PAIRS = pathlib.Path("/mnt/storage/pipeline_output/surveys/survey_B/inferred_sets/pairs.parquet")

OPUS_FIELDS = {
    "General Constraints": ["time1"],
    "Cassini Mission Constraints": ["CASSINIrevnoint", "CASSINIobsname"],
    "Cassini ISS Constraints": ["COISSfilter"],
    "Ring Geometry Constraints": ["RINGGEOphase1", "RINGGEOphase2",
                                  "RINGGEOobserverringelevation1", "RINGGEOobserverringelevation2",
                                  "RINGGEOringradius1", "RINGGEOringradius2",
                                  "RINGGEOresolution1", "RINGGEOresolution2"],
}


def _fetch(oid: str) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(f"https://opus.pds-rings.seti.org/opus/api/metadata/{oid}.json", timeout=60)
            r.raise_for_status()
            d = r.json()
            return {k: d.get(sec, {}).get(k) for sec, ks in OPUS_FIELDS.items() for k in ks}
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def opus_metadata(ids: list[str]) -> dict[str, dict]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [i for i in ids if not cache.get(i)]
    print(f"[20] OPUS metadata: {len(ids) - len(todo)} cached, {len(todo)} to fetch")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for oid, md in zip(todo, ex.map(_fetch, todo)):
            cache[oid] = md
    CACHE.write_text(json.dumps(cache, indent=0))
    return cache


def ansa(lon: float) -> str:
    return ("evening" if 135 <= lon < 225 else "anti-solar" if 225 <= lon < 315
            else "sub-solar" if 45 <= lon < 135 else "morning")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    main_pairs = pd.read_parquet(ROOT / "inferred_sets/pairs.parquet").assign(run="main")
    processed = {pathlib.Path(p).parent.name for p in glob.glob(str(ROOT / "pipeline_output/co-iss-*/metrics.json"))}
    # survey_B pairs: pairs.parquet first, then pairs_infer_set.parquet for the 156EN pairs it lacks
    b = pd.concat([pd.read_parquet(SURVEY_B_PAIRS),
                   pd.read_parquet(SURVEY_B_PAIRS.with_name("pairs_infer_set.parquet"))]).drop_duplicates("science_opusid")
    # These 45 pairs come from an earlier step-02 run (April 2026) whose pair table was later
    # overwritten. Their CLEAR partner is taken from a later reconstruction and could not be
    # verified against the April residuals (no longer stored): clear_verified = False.
    b = b[b.science_opusid.isin(processed - set(main_pairs.science_opusid))].assign(run="early_run")
    df = pd.concat([main_pairs, b], ignore_index=True).drop_duplicates("science_opusid")

    flagged = set(json.loads((ROOT / "pre_pipeline/06_flagged_pairs.json").read_text())["flagged_opusids"])
    clean = set(pd.read_parquet(ROOT / "pre_pipeline/clean_pairs.parquet").science_opusid)
    lab = json.loads((ROOT / "post_pipeline/11_bundle_A_review_labels.json").read_text())
    pos, dbt = set(lab["selected_opusids"]), set(lab["doubtful_opusids"])

    df["clear_verified"] = df.run == "main"
    df["qc_flagged"] = df.science_opusid.isin(flagged)
    rev06b = ROOT / "pre_pipeline/06b_flagged_pairs.json"
    flagged_b = set(json.loads(rev06b.read_text())["flagged_opusids"]) if rev06b.exists() else set()
    df["qc_flagged"] |= df.science_opusid.isin(flagged_b)
    df["in_clean_pairs_573"] = df.science_opusid.isin(clean)
    df["detection_run"] = df.science_opusid.isin(processed)
    df["qc_review"] = df.science_opusid.map(
        lambda o: "06" if (o in clean or o in flagged) else ("06b" if o in processed else ""))
    df["classification"] = df.science_opusid.map(lambda o: "positive" if o in pos else "doubtful" if o in dbt else "")

    md = opus_metadata(sorted(df.science_opusid))
    meta = pd.DataFrame.from_dict({k: md.get(k, {}) for k in df.science_opusid}, orient="index")
    meta.index.name = "science_opusid"
    df = df.merge(meta.reset_index(), on="science_opusid", how="left")
    df["flyby"] = df.CASSINIobsname.str.extract(r"ISS_(\d{3}[A-Z]{2})_")

    res = []
    for o in df.science_opusid:
        p = ROOT / "pipeline_output" / o / "metrics.json"
        m = json.loads(p.read_text()) if p.exists() else {}
        th = m.get("null_peak_theta_64_deg")
        res.append({"science_opusid": o,
                    "theta_star_deg": (th + 90.0) % 180.0 if th is not None else None,
                    "fwhm_deg": m.get("null_fwhm_64_deg"),
                    "zp_snr": m.get("zp_snr")})
    df = df.merge(pd.DataFrame(res), on="science_opusid", how="left")

    c = pd.read_csv(ROOT / "post_pipeline/14b_contrast_table.csv")[["opusid", "delta", "C", "snr"]]
    c.columns = ["science_opusid", "contrast_delta_IF", "contrast_C", "contrast_snr"]
    df = df.merge(c, on="science_opusid", how="left")

    g = json.loads((ROOT / "post_pipeline/12b_geometry_per_epoch.json").read_text())
    geo = pd.DataFrame([{"science_opusid": r["opusid"],
                         "enceladus_longitude_deg": math.degrees(math.atan2(r["enc_y"], r["enc_x"])) % 360,
                         "cassini_x_RS": r["obs_x"], "cassini_y_RS": r["obs_y"],
                         "enceladus_x_RS": r["enc_x"], "enceladus_y_RS": r["enc_y"]}
                        for r in g["all_pairs_records"]])
    geo["ansa"] = geo.enceladus_longitude_deg.map(ansa)
    df = df.merge(geo, on="science_opusid", how="left")

    al = pd.read_csv(ROOT / "figure_scripts/output/wp2_image_plane_alignment.csv")
    df = df.merge(al[["opusid", "th_gvec_img", "d_to_gvec"]].rename(columns={
        "opusid": "science_opusid", "th_gvec_img": "predicted_orientation_deg",
        "d_to_gvec": "deviation_from_prediction_deg"}), on="science_opusid", how="left")

    cols = ["science_opusid", "clear_opusid", "clear_verified", "run", "science_filter", "time1", "clear_time1", "dt_s",
            "CASSINIrevnoint", "flyby", "CASSINIobsname",
            "RINGGEOphase1", "RINGGEOphase2", "RINGGEOobserverringelevation1", "RINGGEOobserverringelevation2",
            "RINGGEOringradius1", "RINGGEOringradius2", "RINGGEOresolution1", "RINGGEOresolution2",
            "qc_review", "qc_flagged", "in_clean_pairs_573", "detection_run", "classification",
            "theta_star_deg", "fwhm_deg", "zp_snr",
            "contrast_delta_IF", "contrast_C", "contrast_snr",
            "enceladus_longitude_deg", "ansa", "cassini_x_RS", "cassini_y_RS", "enceladus_x_RS", "enceladus_y_RS",
            "predicted_orientation_deg", "deviation_from_prediction_deg"]
    df = df[cols].rename(columns={"time1": "science_time_utc", "clear_time1": "clear_time_utc",
                                  "dt_s": "clear_science_dt_s", "CASSINIrevnoint": "orbit"})
    df = df.sort_values("science_time_utc").reset_index(drop=True)
    df.to_csv(OUT / "image_pairs.csv", index=False, float_format="%.6g")

    df[df.classification == "positive"][["science_opusid", "flyby", "science_filter", "theta_star_deg",
        "predicted_orientation_deg", "deviation_from_prediction_deg"]].to_csv(
        OUT / "figure_orientation_agreement.csv", index=False, float_format="%.6g")
    df[df.contrast_C.notna()][["science_opusid", "flyby", "science_filter", "contrast_delta_IF", "contrast_C",
        "contrast_snr", "RINGGEOphase1", "RINGGEOphase2"]].to_csv(
        OUT / "figure_contrast_trends.csv", index=False, float_format="%.6g")
    df[df.detection_run][["science_opusid", "flyby", "classification", "enceladus_longitude_deg", "ansa",
        "cassini_x_RS", "cassini_y_RS", "enceladus_x_RS", "enceladus_y_RS"]].to_csv(
        OUT / "figure_viewing_geometry.csv", index=False, float_format="%.6g")

    print(f"[20] image_pairs.csv: {len(df)} pairs  "
          f"(main {sum(df.run == 'main')}, early_run {sum(df.run == 'early_run')}; "
          f"QC-flagged {int(df.qc_flagged.sum())}; detection run {int(df.detection_run.sum())}; "
          f"positive {sum(df.classification == 'positive')}, doubtful {sum(df.classification == 'doubtful')})")
    print(f"[20] missing OPUS metadata: {int(df.orbit.isna().sum())}, missing flyby: {int(df.flyby.isna().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
