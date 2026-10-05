"""
make_classified_images_table.py
───────────────────────────────
LaTeX table of all 88 classified images (62 positive, 26 doubtful) for the
article appendix (reviewer comment Seignovert BS91).

One full-width table* with the images in two side-by-side halves, sorted by
image number (i.e. time), so it fits the two-column cas-dc layout without
longtable.

Sources:
  post_pipeline/11_bundle_A_review_labels.json   positive / doubtful labels
  pipeline_output/<opusid>/metrics.json          theta* and FWHM (w = 64),
                                                 same values as Figure 11
  PAIR_TABLES (main, then survey_B)             paired CLEAR image
  <raw_images>/<opusid>_CALIB.LBL                OBSERVATION_ID (flyby), filter

theta* is the band orientation (wavevector peak + 90°) mod 180°, measured from
the sample axis towards the line axis (paper Section 4 convention).

Output:
  figure_scripts/output/classified_images_table.tex   (Appendix B, all 88 images)
  figure_scripts/output/orientation_summary_table.tex (Section 5, positives per
                                                       flyby; reviewer BS112)
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "post_pipeline" / "11_bundle_A_review_labels.json"
PIPE = ROOT / "pipeline_output"
LBL_ROOTS = [ROOT / "raw_images", Path("/mnt/storage/raw_images"),
             Path("/mnt/storage/pipeline_output/surveys/survey_B/raw_images"),
             Path("/mnt/storage/pipeline_output/surveys/survey_C/raw_images")]
PAIR_TABLES = [ROOT / "inferred_sets" / "pairs.parquet",
               Path("/mnt/storage/pipeline_output/surveys/survey_B/inferred_sets/pairs.parquet")]
OUT = Path(__file__).parent / "output" / "classified_images_table.tex"
OUT_SUMMARY = Path(__file__).parent / "output" / "orientation_summary_table.tex"


def read_label(opusid: str) -> tuple[str, str]:
    """Return (flyby tag, science filter) from the calibrated PDS label."""
    for r in LBL_ROOTS:
        f = r / f"{opusid}_CALIB.LBL"
        if f.exists():
            t = f.read_text(errors="ignore")
            obs = re.search(r'OBSERVATION_ID\s*=\s*"?([^"\r\n]+)', t).group(1)
            filt = re.search(r"FILTER_NAME\s*=\s*\(([^)]*)\)", t).group(1)
            filt = [x.strip(' "') for x in filt.split(",")]
            sci = [x for x in filt if x not in ("CL1", "CL2")]
            return re.search(r"ISS_(\d{3}[A-Z]{2})_", obs).group(1), "+".join(sci)
    raise FileNotFoundError(opusid)


def load() -> pd.DataFrame:
    lab = json.loads(LABELS.read_text())
    rows = [r for v in lab.values()
            if isinstance(v, list) and v and isinstance(v[0], dict) for r in v]
    df = pd.DataFrame(rows)[["opusid", "label"]]

    met = pd.DataFrame([json.loads((PIPE / o / "metrics.json").read_text())
                        for o in df.opusid])
    df = df.merge(met[["opusid", "null_peak_theta_64_deg", "null_fwhm_64_deg"]],
                  on="opusid")
    df["theta"] = (df.null_peak_theta_64_deg + 90.0) % 180.0
    df["fwhm"] = df.null_fwhm_64_deg

    # Pair tables disagree for some images; the main table first, then the
    # survey_B table for images it lacks (142EN), reproduces the CLEAR frame of
    # every stored residual.npz (checked pixel by pixel, 2026-09-28). The
    # reverse order gives 23 wrong pairs, so keep this order.
    pairs = pd.concat(
        pd.read_parquet(f)[["science_opusid", "clear_opusid"]] for f in PAIR_TABLES
    ).drop_duplicates("science_opusid")
    df = df.merge(pairs, left_on="opusid", right_on="science_opusid", how="left")
    if df.clear_opusid.isna().any():
        raise RuntimeError("missing CLEAR pair for "
                           + ", ".join(df.opusid[df.clear_opusid.isna()]))

    df[["flyby", "filter"]] = [read_label(o) for o in df.opusid]
    return df.sort_values("opusid").reset_index(drop=True)


def img(opusid: str) -> str:
    return opusid.replace("co-iss-n", "N")


def row(r) -> str:
    cls = "P" if r.label == "positive" else "D"
    return (f"{img(r.opusid)} & {img(r.clear_opusid)} & {r.flyby} & {r['filter']} "
            f"& {cls} & {r.theta:.1f} & {r.fwhm:.0f}")


def main() -> None:
    df = load()
    n = len(df)
    half = (n + 1) // 2
    left = [row(r) for _, r in df.iloc[:half].iterrows()]
    right = [row(r) for _, r in df.iloc[half:].iterrows()]
    right += ["& & & & & &"] * (half - len(right))

    head = (r"Image & CLEAR & Flyby & Filter & Class & $\theta^\star$ ($\degree$) "
            r"& FWHM ($\degree$)")
    lines = [
        r"\begin{table*}",
        r"\caption{All images classified in this survey: " + f"{(df.label == 'positive').sum()} "
        r"positive (P) and " + f"{(df.label == 'doubtful').sum()} " +
        r"doubtful (D) detections, ordered by image number. \textit{Image}: "
        r"science-filter frame; \textit{CLEAR}: the paired CLEAR frame subtracted from it "
        r"(Section~\ref{subsec:clear_sub}); \textit{Flyby}: Cassini orbit tag. "
        r"$\theta^\star$ is the band orientation of the strongest peak of $E_{\mathrm{diff}}(\theta)$, "
        r"measured from the image sample axis (Section~\ref{sec:DetectionMethod}), and FWHM "
        r"its full width at half maximum. For doubtful images, $\theta^\star$ and FWHM describe "
        r"the strongest spectral peak and do not imply a confirmed band.}",
        r"\label{tab:classified_images}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllllrr@{\hspace{12pt}}lllllrr}",
        r"\toprule",
        head + " & " + head + r" \\",
        r"\midrule",
    ]
    lines += [f"{a} & {b} \\\\" for a, b in zip(left, right)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT}  ({n} images)")
    print(df.groupby(["flyby", "label"]).size().unstack(fill_value=0))


def summary(df: pd.DataFrame) -> None:
    """Per-flyby theta* and FWHM of the positive detections (unrounded input)."""
    pos = df[df.label == "positive"]
    # Saturn equinox 2009-08-11: 098EN and 106EN precede it
    season = {"098EN": "pre", "106EN": "pre"}
    lines = [
        r"\begin{table}",
        r"\caption{Band orientation $\theta^\star$ (strongest peak of $E_{\mathrm{diff}}$) and "
        r"its FWHM for the " + f"{len(pos)}" + r" positive detections, summarised per flyby "
        r"(median and range). Values for each individual detection are listed in "
        r"Table~\ref{tab:classified_images}.}",
        r"\label{tab:orientation_summary}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{$\theta^\star$ ($\degree$)} & \multicolumn{2}{c}{FWHM ($\degree$)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Flyby & Equinox & $N$ & Median & Range & Median & Range \\",
        r"\midrule",
    ]
    for fly, g in pos.groupby("flyby"):
        lines.append(
            f"{fly} & {season.get(fly, 'post')} & {len(g)} & {g.theta.median():.1f} "
            f"& {g.theta.min():.1f}--{g.theta.max():.1f} & {g.fwhm.median():.0f} "
            f"& {g.fwhm.min():.0f}--{g.fwhm.max():.0f} \\\\")
    lines += [
        r"\midrule",
        f"All & & {len(pos)} & -- & {pos.theta.min():.1f}--{pos.theta.max():.1f} "
        f"& {pos.fwhm.median():.0f} & {pos.fwhm.min():.0f}--{pos.fwhm.max():.0f} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"  all positives: FWHM mean {pos.fwhm.mean():.1f}, median {pos.fwhm.median():.1f}")


if __name__ == "__main__":
    main()
    summary(load())
