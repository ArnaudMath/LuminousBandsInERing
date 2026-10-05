"""
make_contrast_trends_14b.py
────────────────────────────
Copy of make_contrast_trends.py that plots the normalised contrast
C = delta / (S_max + S_min) from post_pipeline/14b_contrast_table.csv
instead of delta. Layout, filters and fits are unchanged. The E ring LOS
column panel is dropped (the LOS model is not used in the article any more),
leaving phase angle, Cassini-Enceladus and Cassini-Saturn distance.
The original figure is not touched.

Output (new file names only):
    figure_scripts/output/contrast_trends_14b.png
    Thesis/icarus_template_used/figs/contrast_trends_14b.png

Usage:
    python figure_scripts/make_contrast_trends_14b.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
DATA_ROOT    = Path("/mnt/storage/projects/Thesis/LuminousBandsInERing")
TABLE_CSV    = DATA_ROOT / "post_pipeline" / "14b_contrast_table.csv"
GEO_CACHE    = DATA_ROOT / "post_pipeline" / "geometry_cache.csv"

OUT_DIR      = Path(__file__).parent / "output"
OUT_ARTICLE  = _ROOT / "Thesis/icarus_template_used/figs/contrast_trends_14b.png"


# ── Figure style (Icarus double column = 7.0 inch) ────────────────────────────
FIG_W   = 7.0
DPI     = 300
FS_LB   = 6      # axis labels
FS_TK   = 5      # tick labels
FS_OFF  = 4      # scientific notation offset text (1e-5)
FS_ROW  = 6      # row (filter) label
FS_COL  = 6.5    # column header

WL_COLORS = {
    338:  "#8b00ff",
    451:  "#0055ff",
    568:  "#55cc00",
    617:  "#ff2200",
    650:  "#ff2200",
    752:  "#bb0000",
    862:  "#770000",
    890:  "#440000",
    930:  "#440000",
}
DEFAULT_COLOR = "#444444"

XVARS = [
    ("phase_mid",         "Phase angle (°)"),
    ("dist_enceladus_km", "Cassini–Enceladus (km)"),
    ("dist_saturn_km",    "Cassini–Saturn (km)"),
]

MIN_N = 4


def load_data() -> pd.DataFrame:
    df_contrast = pd.read_csv(TABLE_CSV)
    geo = pd.read_csv(GEO_CACHE)
    df = df_contrast.merge(geo, on="opusid", how="left")

    df["phase_mid"]         = (df["phase_min"] + df["phase_max"]) / 2
    df["wavelength_mid_nm"] = (
        (df["wavelength1_um"] + df["wavelength2_um"]) / 2 * 1000
    ).round().astype("Int64")

    df["C_pct"] = 100.0 * df["C"]          # plotted quantity, in percent
    return df.dropna(subset=["C_pct"])


def main() -> None:
    df = load_data()

    wl_counts = df["wavelength_mid_nm"].value_counts()
    wl_groups = sorted(wl for wl in wl_counts.index if wl_counts[wl] >= MIN_N)
    n_wl  = len(wl_groups)
    n_var = len(XVARS)

    panel_h = 1.35
    fig_h   = panel_h * n_wl + 0.25   # extra for column headers
    fig, axes = plt.subplots(
        n_wl, n_var,
        figsize=(FIG_W, fig_h),
        gridspec_kw={"hspace": 0.30, "wspace": 0.25},
    )
    fig.patch.set_facecolor("white")
    if n_wl == 1:
        axes = axes[np.newaxis, :]

    y_max = df["C_pct"].max() * 1.1

    for col_i, (xcol, xlabel) in enumerate(XVARS):
        axes[0, col_i].set_title(xlabel, fontsize=FS_COL, pad=3)

    for row_i, wl in enumerate(wl_groups):
        grp   = df[df["wavelength_mid_nm"] == wl]
        filt  = grp["COISSfilter"].iloc[0] if not grp.empty else "?"
        color = WL_COLORS.get(int(wl), DEFAULT_COLOR)

        for col_i, (xcol, xlabel) in enumerate(XVARS):
            ax  = axes[row_i, col_i]
            sub = grp.dropna(subset=[xcol, "C_pct"])

            ax.scatter(sub[xcol], sub["C_pct"],
                       c=[color], s=10, alpha=0.85, edgecolors="none", zorder=4)

            x = sub[xcol].to_numpy(dtype=float)
            y = sub["C_pct"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 2 and x[ok].max() > x[ok].min():
                c = np.polyfit(x[ok], y[ok], 1)
                xf = np.linspace(x[ok].min(), x[ok].max(), 100)
                ax.plot(xf, np.polyval(c, xf), color="black", lw=0.8, ls="--", alpha=0.75)
                y_pred = np.polyval(c, x[ok])
                ss_res = np.sum((y[ok] - y_pred) ** 2)
                ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                ax.text(0.97, 0.95, f"$R^2={r2:.2f}$",
                        transform=ax.transAxes, fontsize=FS_TK,
                        ha="right", va="top", color="black")

            ax.set_ylim(0, y_max)
            ax.tick_params(labelsize=FS_TK, length=2, pad=1)
            ax.grid(True, alpha=0.2, linewidth=0.4)

            if col_i == 0:
                filt_label = filt.replace("CL1+", "").replace("+CL2", "").replace("CL2+", "")
                ax.set_ylabel(
                    f"{filt_label} ({wl} nm)\n$C$ (%)",
                    fontsize=FS_ROW, labelpad=2,
                )
            else:
                ax.set_yticklabels([])

            ax.yaxis.get_offset_text().set_fontsize(FS_OFF)

            if row_i == n_wl - 1:
                ax.set_xlabel(xlabel, fontsize=FS_LB, labelpad=2)
            else:
                ax.set_xticklabels([])


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for out in [OUT_DIR / "contrast_trends_14b.png", OUT_ARTICLE]:
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
