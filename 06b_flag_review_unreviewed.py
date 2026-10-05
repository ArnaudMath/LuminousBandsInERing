"""
06b_flag_review_unreviewed.py
─────────────────────────────
Copy of 06_flag_review.py for the 51 pairs that reached detection without the
visual quality review (45 early-run pairs of 120EN, 141EN, 142EN, 156EN and
6 pairs of 098EN added after the review; see release/zenodo_data/README.md).
Their April residual PNGs are no longer stored, so the April bundle_A.png
diagnostics are shown instead (the residual panels are the top row).
Same exclusion criteria as step 06: disc fills/dominates the field, incoherent
residual from a large time separation, surviving sensor/readout artefacts.
06_flagged_pairs.json is not touched; flags go to 06b_flagged_pairs.json.

Click an image to flag it (red border). Click again to unflag.
Navigate with arrow keys or on-screen buttons.
Press S or close the window to save flagged IDs to 06_flagged_pairs.json.

Usage:
    python 06b_flag_review_unreviewed.py
"""

import sys, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # works headless-free; fall back to Qt if Tk absent
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT        = Path("/home/arnaudm/projects/LuminousBandsInERing")
PAIRS_CSV   = ROOT / "release/zenodo_data/image_pairs.csv"
BUNDLE_DIR  = ROOT / "pipeline_output"
OUT_JSON    = ROOT / "pre_pipeline/06b_flagged_pairs.json"

N_COLS   = 1
N_ROWS   = 2
PER_PAGE = N_COLS * N_ROWS     # 24 per page → fast to render

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading the pairs that skipped the visual QC review …")
pairs_df = pd.read_csv(PAIRS_CSV)
todo = pairs_df[pairs_df.detection_run & ~pairs_df.qc_flagged & ~pairs_df.in_clean_pairs_573]
records = []
for _, row in todo.sort_values("science_time_utc").iterrows():
    rb = BUNDLE_DIR / row.science_opusid / "bundle_A.png"
    if rb.exists():
        records.append({"opusid": row.science_opusid, "rb_path": rb,
                         "filter": f"{row.science_filter} {row.flyby}", "dt_s": row.clear_science_dt_s})

all_df = pd.DataFrame(records)
print(f"Found {len(all_df)} images  |  filters: {sorted(all_df['filter'].unique())}")

# Fast metadata lookup for save/export (one row per opusid)
meta_by_opusid = (
    all_df.drop_duplicates(subset=["opusid"], keep="first")
    .set_index("opusid")[["filter", "dt_s"]]
    .to_dict(orient="index")
)

# Optional filter argument
filter_arg = None
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--filter" and i + 1 < len(sys.argv[1:]):
        filter_arg = sys.argv[i + 2]

if filter_arg:
    all_df = all_df[all_df["filter"] == filter_arg].reset_index(drop=True)
    print(f"Filtered to {filter_arg}: {len(all_df)} images")

if all_df.empty:
    print("No images found. Check CISSCAL_DIR and run step 4+5 first.")
    sys.exit(1)

n_pages = (len(all_df) + PER_PAGE - 1) // PER_PAGE

# ── State ─────────────────────────────────────────────────────────────────────
flagged_ids = set()
# Load existing flags if json exists
if OUT_JSON.exists():
    try:
        existing = json.loads(OUT_JSON.read_text())
        flagged_ids = set(existing.get("flagged_opusids", []))
        print(f"Loaded {len(flagged_ids)} previously flagged IDs from {OUT_JSON.name}")
    except Exception:
        pass

state = {"page": 0}

# ── Thumbnail cache (load on demand per page) ──────────────────────────────────
thumb_cache = {}

def get_thumb(rb_path: Path, size: int = 1400) -> np.ndarray:
    key = str(rb_path)
    if key not in thumb_cache:
        img = Image.open(rb_path).convert("RGB")
        img.thumbnail((size, size))
        thumb_cache[key] = np.array(img)
    return thumb_cache[key]

# ── Save ──────────────────────────────────────────────────────────────────────
def save_flags():
    flagged_list = sorted(flagged_ids)
    missing_ids = [o for o in flagged_list if o not in meta_by_opusid]

    details = []
    for o in flagged_list:
        meta = meta_by_opusid.get(o)
        if meta is None:
            details.append(
                {
                    "opusid": o,
                    "filter": None,
                    "dt_s": None,
                    "missing_in_current_scan": True,
                }
            )
        else:
            details.append(
                {
                    "opusid": o,
                    "filter": meta["filter"],
                    "dt_s": float(meta["dt_s"]),
                }
            )

    payload = {
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "n_flagged":       len(flagged_list),
        "n_total":         len(all_df),
        "n_missing_in_current_scan": len(missing_ids),
        "flagged_opusids": flagged_list,
        "details": details,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    if missing_ids:
        print(
            f"Saved {len(flagged_list)} flagged pairs ({len(missing_ids)} not in current scan) → {OUT_JSON}"
        )
    else:
        print(f"Saved {len(flagged_list)} flagged pairs → {OUT_JSON}")

# ── Render ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(14, 12),
                          facecolor="#1e1e1e")
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07,
                    wspace=0.05, hspace=0.35)

title_text = fig.text(0.5, 0.97, "", ha="center", va="top",
                       color="white", fontsize=11, weight="bold")
flag_text  = fig.text(0.5, 0.93, "", ha="center", va="top",
                       color="#aaffaa", fontsize=9)

# Nav buttons
ax_prev = fig.add_axes([0.02, 0.01, 0.12, 0.04])
ax_next = fig.add_axes([0.16, 0.01, 0.12, 0.04])
ax_save = fig.add_axes([0.44, 0.01, 0.12, 0.04])
from matplotlib.widgets import Button as MplButton
btn_prev = MplButton(ax_prev, "← Prev", color="#333", hovercolor="#555")
btn_next = MplButton(ax_next, "Next →", color="#333", hovercolor="#555")
btn_save = MplButton(ax_save, "Save", color="#2a5", hovercolor="#3b6")
for b in (btn_prev, btn_next, btn_save):
    b.label.set_color("white")

# Map from axes to dataframe index on current page
ax_to_idx = {}

def render_page():
    ax_to_idx.clear()
    page  = state["page"]
    start = page * PER_PAGE
    end   = min(start + PER_PAGE, len(all_df))
    page_rows = all_df.iloc[start:end]

    title_text.set_text(
        f"Page {page + 1} / {n_pages}  "
        f"({start + 1}–{end} of {len(all_df)})  "
        f"{'  |  filter: ' + filter_arg if filter_arg else ''}"
    )
    flag_text.set_text(f"Flagged: {len(flagged_ids)}")

    flat_axes = axes.flatten()
    for i, ax in enumerate(flat_axes):
        ax.cla()
        ax.set_facecolor("#1e1e1e")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        df_idx = start + i
        if df_idx >= end:
            continue

        row    = all_df.iloc[df_idx]
        opusid = row["opusid"]
        label  = opusid.split("-")[-1].upper()

        try:
            img = get_thumb(row["rb_path"])
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "ERR", transform=ax.transAxes,
                    ha="center", va="center", color="red")

        is_flagged = opusid in flagged_ids
        border_color = "#ff4444" if is_flagged else "#444444"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_flagged else 0.5)

        ax.set_title(
            f"{'★ ' if is_flagged else ''}{label}\n"
            f"{row['filter']}  Δt={row['dt_s']:.0f}s",
            fontsize=6.5,
            color="#ff8888" if is_flagged else "#cccccc",
            pad=2,
        )
        ax_to_idx[ax] = df_idx

    btn_prev.ax.set_visible(page > 0)
    btn_next.ax.set_visible(page < n_pages - 1)
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes is None or event.inaxes not in ax_to_idx:
        return
    df_idx = ax_to_idx[event.inaxes]
    opusid = all_df.iloc[df_idx]["opusid"]
    if opusid in flagged_ids:
        flagged_ids.discard(opusid)
    else:
        flagged_ids.add(opusid)
    render_page()

def on_key(event):
    if event.key in ("right", "n"):
        if state["page"] < n_pages - 1:
            state["page"] += 1
            render_page()
    elif event.key in ("left", "p"):
        if state["page"] > 0:
            state["page"] -= 1
            render_page()
    elif event.key in ("s", "S"):
        save_flags()

def on_prev(_):
    if state["page"] > 0:
        state["page"] -= 1
        render_page()

def on_next(_):
    if state["page"] < n_pages - 1:
        state["page"] += 1
        render_page()

def on_save(_):
    save_flags()

def on_close(_):
    save_flags()

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("close_event", on_close)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_save.on_clicked(on_save)

print(f"\nControls:")
print(f"  Click image  → toggle flag (red border = flagged)")
print(f"  ← / → arrows or Prev/Next buttons → navigate pages")
print(f"  S key or Save button → save to {OUT_JSON.name}")
print(f"  Close window → auto-saves\n")

render_page()
plt.show()
