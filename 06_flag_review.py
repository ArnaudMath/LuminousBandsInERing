#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 06 — Interactive residual review
--------------------------------------
Interactive image reviewer for the CLEAR-subtracted redblue PNGs.
Click an image to flag it (red border). Click again to unflag.
Navigate with arrow keys or on-screen buttons.
Press S or close the window to save the flagged IDs.

Usage
-----
  python 06_flag_review.py              # all images
  python 06_flag_review.py --filter IR1 # only IR1 images

Output → data/post_pipeline/06_flagged_pairs.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button as MplButton
from PIL import Image

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent / "data"
CISSCAL_DIR  = DATA_DIR / "cisscal_output"
PAIRS_FILE   = DATA_DIR / "inferred_sets" / "pairs.parquet"
OUT_JSON     = DATA_DIR / "post_pipeline" / "06_flagged_pairs.json"

N_COLS   = 6
N_ROWS   = 4
PER_PAGE = N_COLS * N_ROWS

# ─────────────────────────── load data ────────────────────────────────────────

print("Loading pairs metadata …")
pairs_df = pd.read_parquet(PAIRS_FILE)

print("Scanning cisscal_output …")
records = []
for d in sorted(CISSCAL_DIR.iterdir()):
    if not d.is_dir():
        continue
    rb = d / "redblue.png"
    if not rb.exists():
        continue
    opusid = d.name
    row    = pairs_df[pairs_df["science_opusid"] == opusid]
    filt   = row["science_filter"].iloc[0] if not row.empty else "?"
    dt     = float(row["dt_s"].iloc[0])    if not row.empty else float("nan")
    records.append({"opusid": opusid, "rb_path": rb, "filter": filt, "dt_s": dt})

all_df = pd.DataFrame(records)
print(f"Found {len(all_df)} images  |  filters: {sorted(all_df['filter'].unique())}")

filter_arg = None
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--filter" and i + 1 < len(sys.argv[1:]):
        filter_arg = sys.argv[i + 2]

if filter_arg:
    all_df = all_df[all_df["filter"] == filter_arg].reset_index(drop=True)
    print(f"Filtered to {filter_arg}: {len(all_df)} images")

n_pages = (len(all_df) + PER_PAGE - 1) // PER_PAGE

meta_by_opusid = (
    all_df.drop_duplicates(subset=["opusid"], keep="first")
    .set_index("opusid")[["filter", "dt_s"]]
    .to_dict(orient="index")
)

# ─────────────────────────── state ────────────────────────────────────────────

flagged_ids = set()
if OUT_JSON.exists():
    existing    = json.loads(OUT_JSON.read_text())
    flagged_ids = set(existing.get("flagged_opusids", []))
    print(f"Loaded {len(flagged_ids)} previously flagged IDs")

state     = {"page": 0}
thumb_cache = {}
ax_to_idx   = {}


def get_thumb(rb_path: Path, size: int = 128) -> np.ndarray:
    key = str(rb_path)
    if key not in thumb_cache:
        thumb_cache[key] = np.array(Image.open(rb_path).resize((size, size), Image.LANCZOS))
    return thumb_cache[key]


def save_flags() -> None:
    flagged_list = sorted(flagged_ids)
    details      = [{"opusid": o, **{k: v for k, v in (meta_by_opusid.get(o) or {}).items()}}
                    for o in flagged_list]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "n_flagged":       len(flagged_list),
        "n_total":         len(all_df),
        "flagged_opusids": flagged_list,
        "details":         details,
    }, indent=2))
    print(f"Saved {len(flagged_list)} flagged pairs → {OUT_JSON}")

# ─────────────────────────── figure ───────────────────────────────────────────

fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(N_COLS * 2.4, N_ROWS * 2.6 + 0.6),
                          facecolor="#1e1e1e")
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07,
                    wspace=0.05, hspace=0.35)

title_text = fig.text(0.5, 0.97, "", ha="center", va="top",
                       color="white", fontsize=11, weight="bold")
flag_text  = fig.text(0.5, 0.93, "", ha="center", va="top",
                       color="#aaffaa", fontsize=9)

ax_prev = fig.add_axes([0.02, 0.01, 0.12, 0.04])
ax_next = fig.add_axes([0.16, 0.01, 0.12, 0.04])
ax_save = fig.add_axes([0.44, 0.01, 0.12, 0.04])
btn_prev = MplButton(ax_prev, "← Prev", color="#333", hovercolor="#555")
btn_next = MplButton(ax_next, "Next →", color="#333", hovercolor="#555")
btn_save = MplButton(ax_save, "Save",    color="#2a5", hovercolor="#3b6")
for b in (btn_prev, btn_next, btn_save):
    b.label.set_color("white")


def render_page() -> None:
    ax_to_idx.clear()
    page  = state["page"]
    start = page * PER_PAGE
    end   = min(start + PER_PAGE, len(all_df))
    title_text.set_text(
        f"Page {page + 1}/{n_pages}  ({start + 1}–{end} of {len(all_df)})"
        + (f"  filter: {filter_arg}" if filter_arg else "")
    )
    flag_text.set_text(f"Flagged: {len(flagged_ids)}")

    for i, ax in enumerate(axes.flatten()):
        ax.cla()
        ax.set_facecolor("#1e1e1e")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        df_idx = start + i
        if df_idx >= end:
            continue

        row       = all_df.iloc[df_idx]
        opusid    = row["opusid"]
        is_flagged = opusid in flagged_ids

        ax.imshow(get_thumb(row["rb_path"]))

        border_color = "#ff4444" if is_flagged else "#444444"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_flagged else 0.5)

        ax.set_title(
            f"{'★ ' if is_flagged else ''}{opusid.split('-')[-1].upper()}\n"
            f"{row['filter']}  Δt={row['dt_s']:.0f}s",
            fontsize=6.5,
            color="#ff8888" if is_flagged else "#cccccc",
            pad=2,
        )
        ax_to_idx[ax] = df_idx

    btn_prev.ax.set_visible(page > 0)
    btn_next.ax.set_visible(page < n_pages - 1)
    fig.canvas.draw_idle()


def on_click(event) -> None:
    if event.inaxes not in ax_to_idx:
        return
    opusid = all_df.iloc[ax_to_idx[event.inaxes]]["opusid"]
    flagged_ids.discard(opusid) if opusid in flagged_ids else flagged_ids.add(opusid)
    render_page()


def on_key(event) -> None:
    if event.key in ("right", "n") and state["page"] < n_pages - 1:
        state["page"] += 1;  render_page()
    elif event.key in ("left", "p") and state["page"] > 0:
        state["page"] -= 1;  render_page()
    elif event.key in ("s", "S"):
        save_flags()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event",    on_key)
fig.canvas.mpl_connect("close_event",        lambda _: save_flags())
btn_prev.on_clicked(lambda _: (state.update(page=state["page"] - 1), render_page()))
btn_next.on_clicked(lambda _: (state.update(page=state["page"] + 1), render_page()))
btn_save.on_clicked(lambda _: save_flags())

print("\nControls:")
print("  Click image        → toggle flag (red border = flagged)")
print("  ← / → or buttons  → navigate pages")
print("  S key / Save       → save flags")
print("  Close window       → auto-save\n")

render_page()
plt.show()
