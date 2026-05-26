#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 09 — Interactive detection review
----------------------------------------
Interactive reviewer for the bundle_A diagnostic images produced by
step 07. Click an image to cycle its label: unselected → positive → doubtful.

Controls
--------
  Click image       → cycle: unselected (grey) → positive (green) → doubtful (orange)
  ← / → or buttons  → navigate pages
  S key / Save       → save labels to JSON
  Close window       → auto-save

Output → data/post_pipeline/09_detection_labels.json

Usage
-----
  python 09_review_detections.py
  python 09_review_detections.py --summary-only-ok
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.widgets import Button as MplButton

# ─────────────────────────── paths ────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent / "data"
PIPELINE_DIR = DATA_DIR / "pipeline_output"
CISSCAL_DIR  = DATA_DIR / "cisscal_output"
OUT_JSON     = DATA_DIR / "post_pipeline" / "09_detection_labels.json"

N_COLS        = 2
N_ROWS        = 2
IMAGES_PER_PAGE = 2  # one row per image: bundle_A (left) + redblue (right)

STATE_NONE     = 0
STATE_POSITIVE = 1
STATE_DOUBTFUL = 2

# ─────────────────────────── argument parsing ──────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-only-ok", action="store_true",
                    help="Include only images with status=='ok' in summary.parquet")
parser.add_argument("--thumb-width", type=int, default=900)
parser.add_argument("--rb-scale",    type=float, default=0.55)
args = parser.parse_args()

# ─────────────────────────── load records ─────────────────────────────────────

summary_parquet = PIPELINE_DIR / "summary.parquet"
allowed_opusids = None
if args.summary_only_ok and summary_parquet.exists():
    summary_df      = pd.read_parquet(summary_parquet)
    allowed_opusids = set(summary_df.loc[summary_df["status"] == "ok", "opusid"].astype(str))
    print(f"Filtering to status=='ok': {len(allowed_opusids)} opusids")

records = []
for d in sorted(PIPELINE_DIR.iterdir()):
    if not d.is_dir():
        continue
    opusid   = d.name
    bundle_a = d / "bundle_A.png"
    if not bundle_a.exists():
        continue
    if allowed_opusids is not None and opusid not in allowed_opusids:
        continue
    metrics = {}
    metrics_json = d / "metrics.json"
    if metrics_json.exists():
        metrics = json.loads(metrics_json.read_text())
    records.append({
        "opusid":        opusid,
        "bundle_a_path": bundle_a,
        "redblue_path":  CISSCAL_DIR / opusid / "redblue.png",
        "score":         float(metrics.get("detection_score", float("nan"))),
        "status":        metrics.get("status"),
    })

all_df  = pd.DataFrame(records)
n_pages = (len(all_df) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
print(f"Found {len(all_df)} Bundle A images  ({n_pages} pages)")

# ─────────────────────────── state ────────────────────────────────────────────

label_by_opusid: dict[str, int] = {}
if OUT_JSON.exists():
    existing = json.loads(OUT_JSON.read_text())
    for opusid in existing.get("positive_opusids", existing.get("selected_opusids", [])):
        label_by_opusid[str(opusid)] = STATE_POSITIVE
    for opusid in existing.get("doubtful_opusids", []):
        label_by_opusid.setdefault(str(opusid), STATE_DOUBTFUL)
    print(f"Loaded labels from {OUT_JSON.name}")

state       = {"page": 0}
thumb_cache = {}
ax_to_idx   = {}


def get_thumb(img_path: Path, width: int) -> np.ndarray:
    key = str(img_path)
    if key not in thumb_cache:
        img   = Image.open(img_path)
        scale = width / float(img.size[0])
        new_h = max(1, int(round(img.size[1] * scale)))
        thumb_cache[key] = np.array(img.resize((width, new_h), Image.LANCZOS))
    return thumb_cache[key]


def make_rb_canvas(rb_img: np.ndarray, scale: float) -> np.ndarray:
    if rb_img.ndim == 2:
        rb_img = np.stack([rb_img] * 3, axis=-1)
    if rb_img.shape[-1] == 4:
        rb_img = rb_img[..., :3]
    h, w    = rb_img.shape[:2]
    side    = max(h, w, 512)
    canvas  = np.zeros((side, side, 3), dtype=rb_img.dtype)
    fac     = min(side * scale / max(w, 1), side * scale / max(h, 1))
    new_w   = max(1, int(round(w * fac)))
    new_h   = max(1, int(round(h * fac)))
    rb_small = np.array(Image.fromarray(rb_img).resize((new_w, new_h), Image.LANCZOS))
    y0, x0  = (side - new_h) // 2, (side - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = rb_small
    return canvas


def save_labels() -> None:
    positive = sorted(k for k, v in label_by_opusid.items() if v == STATE_POSITIVE)
    doubtful = sorted(k for k, v in label_by_opusid.items() if v == STATE_DOUBTFUL)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "n_positive":      len(positive),
        "n_doubtful":      len(doubtful),
        "n_total":         len(all_df),
        "positive_opusids": positive,
        "doubtful_opusids": doubtful,
    }, indent=2))
    print(f"Saved {len(positive)} positive + {len(doubtful)} doubtful → {OUT_JSON}")

# ─────────────────────────── figure ───────────────────────────────────────────

fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(N_COLS * 7.4, N_ROWS * 4.8 + 0.8),
                          facecolor="#1e1e1e")
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08, wspace=0.05, hspace=0.25)

title_text = fig.text(0.5, 0.97, "", ha="center", va="top", color="white", fontsize=11, weight="bold")
count_text = fig.text(0.5, 0.935, "", ha="center", va="top", color="#dddddd", fontsize=9)

ax_prev = fig.add_axes([0.02, 0.01, 0.12, 0.05])
ax_next = fig.add_axes([0.16, 0.01, 0.12, 0.05])
ax_save = fig.add_axes([0.44, 0.01, 0.14, 0.05])
btn_prev = MplButton(ax_prev, "← Prev", color="#333", hovercolor="#555")
btn_next = MplButton(ax_next, "Next →", color="#333", hovercolor="#555")
btn_save = MplButton(ax_save, "Save",    color="#1f7a34", hovercolor="#2b9a45")
for b in (btn_prev, btn_next, btn_save):
    b.label.set_color("white")


def render_page() -> None:
    ax_to_idx.clear()
    page  = state["page"]
    start = page * IMAGES_PER_PAGE
    end   = min(start + IMAGES_PER_PAGE, len(all_df))
    n_pos = sum(1 for v in label_by_opusid.values() if v == STATE_POSITIVE)
    n_dbt = sum(1 for v in label_by_opusid.values() if v == STATE_DOUBTFUL)

    title_text.set_text(f"Detection review  |  Page {page + 1}/{n_pages}  ({start + 1}–{end} of {len(all_df)})")
    count_text.set_text(f"Positive: {n_pos}   |   Doubtful: {n_dbt}")

    for row_i in range(N_ROWS):
        for col_i in range(N_COLS):
            ax = axes[row_i, col_i]
            ax.cla()
            ax.set_facecolor("#1e1e1e")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            df_idx = start + row_i
            if df_idx >= end:
                continue

            row          = all_df.iloc[df_idx]
            opusid       = str(row["opusid"])
            label_state  = label_by_opusid.get(opusid, STATE_NONE)
            border_color = {STATE_POSITIVE: "#22cc55", STATE_DOUBTFUL: "#ff9800"}.get(label_state, "#555555")
            text_color   = {STATE_POSITIVE: "#9cff9c", STATE_DOUBTFUL: "#ffcc80"}.get(label_state, "#cccccc")
            marker       = {STATE_POSITIVE: "✓ ", STATE_DOUBTFUL: "? "}.get(label_state, "")

            if col_i == 0:
                ax.imshow(get_thumb(Path(row["bundle_a_path"]), args.thumb_width))
                score_str = "?" if pd.isna(row["score"]) else f"{float(row['score']):.3f}"
                ax.set_title(f"{marker}{opusid}  [Bundle A]\nscore={score_str}  {row['status']}",
                             fontsize=8, color=text_color, pad=3)
            else:
                rb_path = Path(row["redblue_path"])
                if rb_path.exists():
                    rb_img = get_thumb(rb_path, max(240, args.thumb_width // 2))
                    ax.imshow(make_rb_canvas(rb_img, args.rb_scale))
                else:
                    ax.text(0.5, 0.5, "redblue\nmissing", transform=ax.transAxes,
                            ha="center", va="center", color="#bbbbbb", fontsize=10)
                ax.set_title(f"{marker}{opusid}  [redblue]", fontsize=8, color=text_color, pad=3)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4 if label_state != STATE_NONE else 0.8)
            ax_to_idx[ax] = df_idx

    btn_prev.ax.set_visible(page > 0)
    btn_next.ax.set_visible(page < n_pages - 1)
    fig.canvas.draw_idle()


def on_click(event) -> None:
    if event.inaxes not in ax_to_idx:
        return
    opusid  = str(all_df.iloc[ax_to_idx[event.inaxes]]["opusid"])
    current = label_by_opusid.get(opusid, STATE_NONE)
    nxt     = (current + 1) % 3
    if nxt == STATE_NONE:
        label_by_opusid.pop(opusid, None)
    else:
        label_by_opusid[opusid] = nxt
    render_page()


def on_key(event) -> None:
    if event.key in ("right", "n") and state["page"] < n_pages - 1:
        state["page"] += 1;  render_page()
    elif event.key in ("left", "p") and state["page"] > 0:
        state["page"] -= 1;  render_page()
    elif event.key in ("s", "S"):
        save_labels()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event",    on_key)
fig.canvas.mpl_connect("close_event",        lambda _: save_labels())
btn_prev.on_clicked(lambda _: (state.update(page=state["page"] - 1), render_page()))
btn_next.on_clicked(lambda _: (state.update(page=state["page"] + 1), render_page()))
btn_save.on_clicked(lambda _: save_labels())

print("\nControls:")
print("  Click image        → cycle: unselected → positive (green) → doubtful (orange)")
print("  ← / → or buttons  → navigate pages")
print("  S key / Save       → save labels")
print("  Close window       → auto-save\n")

render_page()
plt.show()
