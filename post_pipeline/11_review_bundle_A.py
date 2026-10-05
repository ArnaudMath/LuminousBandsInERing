#!/usr/bin/env python3
"""
11_review_bundle_A.py
─────────────────────
Interactive reviewer for stage-08/08b Bundle A images.

Click cycle per image:
  unselected (gray) -> positive (green) -> doubtful (orange) -> unselected

Selections are persisted and reloaded on the next run.

Usage
-----
    python post_pipeline/11_review_bundle_A.py
    python post_pipeline/11_review_bundle_A.py --data-root /path/to/mnt-storage
    python post_pipeline/11_review_bundle_A.py --summary-only-ok
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

N_COLS = 2
N_ROWS = 2
IMAGES_PER_PAGE = 2

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "mnt-storage"

STATE_NONE = 0
STATE_POSITIVE = 1
STATE_DOUBTFUL = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help="Pipeline data root (default: %(default)s)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help=(
            "Path to selection JSON. "
            "Default: <data-root>/post_pipeline/11_bundle_A_review_labels.json"
        ),
    )
    parser.add_argument(
        "--summary-only-ok",
        action="store_true",
        help="If summary.parquet exists, include only rows with status == 'ok'.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=900,
        help="Thumbnail width in pixels for each Bundle A image (default: %(default)s)",
    )
    parser.add_argument(
        "--rb-scale",
        type=float,
        default=0.55,
        help="Relative size of redblue image inside its tile (default: %(default)s)",
    )
    return parser.parse_args()


def load_records(pipeline_dir: Path, only_ok_from_summary: bool) -> tuple[pd.DataFrame, dict[str, dict[str, float | str | None]]]:
    summary_parquet = pipeline_dir / "summary.parquet"
    cisscal_dir = pipeline_dir.parent / "cisscal_output"

    summary_df: pd.DataFrame | None = None
    if summary_parquet.exists():
        try:
            summary_df = pd.read_parquet(summary_parquet)
        except Exception as exc:
            print(f"WARNING: could not read {summary_parquet}: {exc}")

    allowed_opusids: set[str] | None = None
    if only_ok_from_summary and summary_df is not None and {"opusid", "status"}.issubset(summary_df.columns):
        allowed_opusids = set(summary_df.loc[summary_df["status"] == "ok", "opusid"].astype(str))
        print(f"Using only status=='ok' from summary.parquet: {len(allowed_opusids)} opusids")

    records: list[dict[str, object]] = []
    for d in sorted(pipeline_dir.iterdir()):
        if not d.is_dir():
            continue
        opusid = d.name
        if allowed_opusids is not None and opusid not in allowed_opusids:
            continue

        bundle_a = d / "bundle_A.png"
        if not bundle_a.exists():
            continue

        metrics_json = d / "metrics.json"
        metrics: dict[str, object] = {}
        if metrics_json.exists():
            try:
                metrics = json.loads(metrics_json.read_text())
            except Exception:
                metrics = {}

        score = metrics.get("detection_score")
        status = metrics.get("status")

        records.append(
            {
                "opusid": opusid,
                "bundle_a_path": bundle_a,
                "redblue_path": (cisscal_dir / opusid / "redblue.png") if cisscal_dir.exists() else None,
                "metrics_path": metrics_json if metrics_json.exists() else None,
                "score": float(score) if isinstance(score, (int, float)) else np.nan,
                "status": str(status) if status is not None else None,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df, {}

    meta_by_opusid = (
        df.drop_duplicates(subset=["opusid"], keep="first")
        .set_index("opusid")[["score", "status"]]
        .to_dict(orient="index")
    )
    return df.reset_index(drop=True), meta_by_opusid


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    pipeline_dir = data_root / "pipeline_output"
    out_json = args.out_json.resolve() if args.out_json else (data_root / "post_pipeline" / "11_bundle_A_review_labels.json")

    if not pipeline_dir.exists():
        raise SystemExit(f"ERROR: pipeline_output directory not found: {pipeline_dir}")

    print("Scanning Bundle A images …")
    all_df, meta_by_opusid = load_records(pipeline_dir, args.summary_only_ok)
    if all_df.empty:
        raise SystemExit("No Bundle A images found in pipeline_output/*/bundle_A.png")

    print(f"Found {len(all_df)} Bundle A images")
    n_pages = (len(all_df) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE

    label_by_opusid: dict[str, int] = {}
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            selected = set(existing.get("selected_opusids", existing.get("flagged_opusids", [])))
            doubtful = set(existing.get("doubtful_opusids", []))
            for opusid in selected:
                label_by_opusid[str(opusid)] = STATE_POSITIVE
            for opusid in doubtful:
                if str(opusid) not in label_by_opusid:
                    label_by_opusid[str(opusid)] = STATE_DOUBTFUL
            print(f"Loaded labels from {out_json.name}")
        except Exception as exc:
            print(f"WARNING: could not parse existing JSON ({out_json}): {exc}")

    state = {"page": 0}
    thumb_cache: dict[str, np.ndarray] = {}
    ax_to_idx: dict[plt.Axes, int] = {}

    def get_thumb(img_path: Path, width: int = args.thumb_width) -> np.ndarray:
        key = str(img_path)
        if key in thumb_cache:
            return thumb_cache[key]

        img = Image.open(img_path)
        w, h = img.size
        if w <= 0 or h <= 0:
            thumb = np.zeros((200, 200, 3), dtype=np.uint8)
            thumb_cache[key] = thumb
            return thumb

        scale = width / float(w)
        new_h = max(1, int(round(h * scale)))
        thumb = np.array(img.resize((width, new_h), Image.LANCZOS))
        thumb_cache[key] = thumb
        return thumb

    def make_redblue_canvas(rb_img: np.ndarray, scale: float) -> np.ndarray:
        scale = max(0.2, min(1.0, float(scale)))
        if rb_img.ndim == 2:
            rb_img = np.stack([rb_img, rb_img, rb_img], axis=-1)
        if rb_img.shape[-1] == 4:
            rb_img = rb_img[..., :3]

        h, w = rb_img.shape[:2]
        side = max(h, w, 512)
        canvas = np.zeros((side, side, 3), dtype=rb_img.dtype)

        target = int(side * scale)
        fac = min(target / max(w, 1), target / max(h, 1))
        new_w = max(1, int(round(w * fac)))
        new_h = max(1, int(round(h * fac)))

        rb_pil = Image.fromarray(rb_img)
        rb_small = np.array(rb_pil.resize((new_w, new_h), Image.LANCZOS))

        y0 = (side - new_h) // 2
        x0 = (side - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = rb_small
        return canvas

    def _split_sets() -> tuple[set[str], set[str]]:
        positive = {k for k, v in label_by_opusid.items() if v == STATE_POSITIVE}
        doubtful = {k for k, v in label_by_opusid.items() if v == STATE_DOUBTFUL}
        return positive, doubtful

    def save_selection() -> None:
        positive_ids, doubtful_ids = _split_sets()
        selected_list = sorted(positive_ids)
        doubtful_list = sorted(doubtful_ids)
        all_labeled = sorted(selected_list + doubtful_list)

        missing_ids = [o for o in all_labeled if o not in meta_by_opusid]

        details = []
        for o in all_labeled:
            meta = meta_by_opusid.get(o)
            label = "positive" if o in positive_ids else "doubtful"
            if meta is None:
                details.append(
                    {
                        "opusid": o,
                        "label": label,
                        "score": None,
                        "status": None,
                        "missing_in_current_scan": True,
                    }
                )
            else:
                score_val = meta.get("score")
                details.append(
                    {
                        "opusid": o,
                        "label": label,
                        "score": float(score_val) if isinstance(score_val, (int, float)) and not np.isnan(score_val) else None,
                        "status": meta.get("status"),
                    }
                )

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "selection_meaning": "positive_luminous_band",
            "doubtful_meaning": "doubtful_luminous_band",
            "n_selected": len(selected_list),
            "n_doubtful": len(doubtful_list),
            "n_labeled": len(all_labeled),
            "n_total": int(len(all_df)),
            "n_missing_in_current_scan": len(missing_ids),
            "selected_opusids": selected_list,
            "doubtful_opusids": doubtful_list,
            "flagged_opusids": selected_list,
            "details": details,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved {len(selected_list)} positive + {len(doubtful_list)} doubtful → {out_json}")

    fig, axes = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(N_COLS * 7.4, N_ROWS * 4.8 + 0.8),
        facecolor="#1e1e1e",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08, wspace=0.05, hspace=0.25)

    title_text = fig.text(0.5, 0.97, "", ha="center", va="top", color="white", fontsize=11, weight="bold")
    count_text = fig.text(0.5, 0.935, "", ha="center", va="top", color="#dddddd", fontsize=9)

    ax_prev = fig.add_axes([0.02, 0.01, 0.12, 0.05])
    ax_next = fig.add_axes([0.16, 0.01, 0.12, 0.05])
    ax_save = fig.add_axes([0.44, 0.01, 0.14, 0.05])
    btn_prev = MplButton(ax_prev, "← Prev", color="#333", hovercolor="#555")
    btn_next = MplButton(ax_next, "Next →", color="#333", hovercolor="#555")
    btn_save = MplButton(ax_save, "Save", color="#1f7a34", hovercolor="#2b9a45")
    for b in (btn_prev, btn_next, btn_save):
        b.label.set_color("white")

    def render_page() -> None:
        ax_to_idx.clear()
        page = state["page"]
        start = page * IMAGES_PER_PAGE
        end = min(start + IMAGES_PER_PAGE, len(all_df))
        positive_ids, doubtful_ids = _split_sets()

        title_text.set_text(f"Bundle A + redblue review  |  Page {page + 1}/{n_pages}  ({start + 1}–{end} of {len(all_df)})")
        count_text.set_text(f"Positive: {len(positive_ids)}   |   Doubtful: {len(doubtful_ids)}")

        # Row 0 -> image start+0, Row 1 -> image start+1
        for row_i in range(N_ROWS):
            for col_i in range(N_COLS):
                ax = axes[row_i, col_i]
                ax.cla()
                ax.set_facecolor("#1e1e1e")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                df_idx = start + row_i
                if df_idx >= end:
                    continue

                row = all_df.iloc[df_idx]
                opusid = str(row["opusid"])
                label_state = label_by_opusid.get(opusid, STATE_NONE)

                if label_state == STATE_POSITIVE:
                    border_color = "#22cc55"
                    text_color = "#9cff9c"
                    marker = "✓ "
                elif label_state == STATE_DOUBTFUL:
                    border_color = "#ff9800"
                    text_color = "#ffcc80"
                    marker = "? "
                else:
                    border_color = "#555555"
                    text_color = "#cccccc"
                    marker = ""

                if col_i == 0:
                    # Bundle A
                    try:
                        img = get_thumb(Path(row["bundle_a_path"]))
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, "ERR", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=16)
                    score = row["score"]
                    score_str = "score=?" if pd.isna(score) else f"score={float(score):.3f}"
                    status = row["status"] if row["status"] is not None else "status=?"
                    ax.set_title(
                        f"{marker}{opusid}  [Bundle A]\n{score_str}  {status}",
                        fontsize=8,
                        color=text_color,
                        pad=3,
                    )
                else:
                    # redblue (scaled down inside tile)
                    rb_path = row["redblue_path"]
                    if rb_path is not None and Path(rb_path).exists():
                        try:
                            rb_img = get_thumb(Path(rb_path), width=max(240, args.thumb_width // 2))
                            rb_canvas = make_redblue_canvas(rb_img, args.rb_scale)
                            ax.imshow(rb_canvas)
                        except Exception:
                            ax.text(0.5, 0.5, "ERR", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=16)
                    else:
                        ax.text(0.5, 0.5, "redblue\nmissing", transform=ax.transAxes,
                                ha="center", va="center", color="#bbbbbb", fontsize=10)
                    ax.set_title(
                        f"{marker}{opusid}  [redblue]",
                        fontsize=8,
                        color=text_color,
                        pad=3,
                    )

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(4 if label_state != STATE_NONE else 0.8)

                ax_to_idx[ax] = df_idx

        btn_prev.ax.set_visible(page > 0)
        btn_next.ax.set_visible(page < n_pages - 1)
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is None or event.inaxes not in ax_to_idx:
            return
        df_idx = ax_to_idx[event.inaxes]
        opusid = str(all_df.iloc[df_idx]["opusid"])
        current = label_by_opusid.get(opusid, STATE_NONE)
        nxt = (current + 1) % 3
        if nxt == STATE_NONE:
            label_by_opusid.pop(opusid, None)
        else:
            label_by_opusid[opusid] = nxt
        render_page()

    def on_key(event) -> None:
        if event.key in ("right", "n"):
            if state["page"] < n_pages - 1:
                state["page"] += 1
                render_page()
        elif event.key in ("left", "p"):
            if state["page"] > 0:
                state["page"] -= 1
                render_page()
        elif event.key in ("s", "S"):
            save_selection()

    def on_prev(_) -> None:
        if state["page"] > 0:
            state["page"] -= 1
            render_page()

    def on_next(_) -> None:
        if state["page"] < n_pages - 1:
            state["page"] += 1
            render_page()

    def on_save(_) -> None:
        save_selection()

    def on_close(_) -> None:
        save_selection()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_save.on_clicked(on_save)

    print("\nControls:")
    print("  Click image   → cycle: unselected -> positive (green) -> doubtful (orange) -> unselected")
    print("  Layout        → per row: Bundle A (left) + redblue (right)")
    print("  ← / → or Prev/Next → navigate pages")
    print(f"  S key / Save  → save to {out_json}")
    print("  Close window  → auto-save\n")

    render_page()
    plt.show()


if __name__ == "__main__":
    main()
