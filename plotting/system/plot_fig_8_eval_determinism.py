"""Figure 8: within-seed spread of downstream scores across GPU counts."""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import plotcolours
from plotting.common.common import IMG_TYPES, PAPER_DATA, init_plot, save_figure, tex

EVAL_DATA = PAPER_DATA / "figure_8" / "downstream_evals.json"
PANELS = [("Core_v1", "a", "DCLM CORE"), ("FineWeb", "b", "FineWeb")]
GPUS = [8, 16, 32, 64]

ZEPHON_SLOT, NONDET_SLOT = 3, 0
ZEPHON_COLOR = plotcolours.colours[ZEPHON_SLOT]
NONDET_COLOR = plotcolours.colours[NONDET_SLOT]
ZEPHON_HATCH = plotcolours.hatches[ZEPHON_SLOT]
NONDET_HATCH = plotcolours.hatches[NONDET_SLOT]
NONDET_LABEL = "Standard"

# Fixed canvases in inches, not a tight crop: both panels need the same height
# and the same plot-area width to sit side by side on one scale.
P = dict(
    plot_area=2.95,
    gutter=1.05,
    margin=0.16,
    row_h=0.33,
    ticks_in=0.30,
    top_in=0.14,
    f_label=12,
    f_tick=11,
    f_note=9.0,
)

# The manuscript sets this once under both SVG panels; only previews draw it.
XLABEL = "Spread across GPU counts (pp)"


def load_evals() -> dict:
    if not EVAL_DATA.exists():
        raise FileNotFoundError(f"Frozen eval data not found: {EVAL_DATA}")
    return json.loads(EVAL_DATA.read_text())


def loader_spreads(data: dict, key: str, loader: str, seeds: list[str]) -> list[float]:
    """One max-min spread per seed."""
    out = []
    for sd in seeds:
        arms = data["seeds"][sd]["aggregates"][key]["arms"][loader]
        v = [arms[str(g)] for g in GPUS]
        out.append(max(v) - min(v))
    return out


def spreads(
    data: dict, key: str, seeds: list[str], combine: bool = False
) -> list[tuple[str, float, str, str]]:
    """Rows top to bottom as (label, spread, colour, hatch).

    Spreads are taken within each seed; seeds sit ~1 pp apart, so pooling them
    would measure the seed change. ``combine`` averages the per-seed spreads.
    """
    rows = []
    tag = (lambda sd: "") if (len(seeds) == 1 or combine) else (lambda sd: f" (s{sd})")
    for loader, label, colour, hatch in (
        ("nondet", NONDET_LABEL, NONDET_COLOR, NONDET_HATCH),
        ("zephon", "Zephon", ZEPHON_COLOR, ZEPHON_HATCH),
    ):
        vals = loader_spreads(data, key, loader, seeds)
        if combine:
            rows.append((label, statistics.mean(vals), colour, hatch))
        else:
            for sd, sp in zip(seeds, vals, strict=True):
                rows.append((f"{label}{tag(sd)}", sp, colour, hatch))
    return rows


def eval_noise(data: dict, key: str, seed: str) -> float:
    """Spread of repeated evaluations of one checkpoint."""
    rp = data["seeds"][seed]["aggregates"][key]["floor_repeats"]
    return max(rp) - min(rp)


def _plot_single(rows: list, calib: list, show_rowlabels: bool) -> plt.Figure:
    # +3 counts the "\z " prefix, which matplotlib measures but LaTeX drops.
    longest = max((len(lbl) for lbl, _v, _c, _h in rows), default=0) + 3
    gutter = max(P["gutter"], 0.082 * longest) if show_rowlabels else 0.0
    fig_w = P["margin"] * 2 + gutter + P["plot_area"]
    fig_h = P["ticks_in"] + P["top_in"] + len(rows) * P["row_h"]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_position(
        [
            (P["margin"] + gutter) / fig_w,
            P["ticks_in"] / fig_h,
            P["plot_area"] / fig_w,
            1 - (P["ticks_in"] + P["top_in"]) / fig_h,
        ]
    )

    rung_v, _name = calib[0]
    top = max(max(v for _l, v, _c, _h in rows), rung_v) * 1.30
    # One column of values, right of the longest bar.
    label_x = max(v for _l, v, _c, _h in rows) + top * 0.02
    for i, (_label, val, colour, hatch) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(
            [y],
            [val],
            height=0.6,
            color=colour,
            edgecolor="black",
            hatch=hatch,
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        # Always labelled: some bars are too short to read.
        ax.text(label_x, y, f"{val:.3f}", ha="left", va="center", fontsize=P["f_note"], zorder=5)

    ax.axvline(rung_v, color="black", lw=1.2, linestyle=(0, (5, 3)), zorder=4)
    ax.text(
        rung_v,
        len(rows) - 0.42,
        tex(f"{rung_v:g} pp"),
        ha="center",
        va="bottom",
        fontsize=P["f_note"],
    )

    ax.set_xlim(0, top)
    ax.set_ylim(-0.55, len(rows) - 0.15)
    ax.set_yticks(range(len(rows)))
    if show_rowlabels:
        ax.set_yticklabels([tex(lbl) for lbl, _v, _c, _h in reversed(rows)], fontsize=P["f_tick"])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", labelsize=P["f_tick"], length=0)
    # Wide pad: LaTeX re-renders the tick labels larger.
    ax.tick_params(axis="x", labelsize=P["f_tick"], length=0, pad=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", which="major", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    return fig


def _save(fig: plt.Figure, stem: Path):
    for ext in IMG_TYPES:
        if ext == ".svg":
            save_figure(fig, f"{stem}{ext}", dpi=300)
            continue
        ax = fig.axes[0]
        ax.set_xlabel(XLABEL, fontsize=P["f_label"])
        save_figure(fig, f"{stem}{ext}", dpi=300, bbox_inches="tight")
        ax.set_xlabel("")
    plt.close(fig)


def plot_range(data: dict, plot_dir: Path, seeds: list[str], combine: bool = False):
    plot_dir.mkdir(parents=True, exist_ok=True)
    init_plot()
    for idx, (key, letter, _name) in enumerate(PANELS):
        fig = _plot_single(
            spreads(data, key, seeds, combine), data["calibration"][key], show_rowlabels=idx == 0
        )
        _save(fig, plot_dir / f"fig_8{letter}_{key.lower()}")

    print("\nSpreads (pp):")
    for key, _l, name in PANELS:
        rung = data["calibration"][key][0]
        print(f"  {name}  (rung {rung[0]:g} pp, {rung[1]})")
        for label, val, _c, _h in spreads(data, key, seeds, combine):
            print(f"    {label:22s} {val:.4f}   {val / rung[0]:5.2f}x the rung")
        for sd in seeds:
            noise = eval_noise(data, key, sd)
            arms = data["seeds"][sd]["aggregates"][key]["arms"]["zephon"]
            ze = [arms[str(g)] for g in GPUS]
            print(
                f"    {f'eval noise s{sd} (not drawn)':22s} {noise:.4f}   "
                f"Zephon spread is {(max(ze) - min(ze)) / noise:.2f}x it"
            )
        if combine:
            # Alternative: spread of the per-GPU means across seeds.
            for loader, label in (("nondet", NONDET_LABEL), ("zephon", "Zephon")):
                per = [
                    statistics.mean(
                        data["seeds"][sd]["aggregates"][key]["arms"][loader][str(g)] for sd in seeds
                    )
                    for g in GPUS
                ]
                print(
                    f"    {f'[alt] {label} spread of per-GPU means':38s} {max(per) - min(per):.4f}"
                )
