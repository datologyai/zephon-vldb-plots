"""Figure 7: per-DOP VLM pipeline throughput for the selected allocations."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import BoxStyle
from matplotlib.ticker import MaxNLocator

from plotting.common import plotcolours
from plotting.common.common import (
    DOUBLE_FIG_HEIGHT,
    PAPER_DATA,
    SINGLE_FIG_WIDTH,
    aggregate_seeds,
    hide_borders,
    init_plot,
    read_measurements,
    save_all,
    y_grid,
)

DATA_DIR = PAPER_DATA / "figure_7"

# Per-GPU rate the trainer needs; a loader below it stalls training.
STALL_FREE_TOKENS_PER_SEC = 16_900

# Label boxes get extra top padding: LaTeX sets the text higher in its box than
# matplotlib measures. Pads are fractions of the font size.
LABEL_PAD_X = 0.15
LABEL_PAD_BOTTOM = 0.11
LABEL_PAD_TOP = 0.33
_ROUND = BoxStyle("round", pad=LABEL_PAD_X)


def _label_boxstyle(x0, y0, width, height, mutation_size):
    y0 += (LABEL_PAD_X - LABEL_PAD_BOTTOM) * mutation_size
    height += (LABEL_PAD_TOP + LABEL_PAD_BOTTOM - 2 * LABEL_PAD_X) * mutation_size
    return _ROUND(x0, y0, width, height, mutation_size)


LABEL_BBOX = {"boxstyle": _label_boxstyle, "fc": "white", "ec": "none", "alpha": 0.85}

# Headroom above the tallest bar for the legend, before rounding to a tick.
Y_HEADROOM = 1.5
# Bar label offsets from the bar top, as fractions of the y-axis height.
LABEL_BELOW_FRAC = 0.03
LABEL_ABOVE_FRAC = 0.016

P = dict(
    fig_w=SINGLE_FIG_WIDTH * 1.4,
    fig_h=DOUBLE_FIG_HEIGHT * 0.946,
    f_label=10,
    f_tick=9,
    f_legend=11,
    f_bar=14,
    f_stall_free=14,
    ylabel_pad=22,
    legend_kw=dict(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        labelspacing=0.4,
        handlelength=2.0,
        columnspacing=1.4,
        borderpad=0.5,
        handletextpad=0.5,
    ),
)

FILES = {
    (("source_tag", "zephon-fused"),): "vldb_vlm_zephon-fused.csv",
    (("source_tag", "grain"),): "vldb_vlm_grain.csv",
    (("source_tag", "vlmdataset"),): "vldb_vlm_internalds.csv",
}

PERF_COLS = [
    "result.total_time_s",
    "result.step_time_s",
    "result.it_per_second",
    "result.samples_per_second",
]
TOKEN_COLS = ["result.packing_efficiency", "result.usable_tokens_per_second"]

# Trailing spaces widen the legend entry for LaTeX's wider re-rendering.
SYSTEM_STYLES = {
    "Zephon Fused": {"color_idx": 3, "label": r"\z Zephon"},
    "Grain": {"color_idx": 0, "label": r"\z Grain"},
    "VLMDataset": {"color_idx": 2, "label": r"\z InternalDs       "},
}


def _parse_loader_name(name: str, nw: int) -> tuple[str, int, str]:
    m = re.match(r"zephon-(fused|nonfused)-dop(\d+)-(.*)", name)
    if m:
        return f"zephon-{m.group(1)}", int(m.group(2)), m.group(3)
    if name == "grain-shuffled":
        return "grain", nw, "default"
    if name == "vlmdataset-shuffled":
        return "vlmdataset", nw, "default"
    return name, nw, "default"


def load_and_process(data_dir: Path | None = None) -> pl.DataFrame:
    """Load the CSVs and aggregate each allocation across seeds."""
    raw = read_measurements(DATA_DIR if data_dir is None else Path(data_dir), FILES)
    rows = raw.select("loader_name", "run.num_workers").to_dicts()
    parsed = [_parse_loader_name(r["loader_name"], int(r["run.num_workers"] or 0)) for r in rows]
    raw = raw.with_columns(
        pl.Series("system", [p[0] for p in parsed]),
        pl.Series("dop", [p[1] for p in parsed]).cast(pl.Int64),
        pl.Series("config", [p[2] for p in parsed]),
    )
    agg = aggregate_seeds(
        raw,
        ["system", "dop", "config", "loader_name", "source_tag"],
        PERF_COLS + TOKEN_COLS,
        count="n_seeds",
    ).sort(["system", "dop", "result.samples_per_second__mean"], descending=[False, False, True])
    renames = {
        "result.samples_per_second__mean": "sps_mean",
        "result.samples_per_second__std": "sps_std",
        "result.usable_tokens_per_second__mean": "utps_mean",
        "result.usable_tokens_per_second__std": "utps_std",
    }
    return agg.rename({k: v for k, v in renames.items() if k in agg.columns})


def select_best_configs(agg: pl.DataFrame) -> pl.DataFrame:
    """Keep each system's fastest allocation per DOP, by samples/s as in the paper."""
    parts = []
    for system, label in [
        ("zephon-fused", "Zephon Fused"),
        ("grain", "Grain"),
        ("vlmdataset", "VLMDataset"),
    ]:
        best = (
            agg.filter(pl.col("system") == system)
            .sort("sps_mean", descending=True)
            .group_by("dop", maintain_order=True)
            .first()
            .sort("dop")
        )
        if best.height == 0:
            continue
        parts.append(best.select(agg.columns).with_columns(pl.lit(label).alias("label")))
        for row in best.iter_rows(named=True):
            print(
                f"  {label} DOP {row['dop']:>2}: {row['config']:<16} → {row['sps_mean']:>8.1f} sps"
            )
    return pl.concat(parts, how="diagonal")


def _y_axis(top_value: float) -> tuple[float, list[float]]:
    """Round the headroom up to a tick; the top tick is left unlabelled."""
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(0, top_value * Y_HEADROOM)
    return float(ticks[-1]), [float(t) for t in ticks[:-1]]


def plot_fig_7(scaling: pl.DataFrame, plot_dir: Path, *, filename_prefix: str = "fig_7_"):
    plot_dir.mkdir(parents=True, exist_ok=True)
    init_plot()

    dops = sorted(scaling["dop"].unique().to_list())
    labels = sorted(scaling["label"].unique().to_list())
    bar_width = 0.8 / len(labels)
    x = np.arange(len(dops))

    values = {}
    for label in labels:
        sub = scaling.filter(pl.col("label") == label)
        per_dop = dict(zip(sub["dop"], sub["utps_mean"], strict=True))
        values[label] = [per_dop[d] / d if d in per_dop else 0 for d in dops]
    ymax, yticks = _y_axis(max(max(v) for v in values.values()))

    fig, ax = plt.subplots(figsize=(P["fig_w"], P["fig_h"]))
    for gi, label in enumerate(labels):
        style = SYSTEM_STYLES.get(label, {"color_idx": 0})
        color = plotcolours.colours[style["color_idx"]]
        bars = ax.bar(
            x + (gi - (len(labels) - 1) / 2) * bar_width,
            values[label],
            width=bar_width,
            label=style.get("label", label),
            color=color,
            edgecolor="black",
            hatch=plotcolours.hatches[style["color_idx"]],
            linewidth=0.6,
            alpha=0.9,
        )
        # Zephon's labels sit inside its bars, the others' above.
        inside = "Zephon" in label or "Fused" in label
        for bar, val in zip(bars, values[label], strict=True):
            if val <= 0:
                continue
            if val >= 1e6:
                text = f"{val / 1e6:.1f}M"
            elif val >= 1e3:
                text = f"{val / 1e3:.1f}k"
            else:
                text = f"{val:.0f}"
            if inside:
                y, va = bar.get_height() - LABEL_BELOW_FRAC * ymax, "top"
            else:
                y, va = bar.get_height() + LABEL_ABOVE_FRAC * ymax, "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va=va,
                fontsize=P["f_bar"],
                fontweight="bold",
                color=color,
                bbox=LABEL_BBOX,
            )

    ax.set_xlabel("Degree of parallelism", fontsize=P["f_label"], labelpad=14)
    # Lowered so the label centres on the bars, not on the legend band above.
    ax.set_ylabel(
        "Per-GPU Usable\nTokens/sec",
        fontsize=P["f_label"],
        labelpad=P["ylabel_pad"],
        linespacing=2.1,
        y=0.36,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dops])
    ax.tick_params(axis="x", labelsize=P["f_tick"], pad=11)
    ax.tick_params(axis="y", labelsize=P["f_tick"])
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0" if v == 0 else f"{v / 1000:g}k" for v in yticks])

    # Above the bars, below their labels.
    ax.axhline(STALL_FREE_TOKENS_PER_SEC, color="0.45", ls=(0, (5, 3)), lw=2.0, zorder=4)
    ax.annotate(
        # Spaces widen the box without changing its height; LaTeX doesn't draw them.
        f"  Stall-free {STALL_FREE_TOKENS_PER_SEC / 1000:.1f}k  ",
        xy=(1.0, STALL_FREE_TOKENS_PER_SEC),
        xycoords=ax.get_yaxis_transform(),
        xytext=(-5, -2),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=P["f_stall_free"],
        fontweight="bold",
        color="0.42",
        zorder=5,
        bbox=LABEL_BBOX,
    )

    hide_borders(ax)
    y_grid(ax)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(
        framealpha=0.9,
        edgecolor="lightgray",
        fancybox=False,
        fontsize=P["f_legend"],
        **P["legend_kw"],
    )
    fig.tight_layout()
    save_all(fig, plot_dir / f"{filename_prefix}vlm_scaling", bbox_inches="tight")
