"""Figure 9: VLM training loss for both loaders across GPU counts."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import MaxNLocator, MultipleLocator

from plotting.common import plotcolours
from plotting.common.common import (
    DOUBLE_FIG_HEIGHT,
    PAPER_DATA,
    MarkerLeftHandler,
    hide_borders,
    init_plot,
    save_all,
    tex,
    y_grid,
)

DATA_DIR = PAPER_DATA / "figure_9"
METRIC = "train/loss"
STEP_RANGE = (50, 220)
GPU_COUNTS = (8, 16, 32, 64)
# Room above the curves for the legend, as a multiple of their range.
LEGEND_HEADROOM = 1.5

# ``plot_w`` is the unlabelled panel; the labelled one is 1.25x wider.
P = dict(
    plot_w=3.2,
    fig_h=DOUBLE_FIG_HEIGHT * 0.5,
    f_label=10,
    f_tick=9,
    f_legend=11,
    legend_kw=dict(
        loc="upper right",
        bbox_to_anchor=(1.05, 0.98),
        ncol=2,
        labelspacing=1.3,
        columnspacing=1.5,
        borderpad=0.6,
        handlelength=2.2,
    ),
)


def load_runs(
    arm: str, gpu_counts=GPU_COUNTS, data_dir: Path = DATA_DIR
) -> dict[str, pl.DataFrame]:
    return {f"{n} GPU": pl.read_csv(data_dir / f"{arm}-dp{n}.csv") for n in gpu_counts}


def _window(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select("_step", METRIC)
        .drop_nulls(METRIC)
        .filter(pl.col("_step").is_between(STEP_RANGE[0] - 5, STEP_RANGE[1] + 5))
    )


def _ylim(*arms: dict[str, pl.DataFrame]) -> tuple[float, float, list[float]]:
    """Y-range and ticks covering every curve plus the legend headroom."""
    values = pl.concat([_window(df)[METRIC] for runs in arms for df in runs.values()])
    lo, hi = float(values.min()), float(values.max())
    margin = (hi - lo) * 0.03
    lo, hi = lo - margin, hi + margin
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(
        lo, hi + LEGEND_HEADROOM * (hi - lo)
    )
    return float(ticks[0]), float(ticks[-1]), [float(t) for t in ticks]


def _plot_single(
    runs: dict[str, pl.DataFrame],
    show_ylabel: bool,
    show_legend: bool,
    ylim: tuple[float, float],
    yticks: list[float],
) -> plt.Figure:
    fig_w = P["plot_w"] * 1.25 if show_ylabel else P["plot_w"]
    fig, ax = plt.subplots(figsize=(fig_w, P["fig_h"]))
    # Same plot-area width in both panels: 4.0 * 0.72 == 3.2 * 0.90 inches.
    if show_ylabel:
        ax.set_position([0.24, 0.18, 0.72, 0.78])
    else:
        ax.set_position([0.06, 0.18, 0.90, 0.78])

    for idx, (label, df) in enumerate(runs.items()):
        col = _window(df)
        if col.is_empty():
            continue
        ax.plot(
            col["_step"],
            col[METRIC].cast(pl.Float64),
            label=tex(label),
            color=plotcolours.colours[idx],
            linestyle="-",
            lw=2.5,
        )

    ax.set_xlabel("Training step", fontsize=P["f_label"], labelpad=14)
    ax.set_xlim(*STEP_RANGE)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.set_yticks(yticks)
    ax.tick_params(axis="x", labelsize=P["f_tick"], pad=11)
    ax.tick_params(axis="y", labelsize=P["f_tick"])
    if show_ylabel:
        ax.set_ylabel("Loss", fontsize=P["f_label"], labelpad=20)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    hide_borders(ax)
    y_grid(ax)
    ax.grid(True, which="major", alpha=0.25)
    if show_legend:
        ax.legend(
            framealpha=0.9,
            edgecolor="lightgray",
            fancybox=False,
            fontsize=P["f_legend"],
            handler_map={plt.Line2D: MarkerLeftHandler(numpoints=1)},
            **P["legend_kw"],
        )
    return fig


def plot_fig_9(
    non_det_data: dict[str, pl.DataFrame],
    zephon_data: dict[str, pl.DataFrame],
    plot_dir: Path,
):
    """Write (a) the standard loader with the y-axis and (b) Zephon with the legend."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    init_plot()
    lo, hi, yticks = _ylim(non_det_data, zephon_data)
    save_all(
        _plot_single(
            non_det_data, show_ylabel=True, show_legend=False, ylim=(lo, hi), yticks=yticks
        ),
        plot_dir / "fig_9a_non_det",
    )
    save_all(
        _plot_single(
            zephon_data, show_ylabel=False, show_legend=True, ylim=(lo, hi), yticks=yticks
        ),
        plot_dir / "fig_9b_zephon",
    )
