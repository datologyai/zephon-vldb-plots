"""Figure 6: fetch-and-batch throughput, GIL and shared-memory panels."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl
from adjustText import adjust_text
from matplotlib.patches import Rectangle

from plotting.common import plotcolours
from plotting.common.common import (
    PAPER_DATA,
    MarkerLeftHandler,
    aggregate_seeds,
    hide_borders,
    init_plot,
    read_measurements,
    save_all,
    tex,
)

DATA_DIR = PAPER_DATA / "figure_6"

# Bottom strip of each panel, as an axes fraction, that peak labels avoid.
AXIS_GUARD_FRAC = 0.06

# Every panel has the same plot area; margins are in inches.
P = dict(
    plot_w=2.1,
    plot_h=0.9,
    m_left_ylabel=0.65,
    m_left=0.15,
    m_right=0.05,
    m_bottom=0.45,
    m_top=0.1,
    f_base=7.5,
    f_title=9,
    f_label=8,
    f_tick=6.5,
    f_legend_rc=6.5,
    f_annot=9.0,
    f_legend=7.0,
    lw=1.3,
    ms=3.5,
)

LABEL_BOX = {"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}

PERF_COLS = [
    "result.total_time_s",
    "result.step_time_s",
    "result.it_per_second",
    "result.samples_per_second",
]

GROUP_BY_COLS = [
    "sweep_python",
    "sweep_run_name",
    "loader_name",
    "benchmark.warmup_steps",
    "benchmark.measure_steps",
    "run.batch_size",
    "run.num_workers",
    "run.limit_steps",
    "category",
    "data_type",
    "variant",
]


def _tag(category, data_type, variant):
    return (("category", category), ("data_type", data_type), ("variant", variant))


FILES = {
    _tag("others", "text", "baseline"): [
        "vldb_fnb_others-text.csv",
        "vldb_fnb_streaming-text.csv",
    ],
    _tag("others", "vlm-raw", "baseline"): "vldb_fnb_others-vlm-raw.csv",
    _tag("zephon", "text", "baseline"): "vldb_fnb_zephon-text.csv",
    _tag("zephon", "vlm-raw", "baseline"): "vldb_fnb_zephon-vlm-raw.csv",
    _tag("zephon", "text", "noshm"): "vldb_fnb_zephon-text-noshm.csv",
    _tag("zephon", "vlm-raw", "noshm"): "vldb_fnb_zephon-vlm-raw-noshm.csv",
}

_SLOTS = {"Grain": 0, "Litdata": 1, "Streaming": 2, "Proc": 3, "Thread": 5}
COLORS = {name: plotcolours.colours[i] for name, i in _SLOTS.items()}
MARKERS = {name: plotcolours.markers[i] for name, i in _SLOTS.items()}

PARALLELISM_TICKS = [1, 2, 4, 8, 16, 32, 64]
DATA_TYPES = [("text", "text"), ("vlm", "vlm-raw")]
ROW_YLABELS = {"text": "Text\nSamples/sec", "vlm": "VLM\nSamples/sec"}


def _parse_loader_name(loader: str, category: str, nw: int):
    shuffled = "shuffled" in loader and "unshuffled" not in loader
    if category == "zephon":
        runner = "threads" if "threads" in loader else "processes"
        m = re.search(r"fdop(\d+)", loader)
        return "zephon", shuffled, runner, int(m.group(1)) if m else None
    for system in ("grain", "datology", "litdata"):
        if loader.startswith(system):
            return system, shuffled, None, nw
    return loader, shuffled, None, nw


def load_and_process(data_dir: Path | None = None) -> pl.DataFrame:
    raw = read_measurements(
        DATA_DIR if data_dir is None else Path(data_dir),
        FILES,
        null_values=["", "null", "None"],
        infer_schema_length=5000,
        schema_overrides={"sweep_python": pl.String},
    )
    agg = (
        aggregate_seeds(raw, GROUP_BY_COLS, PERF_COLS, count="n_runs")
        .sort(["loader_name", "result.samples_per_second__mean"], descending=[False, True])
        .rename(
            {
                "sweep_python": "py",
                "loader_name": "loader",
                "run.batch_size": "bs",
                "run.num_workers": "nw",
                "result.samples_per_second__mean": "sps_mean",
                "result.samples_per_second__std": "sps_sd",
            }
        )
    )
    rows = agg.select("loader", "category", "nw").to_dicts()
    parsed = [_parse_loader_name(r["loader"], r["category"], r["nw"] or 0) for r in rows]
    return agg.with_columns(
        pl.Series("system", [p[0] for p in parsed]),
        pl.Series("shuffled", [p[1] for p in parsed]),
        pl.Series("runner", [p[2] for p in parsed]),
        pl.Series("parallelism", [p[3] for p in parsed]).cast(pl.Int64, strict=False),
    )


# =============================================================================
# Panels: each returns the lines it draws
# =============================================================================


def _xy(df: pl.DataFrame, dt, py, variant, system, runner=None):
    f = (
        (pl.col("data_type") == dt)
        & pl.col("shuffled")
        & (pl.col("bs") == 32)
        & (pl.col("py") == py)
        & (pl.col("variant") == variant)
        & (pl.col("system") == system)
    )
    if runner:
        f = f & (pl.col("runner") == runner)
    s = df.filter(f).sort("parallelism")
    return s["parallelism"].to_list(), s["sps_mean"].to_list()


def _spec(xy, label, style, ls="-", peak_x_offset=0):
    x, y = xy
    return dict(
        x=x,
        y=y,
        label=label,
        color=COLORS[style],
        marker=MARKERS[style],
        ls=ls,
        peak_x_offset=peak_x_offset,
    )


def _overview(df, dt):
    lines = [
        _spec(_xy(df, dt, "3.13", "baseline", system), tex(name), name)
        for system, name in [("grain", "Grain"), ("litdata", "Litdata"), ("datology", "Streaming")]
    ]
    lines += [
        _spec(_xy(df, dt, "3.13", "baseline", "zephon", runner), tex(name), style)
        for runner, style, name in [
            ("processes", "Proc", "Z-Proc"),
            ("threads", "Thread", "Z-Thread"),
        ]
    ]
    return lines


# Trailing spaces widen the legend box: LaTeX re-renders the macro wider than
# matplotlib measures it.
def _shm(df, dt):
    return [
        _spec(_xy(df, dt, "3.13", "baseline", "zephon", "processes"), "_proc_baseline", "Proc"),
        _spec(
            _xy(df, dt, "3.13", "noshm", "zephon", "processes"),
            r"\nc                        ",
            "Proc",
            ls=":",
        ),
    ]


def _multithread(df, dt):
    lines = []
    for runner, style, macro, offset in [
        ("processes", "Proc", r"\na                         ", 0),
        ("threads", "Thread", r"\nb                           ", 4),
    ]:
        lines.append(
            _spec(_xy(df, dt, "3.13", "baseline", "zephon", runner), f"_{style}_gil", style)
        )
        lines.append(
            _spec(
                _xy(df, dt, "3.14t", "baseline", "zephon", runner),
                macro,
                style,
                ls="--",
                peak_x_offset=offset,
            )
        )
    return lines


SECTIONS = [("overview", _overview), ("multithread", _multithread), ("shm", _shm)]


# =============================================================================
# Drawing
# =============================================================================


def _draw_line(ax, spec) -> tuple[float, float, str] | None:
    """Plot one line; return its peak for labelling."""
    x, y = spec["x"], spec["y"]
    if not x or not y:
        return None
    ax.plot(
        x,
        y,
        label=spec["label"],
        color=spec["color"],
        marker=spec["marker"],
        ls=spec["ls"],
        lw=1.8,
        ms=5.0,
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
    i = max(range(len(y)), key=lambda i: y[i])
    return x[i] + spec["peak_x_offset"], y[i], spec["color"]


def _fmt_y(x, _):
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    if x >= 1e4:
        return f"{x / 1e3:.0f}k"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def _annotate_peaks(ax, peaks, avoid=None):
    """Label each line's peak without overlaps. ``avoid`` is the legend."""
    if not peaks:
        return
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    peak_ys = [py for _, py, _ in peaks]
    peak_span = max(peak_ys) - min(peak_ys) if len(peak_ys) > 1 else 0

    if peak_span < y_range * 0.3 and len(peaks) > 3:
        # Peaks bunched together: stack the labels down from the top instead.
        # Label height in data units; 1.5 font sizes covers the line box and pad.
        label_h = y_range * (P["f_annot"] * 1.5 / 72.0) / P["plot_h"]
        spacing = label_h * 0.88
        top_y = ylim[1] - label_h * 0.6
        prev = None
        for i, (px, py, color) in enumerate(sorted(peaks, key=lambda a: -a[1])):
            # Stay above the peak and clear of the x-axis.
            label_y = max(top_y - i * spacing, py + label_h * 0.6, ylim[0] + label_h * 0.75)
            if prev is not None and prev - label_y < spacing:
                label_y = prev - spacing
            prev = label_y
            ax.annotate(
                _fmt_y(py, None),
                xy=(px, py),
                xytext=(px, label_y),
                fontsize=P["f_annot"],
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                bbox=LABEL_BOX,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5, "alpha": 0.4},
            )
        return

    # Keep labels of low peaks off the x-axis.
    floor = ylim[0] + (0.15 * P["f_annot"] + 3.0) * y_range / (P["plot_h"] * 72.0)
    texts = [
        ax.text(
            px,
            max(py, floor),
            _fmt_y(py, None),
            fontsize=P["f_annot"],
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox=LABEL_BOX,
        )
        for px, py, color in peaks
    ]
    # Invisible strip standing in for the x-axis, which adjust_text can't see.
    guard = Rectangle((0, 0), 1, AXIS_GUARD_FRAC, transform=ax.transAxes, visible=False)
    ax.add_patch(guard)
    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5, "alpha": 0.5},
        force_text=(0.5, 1.0),
        expand=(1.2, 1.4),
        objects=[guard] + ([avoid] if avoid is not None else []),
        force_static=(2.0, 4.0),
    )
    guard.remove()


def _style_ax(ax):
    ax.set_xscale("log", base=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_y))
    ax.grid(True, which="major", alpha=0.25)
    ax.xaxis.set_major_locator(mticker.FixedLocator(PARALLELISM_TICKS))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    hide_borders(ax)
    ax.set_ylim(bottom=0)


def _apply_rcparams():
    plt.rcParams.update(
        {
            "font.size": P["f_base"],
            "axes.titlesize": P["f_title"],
            "axes.titlepad": 4,
            "axes.labelsize": P["f_label"],
            "axes.labelpad": 3,
            "xtick.labelsize": P["f_tick"],
            "ytick.labelsize": P["f_tick"],
            "legend.fontsize": P["f_legend_rc"],
            "legend.handlelength": 1.8,
            "legend.labelspacing": 0.2,
            "legend.borderpad": 0.3,
            "legend.handletextpad": 0.4,
            "lines.linewidth": P["lw"],
            "lines.markersize": P["ms"],
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def _row_ylim(df: pl.DataFrame, dt: str) -> tuple[float, float]:
    """Shared y-range for one data type across its three panels."""
    ys = [y for _, section in SECTIONS for spec in section(df, dt) for y in spec["y"]]
    return (0, max(ys) * 1.15) if ys else (0, 1)


def _plot_panel(
    lines: list[dict],
    ylim: tuple[float, float],
    ylabel: str | None,
    show_legend: bool,
    show_xlabel: bool,
    ylabel_pad: int,
    plot_w_shrink: float,
) -> plt.Figure:
    margin_left = P["m_left_ylabel"] if ylabel else P["m_left"]
    margin_bottom = P["m_bottom"] if show_xlabel else P["m_top"]
    fig_w = margin_left + P["plot_w"] + P["m_right"]
    fig_h = margin_bottom + P["plot_h"] + P["m_top"]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_position(
        [
            margin_left / fig_w,
            margin_bottom / fig_h,
            (P["plot_w"] - plot_w_shrink) / fig_w,
            P["plot_h"] / fig_h,
        ]
    )

    peaks = [peak for spec in lines if (peak := _draw_line(ax, spec))]
    _style_ax(ax)
    ax.set_ylim(ylim)

    if show_xlabel:
        ax.set_xlabel("Degree of parallelism")
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=ylabel_pad)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    legend = None
    if show_legend:
        legend = ax.legend(
            loc="upper left",
            framealpha=0.9,
            edgecolor="lightgray",
            fancybox=False,
            labelspacing=0.5,
            borderpad=0.4,
            handlelength=1.5,
            fontsize=P["f_legend"],
            handler_map={plt.Line2D: MarkerLeftHandler(numpoints=1)},
        )
    # After the legend exists, so the labels can avoid it.
    _annotate_peaks(ax, peaks, avoid=legend)
    return fig


def plot_fig_6(df: pl.DataFrame, plot_dir: Path, *, filename_prefix: str = "fig_6_"):
    """Write six panels: text and VLM rows, overview/multithread/shm columns."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    init_plot()
    _apply_rcparams()
    for row, (dt_key, dt) in enumerate(DATA_TYPES):
        ylim = _row_ylim(df, dt)
        for col, (section_key, section) in enumerate(SECTIONS):
            fig = _plot_panel(
                section(df, dt),
                ylim,
                ylabel=ROW_YLABELS[dt_key] if col == 0 else None,
                show_legend=dt_key == "text",
                show_xlabel=row == len(DATA_TYPES) - 1,
                # The VLM overview's shorter tick labels need a wider label gap,
                # and the text overview a slightly narrower plot, to line up.
                ylabel_pad=15 if (dt_key, section_key) == ("vlm", "overview") else 10,
                plot_w_shrink=0.021 if (dt_key, section_key) == ("text", "overview") else 0.0,
            )
            save_all(fig, plot_dir / f"{filename_prefix}{section_key}_{dt_key}")
