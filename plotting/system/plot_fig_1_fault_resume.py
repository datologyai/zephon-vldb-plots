"""Figure 1: loss across a checkpoint and GPU-count change, with batch-hash barcodes."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator

from plotting.common import plotcolours
from plotting.common.common import PAPER_DATA, hide_borders, init_plot, save_all, tex

FAULT_DATA = PAPER_DATA / "figure_1"

# (subdir, resume world size). All three resume the same 16-GPU reference.
RESUME_CONFIGS = [("dp8", 8), ("dp32", 32), ("dp64", 64)]
REFERENCE_WORLD_SIZE = 16
RESUME_WIDTHS = [dp for _, dp in RESUME_CONFIGS]

# Whole-batch hash match; the barcode colours each step by it.
MATCH_COL = "full_hash_match"
FAULT_HASH_COL = "fault_model_batch_multiset_sha256"

# Steps shown either side of the fault. Over the full run the loss drop dwarfs
# the divergence and the curves collapse onto one line.
HALF_WINDOW = 30

# Y-axis top. The panel height scales with the range to keep one loss unit per inch.
SMALL_YMAX = 6.1

FAULT_RULE = dict(color="0.25", ls=(0, (4, 3)), lw=1.1)
BARCODE_LABEL_DROP_PT = 2.0

ARMS = ["elastic", "control"]
ARM_LABELS = {"elastic": "Zephon", "control": "Standard loader"}
# Reference first, then the three resume widths.
ZEPHON_TINTS = ("#1C7FD0", "#0B4F9E", "#4BA3EE", "#93C8F7")

# Leading digest bits drawn per step, one square pixel each. This sets the
# barcode height; the colour still comes from the full-hash comparison.
HASH_BITS = 11
HASH_MATCH_HUE = "#0070DE"
HASH_DIFFER_HUE = "#C41F3B"

P = dict(
    plot_w_in=1.85,
    loss_h_in=0.98,
    stack_gap_in=0.10,
    gutter_in=0.58,
    margin_in=0.24,
    xaxis_in=0.46,
    f_label=12,
    f_tick=11,
    f_note=10,
    lw=1.6,
)


@dataclass(frozen=True)
class _Geometry:
    reference: int
    resumes: tuple[int, ...]

    def __post_init__(self):
        widths = (self.reference, *self.resumes)
        if len(self.resumes) != 3 or len(set(widths)) != 4 or any(w < 1 for w in widths):
            raise ValueError("Select one reference and three distinct positive resume widths")

    @property
    def order(self):
        return (self.reference, *self.resumes)

    @property
    def colours(self):
        return {w: plotcolours.colours[i] for i, w in enumerate(self.order)}

    @property
    def tints(self):
        return dict(zip(self.order, ZEPHON_TINTS, strict=True))


def load_merged() -> tuple[pl.DataFrame, int]:
    """Merge the per-step CSVs, keyed by resume width; return it and the checkpoint step.

    Raises if the runs disagree on reference width or checkpoint step.
    """
    frames, checkpoint_steps = [], set()
    for subdir, resume_dp in RESUME_CONFIGS:
        run_dir = FAULT_DATA / subdir
        summary = json.loads((run_dir / "summary.json").read_text())
        checkpoint_steps.add(summary["checkpoint_step"])
        for arm in ARMS:
            ref_ws = summary["runs"][f"{arm}_reference"]["world_size"]
            got_ws = summary["runs"][f"{arm}_resume"]["world_size"]
            if ref_ws != REFERENCE_WORLD_SIZE:
                raise ValueError(
                    f"{subdir}/{arm}: reference world size {ref_ws} != {REFERENCE_WORLD_SIZE}"
                )
            if got_ws != resume_dp:
                raise ValueError(
                    f"{subdir}/{arm}: resume world size {got_ws} != labelled {resume_dp}"
                )
        frames.append(
            pl.read_csv(run_dir / "per_step.csv").with_columns(
                resume_dp=pl.lit(resume_dp, dtype=pl.Int64)
            )
        )
    if len(checkpoint_steps) != 1:
        raise ValueError(f"runs disagree on checkpoint step: {sorted(checkpoint_steps)}")
    return pl.concat(frames), checkpoint_steps.pop()


def _series(df: pl.DataFrame, arm: str, dp: int) -> pl.DataFrame:
    return df.filter((pl.col("arm") == arm) & (pl.col("resume_dp") == dp)).sort("step")


def _window(df: pl.DataFrame, checkpoint: int) -> tuple[int, int]:
    widest = min(checkpoint - 1, int(df["step"].max()) - checkpoint)
    half = min(HALF_WINDOW, widest)
    return checkpoint - half, checkpoint + half


def _xlim(x0: int, x1: int) -> tuple[float, float]:
    # Half a step of padding so the first and last barcode columns aren't cut.
    return x0 - 0.5, x1 + 0.5


def _loss_ylim(df: pl.DataFrame, x0: int, x1: int) -> tuple[float, float]:
    """Reference y-range; panel heights are scaled against it."""
    win = df.filter(pl.col("step").is_between(x0, x1))
    lo = float(min(win["reference_loss"].min(), win["fault_loss"].min()))
    hi = float(max(win["reference_loss"].max(), win["fault_loss"].max()))
    span = hi - lo
    return lo - span * 0.08, hi + span * 1.15


def _draw_loss(ax, df, checkpoint, arm, ylim, show_ylabel, geometry: _Geometry) -> None:
    """One arm's resumed curves against its own reference."""
    x0, x1 = _window(df, checkpoint)
    ax.axvline(checkpoint, zorder=5, **FAULT_RULE)

    zephon = arm == "elastic"
    widths = sorted(geometry.order)
    for j, w in enumerate(widths):
        if w == geometry.reference:
            s = _series(df, arm, geometry.resumes[0])
            y = s["reference_loss"]
        else:
            # Resumed runs replay the reference before the fault.
            s = _series(df, arm, w).filter(pl.col("step") >= checkpoint)
            y = s["fault_loss"]
        colour = geometry.tints[w] if zephon else geometry.colours[w]
        # Larger widths draw on top. Zephon's curves are identical, so its
        # reference stays on top and the line keeps one colour.
        z = 2 + len(widths) if (zephon and w == geometry.reference) else 2 + j
        ax.plot(s["step"], y, label=tex(str(w)), color=colour, ls="-", lw=P["lw"], zorder=z)

    ax.set_xlim(*_xlim(x0, x1))
    ax.set_ylim(ylim)
    ax.set_xticks([x0, checkpoint, x1])
    ax.set_xticklabels([])  # The barcode below carries the step axis.
    # Fixed 0.5 steps; the default locator drops the 5.5 gridline.
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="both", labelsize=P["f_tick"])
    if show_ylabel:
        ax.set_ylabel("Loss", fontsize=P["f_label"])
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.annotate(
        tex("fault"),
        xy=(checkpoint, ylim[0]),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=P["f_note"],
        color="0.35",
    )
    hide_borders(ax)
    ax.grid(axis="y", which="major", lw=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def _hash_raster(s: pl.DataFrame) -> np.ndarray:
    """RGB image, one row per step: inked pixels for 1 bits, white for 0 bits.

    Colour is per step, not per bit: a mismatched digest still shares about half
    its bits with the reference.
    """
    bits = np.array(
        [[(int(h, 16) >> (255 - i)) & 1 for i in range(HASH_BITS)] for h in s[FAULT_HASH_COL]],
        dtype=bool,
    )
    matches = s[MATCH_COL].to_numpy().astype(bool)
    hue = np.where(matches[:, None, None], to_rgb(HASH_MATCH_HUE), to_rgb(HASH_DIFFER_HUE))
    return np.where(bits[..., None], hue, 1.0)


def _draw_barcode(ax, df, checkpoint, arm, x0, x1, show_ylabel, geometry: _Geometry) -> None:
    s = _series(df, arm, geometry.resumes[0]).filter(pl.col("step").is_between(x0, x1))
    # aspect="auto": the caller sizes the axes for square pixels.
    ax.imshow(
        np.transpose(_hash_raster(s), (1, 0, 2)),
        aspect="auto",
        interpolation="nearest",
        extent=(x0 - 0.5, x1 + 0.5, HASH_BITS, 0),
    )
    ax.axvline(checkpoint, zorder=4, **FAULT_RULE)
    ax.set_xlim(*_xlim(x0, x1))
    ax.set_xticks([x0, checkpoint, x1])
    ax.set_yticks([])
    ax.set_xlabel("Training step", fontsize=P["f_label"], labelpad=2)
    ax.tick_params(axis="both", labelsize=P["f_tick"], length=2)
    if show_ylabel:
        ax.set_ylabel("Batch\nSHA", fontsize=P["f_note"], labelpad=4, linespacing=1.7)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(0.9)


def _align_ylabel_x(fig, ax_ref, ax_target, dy_pt: float = 0.0) -> None:
    """Move ``ax_target``'s y-label to line up with ``ax_ref``'s, then down by ``dy_pt``."""
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        dx = (
            ax_ref.yaxis.label.get_window_extent(r).x0
            - ax_target.yaxis.label.get_window_extent(r).x0
        )
        span = ax_target.transAxes.transform((1, 0))[0] - ax_target.transAxes.transform((0, 0))[0]
        x, y = ax_target.yaxis.label.get_position()
        if abs(dx) < 0.5:
            break
        ax_target.yaxis.set_label_coords(x + dx / span, y)
    if dy_pt:
        height = ax_target.transAxes.transform((0, 1))[1] - ax_target.transAxes.transform((0, 0))[1]
        ax_target.yaxis.set_label_coords(x, y - dy_pt * fig.dpi / 72 / height)


def _plot_arm(
    df: pl.DataFrame,
    checkpoint: int,
    arm: str,
    ylim: tuple[float, float],
    show_ylabel: bool,
    loss_h_in: float | None = None,
    geometry: _Geometry | None = None,
) -> plt.Figure:
    """Loss panel over a barcode, laid out in inches so both arms match.

    The labelled figure is wider by exactly its label gutter.
    """
    geometry = geometry or _Geometry(REFERENCE_WORLD_SIZE, tuple(RESUME_WIDTHS))
    x0, x1 = _window(df, checkpoint)
    loss_h = P["loss_h_in"] if loss_h_in is None else loss_h_in
    left = P["gutter_in"] if show_ylabel else P["margin_in"]
    bar_h = P["plot_w_in"] * HASH_BITS / (x1 - x0 + 1)
    fig_w = left + P["plot_w_in"] + P["margin_in"]
    fig_h = P["margin_in"] + loss_h + P["stack_gap_in"] + bar_h + P["xaxis_in"]

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_loss = fig.add_axes(
        (
            left / fig_w,
            (P["xaxis_in"] + bar_h + P["stack_gap_in"]) / fig_h,
            P["plot_w_in"] / fig_w,
            loss_h / fig_h,
        )
    )
    ax_bar = fig.add_axes(
        (left / fig_w, P["xaxis_in"] / fig_h, P["plot_w_in"] / fig_w, bar_h / fig_h)
    )
    _draw_loss(ax_loss, df, checkpoint, arm, ylim, show_ylabel, geometry)
    _draw_barcode(ax_bar, df, checkpoint, arm, x0, x1, show_ylabel, geometry)
    if show_ylabel:
        _align_ylabel_x(fig, ax_loss, ax_bar, dy_pt=BARCODE_LABEL_DROP_PT)
    return fig


def _report(df: pl.DataFrame, checkpoint: int):
    post = df.filter(pl.col("is_post_fault"))
    print(
        f"\nMerged {df.height} rows from {len(RESUME_CONFIGS)} runs "
        f"(reference {REFERENCE_WORLD_SIZE} GPU, checkpoint step {checkpoint})."
    )
    print(f"{'arm':<16}{'resume':>8}{'batch match':>14}{'loss exact':>13}{'max |dloss|':>14}")
    for arm in ARMS:
        for dp in sorted(RESUME_WIDTHS):
            s = post.filter((pl.col("arm") == arm) & (pl.col("resume_dp") == dp))
            match = f"{s[MATCH_COL].sum()}/{s.height}"
            exact = f"{s['loss_exact'].sum()}/{s.height}"
            print(
                f"{ARM_LABELS[arm]:<16}{f'{REFERENCE_WORLD_SIZE}->{dp}':>8}{match:>14}"
                f"{exact:>13}{s['loss_abs_diff'].max():>14.4g}"
            )


def plot_checkpoint_resume(
    data: pl.DataFrame,
    checkpoint: int,
    output: Path,
    *,
    reference_width: int,
    resume_widths: tuple[int, ...],
) -> list[Path]:
    """Render measured traces with explicit GPU widths and data-driven limits.

    Reads no stored paper data. Writes ``figure-1-control`` and ``figure-1-zephon``.
    """
    geometry = _Geometry(reference_width, tuple(sorted(resume_widths)))
    arms = [arm for arm in ("control", "elastic") if arm in data["arm"].unique()]
    if not arms or set(data["arm"].unique()) - set(ARMS):
        raise ValueError("Measured traces must declare control or elastic loader arms")
    for arm in arms:
        if set(data.filter(pl.col("arm") == arm)["resume_dp"]) != set(geometry.resumes):
            raise ValueError("Measured traces do not cover the declared resume widths")
    output.mkdir(parents=True, exist_ok=True)
    init_plot()
    x0, x1 = _window(data, checkpoint)
    window = data.filter(pl.col("step").is_between(x0, x1))
    lo = min(window["reference_loss"].min(), window["fault_loss"].min())
    hi = max(window["reference_loss"].max(), window["fault_loss"].max())
    full = _loss_ylim(data, x0, x1)
    ylim = (full[0], hi + (hi - lo) * 0.08)
    height = P["loss_h_in"] * (ylim[1] - ylim[0]) / (full[1] - full[0])
    paths = []
    for arm in arms:
        fig = _plot_arm(
            data,
            checkpoint,
            arm,
            ylim,
            show_ylabel=arm == arms[0],
            loss_h_in=height,
            geometry=geometry,
        )
        name = "figure-1-zephon" if arm == "elastic" else "figure-1-control"
        paths += save_all(fig, output / name)
    return paths


def plot_fig_1(df: pl.DataFrame, checkpoint: int, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    init_plot()
    full = _loss_ylim(df, *_window(df, checkpoint))
    ylim = (full[0], SMALL_YMAX)
    height = P["loss_h_in"] * (ylim[1] - ylim[0]) / (full[1] - full[0])
    for arm, name, show_ylabel in [
        ("control", "fig_1a_standard_loader_small", True),
        ("elastic", "fig_1b_zephon_small", False),
    ]:
        fig = _plot_arm(df, checkpoint, arm, ylim, show_ylabel=show_ylabel, loss_h_in=height)
        save_all(fig, plot_dir / name)
    _report(df, checkpoint)
