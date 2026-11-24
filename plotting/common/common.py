"""Shared layout, styling and saving for the Zephon VLDB figures."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FuncFormatter

from plotting.common import plotcolours

PAPER_DATA = Path(__file__).resolve().parents[2] / "paper_data"
SINGLE_FIG_WIDTH = 5
DOUBLE_FIG_HEIGHT = 3.5
IMG_TYPES = [".png", ".pdf", ".svg"]

# Macros the manuscript defines; PNG/PDF previews spell them out.
_MACROS = {r"\na": "Z-Proc (3.14t)", r"\nb": "Z-Thread (3.14t)", r"\nc": "OS Pipe"}


def tex(name: str) -> str:
    """Mark text for LaTeX to re-render in the document font."""
    return rf"\z {name}"


def _plain(text: str) -> str:
    plain = text.replace(r"\z ", "").replace(r"\z", "")
    for macro, value in _MACROS.items():
        plain = plain.replace(macro, value)
    return plain


def save_figure(fig, path, **kwargs):
    """Save one file. SVGs keep the LaTeX macros; PNG/PDF get plain text."""
    saved = []
    formatters = []
    if Path(path).suffix != ".svg":
        # Tick labels are rebuilt from their formatter at draw time, so wrap it.
        # Numeric formatters are left alone: they need set_locs on themselves.
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                formatter = axis.get_major_formatter()
                if isinstance(formatter, FixedFormatter | FuncFormatter):
                    formatters.append((axis, formatter))
                    axis.set_major_formatter(
                        FuncFormatter(lambda x, pos, f=formatter: _plain(f(x, pos)))
                    )
        for label in fig.findobj(Text):
            text = label.get_text()
            plain = _plain(text)
            if plain != text:
                saved.append((label, text, label.get_fontsize()))
                label.set_text(plain.rstrip())
                if plain.strip() == "fault":
                    label.set_text("FAULT")
                    label.set_fontsize(label.get_fontsize() * 0.8)
    try:
        fig.savefig(path, **kwargs)
    finally:
        for label, text, size in saved:
            label.set_text(text)
            label.set_fontsize(size)
        for axis, formatter in formatters:
            axis.set_major_formatter(formatter)


def save_all(fig, stem: Path, **kwargs) -> list[Path]:
    """Save ``fig`` as PNG, PDF and SVG at 300 dpi, then close it."""
    paths = [Path(f"{stem}{ext}") for ext in IMG_TYPES]
    for path in paths:
        save_figure(fig, path, dpi=300, **kwargs)
    plt.close(fig)
    return paths


def read_measurements(data_dir: Path, files: dict, **read_kwargs) -> pl.DataFrame:
    """Read ``{tag: filename(s)}`` CSVs and add the tag as columns.

    A missing file is skipped with a message; no files at all is an error.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Measurement directory not found: {data_dir}")
    frames = []
    for tag, names in files.items():
        for name in [names] if isinstance(names, str) else names:
            path = data_dir / name
            if not path.exists():
                print(f"SKIP (not found): {path}")
                continue
            frame = pl.read_csv(path, **read_kwargs)
            frames.append(frame.with_columns(pl.lit(v).alias(k) for k, v in tag))
    if not frames:
        raise FileNotFoundError(f"No measurement CSVs found in {data_dir}")
    return pl.concat(frames, how="diagonal")


def aggregate_seeds(
    raw: pl.DataFrame, group_cols: list[str], metrics: list[str], count: str
) -> pl.DataFrame:
    """Cast the measurement columns and take the mean and std across seeds."""
    casts = [pl.col("run.seed").cast(pl.Int64, strict=False)]
    casts += [pl.col(c).cast(pl.Float64, strict=False) for c in metrics if c in raw.columns]
    casts += [
        pl.col(c).cast(pl.Int64, strict=False)
        for c in ("run.batch_size", "run.num_workers")
        if c in raw.columns
    ]
    raw = raw.with_columns(casts)
    aggs = [pl.len().alias(count)]
    for m in metrics:
        if m in raw.columns:
            aggs += [pl.col(m).mean().alias(f"{m}__mean"), pl.col(m).std().alias(f"{m}__std")]
    return raw.group_by([c for c in group_cols if c in raw.columns]).agg(aggs)


class MarkerLeftHandler(HandlerLine2D):
    """Legend line with its marker a quarter of the way in, not centred."""

    def create_artists(self, *args, **kwargs):
        artists = super().create_artists(*args, **kwargs)
        line = artists[0]
        xd = line.get_xdata()
        if len(xd) >= 2:
            start, end = xd[0], xd[-1]
            line.set_xdata([start, start + (end - start) * 0.25, end])
            line.set_ydata([line.get_ydata()[0]] * 3)
            line.set_markevery([1])
        return artists


def init_plot():
    plotcolours.apply(with_markers=False)
    matplotlib.rcParams.update(
        {
            "font.size": 14,
            "font.family": "serif",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )


def y_grid(ax):
    ax.grid(axis="y", which="major", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def hide_borders(ax, show_left: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(show_left)
