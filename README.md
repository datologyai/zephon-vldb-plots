# Zephon VLDB plots

This repository contains the plotting code and measurements for the figures in
the [Zephon](https://github.com/datologyai/zephon)
[VLDB paper](https://arxiv.org/abs/2610.03087). You can render the paper's
figures from the included data or use the same plotting functions with new
measurements.

To reproduce the experiments, you don't need to run this repository yourself.
[zephon-vldb-experiments](https://github.com/datologyai/zephon-vldb-experiments)
downloads a pinned copy and uses it to plot the figures when a full experiment
run completes.

| Figure | Command | Experiment |
| --- | --- | --- |
| 1 | `zephon-plots plot 1` | Training across a checkpoint and a change in GPU count. |
| 6 | `zephon-plots plot 6` | Fetch-and-batch throughput. |
| 7 | `zephon-plots plot 7` | Throughput of the complete VLM pipeline. |
| 8 | `zephon-plots plot 8` | Downstream evaluation across GPU counts. |
| 9 | `zephon-plots plot 9` | VLM training determinism. |

Figures 2–5 are diagrams and are not generated here.

## Render the paper figures manually

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
git clone https://github.com/datologyai/zephon-vldb-plots.git
cd zephon-vldb-plots
uv sync --locked
uv run zephon-plots plot all
```

The measurements are included under `paper_data/`, so rendering does not need
W&B credentials or access to a training cluster. The command checks the data
before rendering. Each figure's data holds only what the figure plots. For
example, Figure 7 keeps the selected allocation at each DOP, not every
allocation measured. The input list, checksums and selections are in
[`paper_data/manifest.json`](paper_data/manifest.json).

Figures are saved under `plots/figure-N/` in PNG, PDF and SVG formats. Each
figure is split into its panels, as laid out in the manuscript. PNG and PDF use
plain text; SVGs keep the manuscript's LaTeX macros. To render one figure or
change the output directory:

```sh
uv run zephon-plots plot 6 --output figures
```

This writes Figure 6 under `figures/figure-6/`.

## CLI overview

| Command | Use |
| --- | --- |
| `zephon-plots plot all` | Render every included paper figure. |
| `zephon-plots plot N --output PATH` | Render one figure under `PATH/figure-N/`. |
| `zephon-plots verify` | Check the included measurements without rendering. |
| `zephon-plots fetch-wandb` | Export measurements from your own W&B run. |

Prefix commands with `uv run`; `uv run zephon-plots --help` or a command's
`--help` lists the options.

## Plot new measurements

zephon-vldb-experiments regenerates a run's figures with
`uv run reproduce plot RUN`. The paper-rendering CLI here always uses the
included measurements. To plot your own outside the launcher, call the Python
functions directly. For example, with Figure 6 CSVs that use the same file
names and columns as the included data:

```python
from pathlib import Path
from plotting.system import plot_fig_6_fetchnbatch as figure

points = figure.load_and_process(Path("measured-csvs"))
figure.plot_fig_6(points, Path("figures"), filename_prefix="figure-6-")
```

See the [included Figure 6 data](paper_data/figure_6/) for the file names and
columns. A missing file is skipped with a `SKIP` message, not an error.

Figure 7 uses `load_and_process`, `select_best_configs` and `plot_fig_7`.
Figure 1 uses `plot_checkpoint_resume`, which takes per-step traces and the
baseline and resume GPU counts. For a worked example, see
[`training/plotting.py`](https://github.com/datologyai/zephon-vldb-experiments/blob/main/src/vldb_experiments/training/plotting.py)
in zephon-vldb-experiments. The renderers are in
[`plotting/system/`](plotting/system/).

## Fetch your own W&B run

W&B fetching is optional. Install the extra and log in:

```sh
uv sync --locked --extra wandb
uv run wandb login
uv run zephon-plots fetch-wandb ENTITY/PROJECT/RUN_ID \
  --metric train/loss --output plots/my-run.csv
```

Fetching exports every logged step without sampling. Repeat `--metric` to export
several metrics, but only steps that log all of them are kept. Export metrics
logged at different steps separately.

The export is cached under `~/.cache/zephon-vldb-plots/wandb` and refreshed when
the run changes. `--cache PATH` chooses another location and `--force`
downloads it again. The exported CSV does not replace the included paper
measurements or add itself to a figure.

## Development

```sh
uv run ruff check plotting tests
MPLBACKEND=Agg uv run pytest
```

The accessible palette comes from [plotcolours](https://github.com/Sipondo/plotcolours).
