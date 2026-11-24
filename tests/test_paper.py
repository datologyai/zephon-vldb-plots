import json
import shutil
import socket

import pytest

from plotting import cli


def test_paper_inputs_are_complete_and_verified():
    manifest = cli.verify_data()
    assert set(manifest["figures"]) == {"1", "6", "7", "8", "9"}
    assert len(manifest["files"]) == 25


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_bad_paper_input_cannot_render(tmp_path, mutation):
    data = tmp_path / "paper_data"
    shutil.copytree(cli.ROOT / "paper_data", data)
    target = data / "figure_9/control-dp8.csv"
    if mutation == "changed":
        target.write_text(target.read_text().replace("train/loss", "wrong/loss"))
    elif mutation == "missing":
        target.unlink()
    else:
        (data / "extra.csv").write_text("not a paper input")
    with pytest.raises(ValueError):
        cli.verify_data(data)


def test_checkpoint_comparisons_cover_both_loaders_and_all_widths():
    from plotting.system import plot_fig_1_fault_resume as figure

    data, checkpoint = figure.load_merged()
    assert checkpoint == 150
    assert data.height == 1800
    for arm in ("control", "elastic"):
        for dp in (8, 32, 64):
            series = figure._series(data, arm, dp)
            assert series["step"].to_list() == list(range(1, 301))
            assert series["rank_coverage_ok"].all()
            assert all(
                len(value) == 64 for value in series["reference_model_batch_multiset_sha256"]
            )


def test_evaluation_spread_averages_within_seed_ranges():
    from plotting.system import plot_fig_8_eval_determinism as figure

    data = figure.load_evals()
    assert set(data["seeds"]) == {"42", "1339"}
    for key in ("Core_v1", "FineWeb"):
        rows = figure.spreads(data, key, ["42", "1339"], combine=True)
        for row, loader in zip(rows, ("nondet", "zephon"), strict=True):
            ranges = []
            for seed in ("42", "1339"):
                values = list(data["seeds"][seed]["aggregates"][key]["arms"][loader].values())
                ranges.append(max(values) - min(values))
            assert row[1] == pytest.approx(sum(ranges) / 2)


def test_every_paper_figure_renders_without_network(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Paper rendering attempted network access")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    cli.render("all", tmp_path)
    for number in cli.FIGURES:
        directory = tmp_path / f"figure-{number}"
        for extension in ("svg", "pdf", "png"):
            files = list(directory.glob(f"*.{extension}"))
            assert files and all(path.stat().st_size > 1000 for path in files)
    svg = (tmp_path / "figure-1/fig_1a_standard_loader_small.svg").read_text()
    assert "fault" in svg.lower()


def test_frozen_inputs_do_not_embed_storage_or_local_paths():
    for path in (cli.ROOT / "paper_data").rglob("*"):
        if path.is_file():
            content = path.read_text()
            for marker in ("s3://", "/Users/", "/home/", "WANDB_API_KEY"):
                assert marker not in content, path
    data = json.loads((cli.ROOT / "paper_data/figure_8/downstream_evals.json").read_text())
    assert all(
        set(block["aggregates"]) == {"Core_v1", "FineWeb"} for block in data["seeds"].values()
    )


@pytest.mark.parametrize("number", ["6", "7"])
def test_throughput_accepts_external_csv_measurements(tmp_path, monkeypatch, number):
    import importlib

    import polars as pl

    figure = importlib.import_module(f"plotting.system.{cli.FIGURES[number]}")
    baseline = figure.load_and_process()
    if number == "7":
        baseline = figure.select_best_configs(baseline)
    source = figure.DATA_DIR
    external = tmp_path / "measured-csvs"
    external.mkdir()
    for path in source.glob("*.csv"):
        frame = pl.read_csv(path, infer_schema_length=10000)
        frame.with_columns(pl.col("result.samples_per_second") * 2).write_csv(external / path.name)
    monkeypatch.setattr(figure, "DATA_DIR", external)
    measured = figure.load_and_process()
    if number == "7":
        measured = figure.select_best_configs(measured)
    assert sorted(measured["sps_mean"].to_list()) == pytest.approx(
        sorted((baseline["sps_mean"] * 2).to_list())
    )
    getattr(figure, f"plot_fig_{number}")(measured, tmp_path / "figures")
    assert list((tmp_path / "figures").glob("*.svg"))


def test_training_renderer_accepts_supplied_trace_frame(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import polars as pl

    from plotting.system import plot_fig_1_fault_resume as figure

    data, checkpoint = figure.load_merged()
    measured = data.with_columns(pl.col("reference_loss") + 100, pl.col("fault_loss") + 100)

    def forbidden():
        raise AssertionError("Supplied trace rendering read stored paper inputs")

    monkeypatch.setattr(figure, "load_merged", forbidden)
    canvas = figure._plot_arm(measured, checkpoint, "control", (100, 113), show_ylabel=True)
    assert any(
        min(line.get_ydata()) > 100 for line in canvas.axes[0].lines if len(line.get_ydata()) > 2
    )
    canvas.savefig(tmp_path / "measured-training.svg")
    plt.close(canvas)


@pytest.mark.parametrize("reference,resumes", [(16, (8, 32, 64)), (2, (1, 4, 8))])
def test_measured_training_api_uses_explicit_widths_without_global_changes(
    tmp_path, monkeypatch, reference, resumes
):
    import polars as pl

    from plotting.system import plot_fig_1_fault_resume as figure

    data, checkpoint = figure.load_merged()
    data = data.with_columns(
        pl.col("reference_loss") + 100,
        pl.col("fault_loss") + 100,
        pl.col("resume_dp").replace_strict(dict(zip((8, 32, 64), resumes))),
    )
    defaults = (figure.REFERENCE_WORLD_SIZE, tuple(figure.RESUME_WIDTHS))

    def forbidden():
        raise AssertionError("Measured rendering read stored paper inputs")

    monkeypatch.setattr(figure, "load_merged", forbidden)
    geometries = []
    draw = figure._plot_arm

    def capture(*args, **kwargs):
        geometries.append(kwargs["geometry"])
        return draw(*args, **kwargs)

    monkeypatch.setattr(figure, "_plot_arm", capture)
    paths = figure.plot_checkpoint_resume(
        data, checkpoint, tmp_path, reference_width=reference, resume_widths=resumes
    )
    assert len(paths) == 6 and all(p.stat().st_size > 1000 for p in paths)
    assert {(g.reference, g.resumes) for g in geometries} == {(reference, resumes)}
    assert defaults == (
        figure.REFERENCE_WORLD_SIZE,
        tuple(figure.RESUME_WIDTHS),
    )
    assert r"\z fault" in (tmp_path / "figure-1-control.svg").read_text()


def test_measured_training_api_rejects_mismatched_widths(tmp_path):
    from plotting.system import plot_fig_1_fault_resume as figure

    data, checkpoint = figure.load_merged()
    with pytest.raises(ValueError, match="declared resume widths"):
        figure.plot_checkpoint_resume(
            data, checkpoint, tmp_path, reference_width=2, resume_widths=(1, 4, 8)
        )


def test_vlm_axis_matches_paper_and_grows_with_data():
    from plotting.system import plot_fig_7_vlm_scaling as figure

    scaling = figure.select_best_configs(figure.load_and_process())
    top = max(scaling["utps_mean"] / scaling["dop"])
    assert figure._y_axis(top) == (50000.0, [0.0, 10000.0, 20000.0, 30000.0, 40000.0])
    ymax, _ = figure._y_axis(top * 3)
    assert ymax >= top * 3


def test_vlm_loss_axis_matches_paper_and_follows_data():
    import polars as pl

    from plotting.system import plot_fig_9_vlm_determinism as figure

    control, zephon = figure.load_runs("control"), figure.load_runs("zephon")
    lo, hi, ticks = figure._ylim(control, zephon)
    assert (lo, hi) == (1.0, 2.0) and ticks == pytest.approx([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    scaled = [
        {k: v.with_columns(pl.col(figure.METRIC) * 3) for k, v in runs.items()}
        for runs in (control, zephon)
    ]
    lo, hi, _ = figure._ylim(*scaled)
    assert lo <= 3.2 and hi >= 4.3


def test_missing_measurements_fail_clearly(tmp_path):
    from plotting.system import plot_fig_6_fetchnbatch as figure

    with pytest.raises(FileNotFoundError, match="No measurement CSVs"):
        figure.load_and_process(tmp_path)
