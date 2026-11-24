"""Render the paper figures from frozen, checksum-verified measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = {
    "1": "plot_fig_1_fault_resume",
    "6": "plot_fig_6_fetchnbatch",
    "7": "plot_fig_7_vlm_scaling",
    "8": "plot_fig_8_eval_determinism",
    "9": "plot_fig_9_vlm_determinism",
}


def verify_data(root: Path = ROOT / "paper_data") -> dict:
    """Reject missing, changed or extra paper input files."""
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {Path(name).relative_to("paper_data").as_posix() for name in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest.json"
    }
    if actual != expected:
        raise ValueError(
            f"Paper inputs differ: missing={expected - actual}, extra={actual - expected}"
        )
    for name, entry in manifest["files"].items():
        path = root / Path(name).relative_to("paper_data")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Paper input checksum mismatch: {name}")
        if path.suffix == ".csv":
            with path.open() as stream:
                reader = csv.DictReader(stream)
                count = sum(1 for _ in reader)
            if count != entry["rows"] or reader.fieldnames != entry["columns"]:
                raise ValueError(f"Paper input schema mismatch: {name}")
    return manifest


def render(figure: str, output: Path) -> None:
    verify_data()
    for number in FIGURES if figure == "all" else [figure]:
        module = importlib.import_module(f"plotting.system.{FIGURES[number]}")
        destination = output / f"figure-{number}"
        if number == "1":
            data, checkpoint = module.load_merged()
            module.plot_fig_1(data, checkpoint, destination)
        elif number == "6":
            module.plot_fig_6(module.load_and_process(), destination)
        elif number == "7":
            module.plot_fig_7(module.select_best_configs(module.load_and_process()), destination)
        elif number == "8":
            module.plot_range(module.load_evals(), destination, ["42", "1339"], combine=True)
        else:
            module.plot_fig_9(module.load_runs("control"), module.load_runs("zephon"), destination)
        print(f"Figure {number} saved to {destination}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify all included paper-data checksums")
    plot = commands.add_parser("plot", help="Render measured paper figures as PNG/PDF/SVG")
    plot.add_argument("figure", choices=["all", *FIGURES])
    plot.add_argument("--output", type=Path, default=Path("plots"))
    fetch = commands.add_parser("fetch-wandb", help="Export an unsampled user-supplied W&B run")
    fetch.add_argument("run", help="ENTITY/PROJECT/RUN_ID")
    fetch.add_argument("--metric", action="append", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--cache", type=Path)
    fetch.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(f"Verified {len(verify_data()['files'])} paper input files")
    elif args.command == "plot":
        render(args.figure, args.output)
    else:
        from plotting.common.wandb_data import WandbDataLoader

        parts = args.run.split("/")
        if len(parts) != 3 or not all(parts):
            parser.error("Use ENTITY/PROJECT/RUN_ID")
        if args.output.resolve().is_relative_to((ROOT / "paper_data").resolve()):
            parser.error("Choose an output outside the frozen paper_data directory")
        entity, project, run_id = parts
        data = WandbDataLoader(project, entity, args.cache).load_run(
            run_id, args.metric, force=args.force
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
