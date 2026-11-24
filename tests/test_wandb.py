import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plotting.common.wandb_data import WandbDataLoader


def test_wandb_fetch_scans_full_history_and_reuses_valid_cache(tmp_path, monkeypatch):
    run = Mock(updated_at="one")
    run.scan_history.return_value = [
        {"_step": step, "_timestamp": step, "train/loss": 10.0 / (step + 1)} for step in range(1001)
    ]
    api = Mock()
    api.run.return_value = run
    constructor = Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=constructor))
    loader = WandbDataLoader("project", "entity", tmp_path)
    data = loader.load_run("run", ["train/loss"])
    assert len(data) == 1001
    api.run.assert_called_with("entity/project/run")
    run.scan_history.assert_called_once_with(keys=["_step", "_timestamp", "train/loss"])
    loader.load_run("run", ["train/loss"])
    assert run.scan_history.call_count == 1
    offline = WandbDataLoader("project", "entity", tmp_path, offline=True)
    assert len(offline.load_run("run", ["train/loss"])) == 1001
    with pytest.raises(RuntimeError, match="offline"):
        _ = offline.api
    constructor.assert_called_once()


def test_missing_offline_run_fails_without_fetching(tmp_path):
    loader = WandbDataLoader("project", "entity", tmp_path, offline=True)
    with pytest.raises(FileNotFoundError):
        loader.load_run("absent", ["train/loss"])
