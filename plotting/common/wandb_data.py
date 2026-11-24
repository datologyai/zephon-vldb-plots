"""WandB data loader with CSV caching.

Downloads run metrics from WandB and caches them locally as CSV files.
On subsequent calls, skips the download if the run hasn't been updated
since the last fetch.
"""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "zephon-vldb-plots" / "wandb"
CACHE_META_SUFFIX = ".meta.json"


class WandbDataLoader:
    """Fetch WandB run history with local CSV caching.

    Usage::

        loader = WandbDataLoader(entity="my_entity", project="my_project")
        df = loader.load_run("my_run_id", metrics=["loss", "lr"])
    """

    def __init__(
        self,
        project: str,
        entity: str,
        cache_dir: Path | str | None = None,
        offline: bool = False,
    ):
        self.entity = entity
        self.project = project
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._api = None

    @property
    def api(self):
        if self.offline:
            raise RuntimeError("WandbDataLoader is in offline mode — no API access")
        if self._api is None:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError("Install W&B support with: uv sync --extra wandb") from error
            self._api = wandb.Api()
        return self._api

    # --------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------- #

    def load_run(
        self,
        run_id: str,
        metrics: list[str] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Load a single run's history, using the cache when possible.

        Args:
            run_id: WandB run ID.
            metrics: Optional list of metric keys to keep. If None, all
                     columns are returned.
            force: If True, ignore the cache and re-download.

        Returns:
            DataFrame with the run's metric history.
        """
        cache_key = self._cache_key(run_id, metrics)
        csv_path = self.cache_dir / f"{cache_key}.csv"
        meta_path = self.cache_dir / f"{cache_key}{CACHE_META_SUFFIX}"

        if self.offline:
            # Offline mode: only read from cache, never contact WandB
            if csv_path.exists():
                print(f"[offline]    {self.project}/{run_id}")
                return pd.read_csv(csv_path)
            # Try to find any cached CSV for this run_id (different metric sets)
            matches = list(self.cache_dir.glob(f"*_{run_id}.csv"))
            if matches:
                print(f"[offline]    {self.project}/{run_id} (alt cache)")
                return pd.read_csv(matches[0])
            raise FileNotFoundError(
                f"No cached data for {self.project}/{run_id} in {self.cache_dir}"
            )

        if not force and csv_path.exists() and meta_path.exists():
            cached_meta = self._read_meta(meta_path)
            remote_updated_at = self._get_run_updated_at(run_id)
            if remote_updated_at and cached_meta.get("updated_at") == remote_updated_at:
                print(f"[cache hit]  {self.project}/{run_id}")
                return pd.read_csv(csv_path)
            print(f"[cache stale] {self.project}/{run_id}")
        elif not force:
            print(f"[cache miss] {self.project}/{run_id}")
        else:
            print(f"[force]      {self.project}/{run_id}")

        df = self._download_run(run_id, metrics)
        self._write_cache(df, csv_path, meta_path, run_id)
        return df

    def load_runs(
        self,
        run_ids: dict[str, str],
        metrics: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Load multiple runs by {label: run_id} mapping.

        Returns:
            Dict mapping label -> DataFrame.
        """
        results = {}
        for label, run_id in run_ids.items():
            df = self.load_run(run_id, metrics=metrics, force=force)
            if not df.empty:
                results[label] = df
        return results

    def load_runs_by_name(
        self,
        run_names: dict[str, str],
        metrics: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Load runs by {label: run_name}, resolving names to IDs first.

        This iterates all runs in the project to find matching names.
        Prefer ``load_runs`` (by ID) when possible — it's faster.
        """
        target_names = set(run_names.values())
        name_to_label = {v: k for k, v in run_names.items()}
        resolved: dict[str, str] = {}

        for run in self.api.runs(f"{self.entity}/{self.project}"):
            if run.name in target_names:
                label = name_to_label[run.name]
                resolved[label] = run.id
                if len(resolved) == len(run_names):
                    break

        missing = set(run_names) - set(resolved)
        if missing:
            print(f"[warning] Could not find runs: {missing}")

        return self.load_runs(resolved, metrics=metrics, force=force)

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _download_run(self, run_id: str, metrics: list[str] | None) -> pd.DataFrame:
        run_path = f"{self.entity}/{self.project}/{run_id}"
        run = self.api.run(run_path)

        # scan_history streams all logged rows without sampling
        if metrics:
            keys = ["_step", "_timestamp"] + metrics
            rows = list(run.scan_history(keys=keys))
        else:
            rows = list(run.scan_history())

        if not rows:
            print(f"[empty]      {run_path}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        print(f"[downloaded] {run_path}  ({len(df)} rows)")
        return df

    def _get_run_updated_at(self, run_id: str) -> str | None:
        """Fetch the run's last updated timestamp (lightweight call)."""
        try:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            return run.updated_at
        except Exception:
            return None

    def _cache_key(self, run_id: str, metrics: list[str] | None) -> str:
        """Deterministic cache filename from project + run_id + metrics."""
        parts = f"{self.entity}/{self.project}/{run_id}"
        if metrics:
            parts += "/" + ",".join(sorted(metrics))
        return hashlib.sha256(parts.encode()).hexdigest()[:16] + f"_{run_id}"

    def _write_cache(
        self,
        df: pd.DataFrame,
        csv_path: Path,
        meta_path: Path,
        run_id: str,
    ):
        df.to_csv(csv_path, index=False)
        meta = {
            "entity": self.entity,
            "project": self.project,
            "run_id": run_id,
            "updated_at": self._get_run_updated_at(run_id),
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_rows": len(df),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    @staticmethod
    def _read_meta(meta_path: Path) -> dict:
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
