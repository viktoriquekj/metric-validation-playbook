
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, date, timezone

import numpy as np
import yaml


def _parse_date(x: Any) -> Optional[date]:
    """Parse YYYY-MM-DD (or already-date) into datetime.date."""
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        # Accept 'YYYY-MM-DD'
        return datetime.fromisoformat(x).date()
    raise TypeError(f"Cannot parse date from type={type(x)} value={x!r}")


@dataclass(frozen=True)
class RunContext:
    """
    RunContext is the shared configuration/state object passed to:
      DatasetAdapter.load(ctx)
      Metric.compute(df, ctx)
      Validator.run(df, metric, metric_result, ctx)
      ReportBuilder.build(..., ctx)

    It centralizes parameters (thresholds, windows, seeds) and ensures
    reproducible runs (via ctx.rng()) and consistent output paths.

    Keep it small, stable, and config-driven.
    """

    # Identity / bookkeeping
    dataset_name: str
    run_id: str = "run"
    project_root: Path = field(default_factory=lambda: Path(".").resolve())

    # Output
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    figures_subdir: str = "figures"

    # Time handling
    time_col: Optional[str] = None
    reference_period: Optional[Tuple[Optional[date], Optional[date]]] = None  # (start, end)
    current_period: Optional[Tuple[Optional[date], Optional[date]]] = None    # (start, end)
    timezone_name: str = "UTC"  # informational (you can enforce later)

    # Generic thresholds / knobs
    min_group_size: int = 50

    # Uncertainty / bootstrap
    bootstrap_n: int = 1000
    seed: int = 42

    # Drift
    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_fail: float = 0.25
    rolling_window_days: int = 7

    # Free-form config passthrough (keeps context extensible)
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---------- Constructors ----------
    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> "RunContext":
        path = Path(path)
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file must be a YAML mapping/dict, got: {type(cfg)}")
        if project_root is not None:
            cfg["project_root"] = str(project_root)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RunContext":
        # required
        dataset_name = cfg.get("dataset_name") or cfg.get("dataset")
        if not dataset_name:
            raise ValueError("Config must include 'dataset_name' (or alias 'dataset').")

        run_id = cfg.get("run_id", "run")

        project_root = Path(cfg.get("project_root", ".")).resolve()

        reports_dir = Path(cfg.get("reports_dir", "reports"))
        figures_subdir = cfg.get("figures_subdir", "figures")

        time_col = cfg.get("time_col")

        ref = cfg.get("reference_period")
        cur = cfg.get("current_period")

        def _parse_period(p: Any) -> Optional[Tuple[Optional[date], Optional[date]]]:
            if p is None:
                return None
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError("Period must be [start, end] where values are 'YYYY-MM-DD' or null.")
            return (_parse_date(p[0]), _parse_date(p[1]))

        reference_period = _parse_period(ref)
        current_period = _parse_period(cur)

        # typed knobs
        min_group_size = int(cfg.get("min_group_size", 50))
        bootstrap_n = int(cfg.get("bootstrap_n", 1000))
        seed = int(cfg.get("seed", 42))

        psi_bins = int(cfg.get("psi_bins", 10))
        psi_warn = float(cfg.get("psi_warn", 0.10))
        psi_fail = float(cfg.get("psi_fail", 0.25))
        rolling_window_days = int(cfg.get("rolling_window_days", 7))

        timezone_name = str(cfg.get("timezone_name", "UTC"))

        # extras: keep anything else here so context can evolve without breaking init
        known_keys = {
            "dataset_name", "dataset", "run_id", "project_root",
            "reports_dir", "figures_subdir",
            "time_col", "reference_period", "current_period", "timezone_name",
            "min_group_size", "bootstrap_n", "seed",
            "psi_bins", "psi_warn", "psi_fail", "rolling_window_days",
        }
        extras = {k: v for k, v in cfg.items() if k not in known_keys}

        return cls(
            dataset_name=str(dataset_name),
            run_id=str(run_id),
            project_root=project_root,
            reports_dir=reports_dir,
            figures_subdir=str(figures_subdir),
            time_col=time_col,
            reference_period=reference_period,
            current_period=current_period,
            timezone_name=timezone_name,
            min_group_size=min_group_size,
            bootstrap_n=bootstrap_n,
            seed=seed,
            psi_bins=psi_bins,
            psi_warn=psi_warn,
            psi_fail=psi_fail,
            rolling_window_days=rolling_window_days,
            extras=extras,
        )

    # ---------- Reproducibility ----------
    def rng(self) -> np.random.Generator:
        """Central RNG for all stochastic steps (bootstrap, sampling)."""
        return np.random.default_rng(self.seed)

    # ---------- Paths / outputs ----------
    def run_dir(self) -> Path:
        """Root output folder for this run, e.g. reports/opensky_ekch_jan2025/run"""
        return (self.project_root / self.reports_dir / self.dataset_name / self.run_id).resolve()

    def figures_dir(self) -> Path:
        return self.run_dir() / self.figures_subdir

    def ensure_output_dirs(self) -> None:
        self.run_dir().mkdir(parents=True, exist_ok=True)
        self.figures_dir().mkdir(parents=True, exist_ok=True)

    # ---------- Convenience ----------
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def as_dict(self) -> Dict[str, Any]:
        """Useful for logging/debugging."""
        return {
            "dataset_name": self.dataset_name,
            "run_id": self.run_id,
            "project_root": str(self.project_root),
            "reports_dir": str(self.reports_dir),
            "figures_subdir": self.figures_subdir,
            "time_col": self.time_col,
            "reference_period": self.reference_period,
            "current_period": self.current_period,
            "timezone_name": self.timezone_name,
            "min_group_size": self.min_group_size,
            "bootstrap_n": self.bootstrap_n,
            "seed": self.seed,
            "psi_bins": self.psi_bins,
            "psi_warn": self.psi_warn,
            "psi_fail": self.psi_fail,
            "rolling_window_days": self.rolling_window_days,
            "extras": self.extras,
        }
