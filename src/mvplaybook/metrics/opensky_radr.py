from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Optional

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult
from mvplaybook.metrics.base import Metric


def _to_datetime_utc_day(s: pd.Series) -> pd.Series:
    """Parse to UTC datetime and floor to day."""
    return pd.to_datetime(s, utc=True, errors="coerce").dt.floor("D")


def _trimmed_mean(x: np.ndarray, low_q: float, high_q: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    lo = np.quantile(x, low_q)
    hi = np.quantile(x, high_q)
    x2 = x[(x >= lo) & (x <= hi)]
    return float(np.mean(x2)) if len(x2) else np.nan


def _pctl(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q))


@dataclass(frozen=True)
class OpenSkyRADRMetric(Metric):
    """
    RADR = Route/Airline Airborne Duration Reliability (proxy).

    Output is a time series by default:
      group_keys + time_col -> metric_value + n

    Default definition (can be changed via spec_extras / variants):
      - filter kinds (arrival/departure)
      - filter duration range (min/max)
      - aggregation: trimmed_mean (e.g. 5%-95%), mean, median, p90
    """

    # defaults for this metric
    time_col: Optional[str] = "date_utc"
    value_col: str = "metric_value"
    count_col: str = "n"

    # Default knobs (can be overwritten by spec_extras at construction)
    spec_extras: Dict[str, Any] = field(default_factory=lambda: {
        "kind_filter": None,             # e.g. "ARRIVAL" or "DEPARTURE" or None
        "min_duration_min": 10.0,        # filter out tiny / bad values
        "max_duration_min": 600.0,       # filter out extreme outliers
        "agg": "trimmed_mean",           # "mean" | "median" | "trimmed_mean" | "p90"
        "trim_low_q": 0.05,
        "trim_high_q": 0.95,
        "pctl_q": 0.90,                  # used if agg == "p90"
        "make_route_col": True,          # create route = DEP-ARR if group_keys includes "route"
        "route_col": "route",
    })

    @property
    def required_columns(self) -> Sequence[str]:
        # We keep this minimal; route is derived if needed
        return [
            "obs_duration_min",
            "date_utc",
            "kind",
            "callsign_prefix",
            "estDepartureAirport",
            "estArrivalAirport",
        ]

    def compute(self, df: pd.DataFrame, ctx: RunContext) -> MetricResult:
        self.validate_input(df)

        cfg = dict(self.spec_extras)

        # --- Normalize base columns ---
        work = df.copy()

        # Ensure date_utc is daily UTC datetime
        work["date_utc"] = _to_datetime_utc_day(work["date_utc"])

        # Normalize strings
        for col in ["kind", "callsign_prefix", "estDepartureAirport", "estArrivalAirport"]:
            work[col] = work[col].astype("string").str.upper()

        # Numeric duration
        work["obs_duration_min"] = pd.to_numeric(work["obs_duration_min"], errors="coerce")

        # --- Optional derived columns ---
        route_col = cfg.get("route_col", "route")
        if cfg.get("make_route_col", True) and ("route" in self.group_keys) and (route_col not in work.columns):
            dep = work["estDepartureAirport"].fillna("<NA>").astype("string")
            arr = work["estArrivalAirport"].fillna("<NA>").astype("string")
            work[route_col] = dep + "-" + arr

        # --- Filters ---
        # kind filter
        kind_filter = cfg.get("kind_filter")
        if kind_filter:
            kind_filter = str(kind_filter).upper()
            work = work[work["kind"] == kind_filter]

        # duration range
        min_d = float(cfg.get("min_duration_min", 10.0))
        max_d = float(cfg.get("max_duration_min", 600.0))
        work = work[(work["obs_duration_min"] >= min_d) & (work["obs_duration_min"] <= max_d)]

        # Drop rows missing key fields used for grouping
        group_cols = list(self.group_keys)
        # Replace "route" with actual column name if used
        if "route" in group_cols:
            group_cols = [route_col if c == "route" else c for c in group_cols]

        time_col = self.time_col or "date_utc"
        full_group_cols = group_cols + [time_col]

        work = work.dropna(subset=full_group_cols + ["obs_duration_min"])

        # --- Aggregate ---
        agg = str(cfg.get("agg", "trimmed_mean")).lower()

        def agg_fn(x: pd.Series) -> float:
            arr = x.to_numpy(dtype=float)
            if agg == "mean":
                return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
            if agg == "median":
                return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
            if agg == "p90":
                return _pctl(arr, float(cfg.get("pctl_q", 0.90)))
            if agg == "trimmed_mean":
                return _trimmed_mean(arr, float(cfg.get("trim_low_q", 0.05)), float(cfg.get("trim_high_q", 0.95)))
            raise ValueError(f"Unsupported agg='{agg}'")

        out = (
            work.groupby(full_group_cols, dropna=False)
                .agg(
                    metric_value=("obs_duration_min", agg_fn),
                    n=("obs_duration_min", "size"),
                )
                .reset_index()
        )

        # Rename route_col back to "route" if caller used group_keys=["route", ...]
        if "route" in self.group_keys and route_col in out.columns:
            # Keep it as 'route' (stable external interface)
            out = out.rename(columns={route_col: "route"})

        spec = self.make_spec(ctx)
        res = MetricResult(
            metric_df=out,
            spec=spec,
            artifacts={
                "filters": {
                    "kind_filter": kind_filter,
                    "min_duration_min": min_d,
                    "max_duration_min": max_d,
                },
                "definition": cfg,
                "n_input_rows": int(len(df)),
                "n_after_filters": int(len(work)),
            },
        )
        res.validate()
        return res

    def compute_variant(self, df: pd.DataFrame, ctx: RunContext, variant: Dict[str, Any]) -> MetricResult:
        """
        Recompute metric with definition tweaks.
        Variant keys override spec_extras.
        """
        new_extras = dict(self.spec_extras)
        new_extras.update(variant)

        # Create a new metric instance with updated spec_extras
        m2 = OpenSkyRADRMetric(
            name=self.name + "::variant",
            description=self.description + f" (variant={variant})",
            group_keys=list(self.group_keys),
            time_col=self.time_col,
            value_col=self.value_col,
            count_col=self.count_col,
            spec_extras=new_extras,
        )
        return m2.compute(df, ctx)
