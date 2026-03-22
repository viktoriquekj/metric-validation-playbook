from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator


@dataclass(frozen=True)
class UncertaintyValidator(Validator):
    """
    Bootstrap uncertainty intervals for metric_value per group.

    Strategy:
      - resample rows WITH replacement within each group key
      - recompute metric on resampled data
      - collect metric_value distribution per group
    """
    ci_low: float = 0.05
    ci_high: float = 0.95

    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        self._validate_metric_result(result)

        value_col = result.spec.value_col
        count_col = result.spec.count_col
        group_keys = result.spec.group_keys
        time_col = result.spec.time_col

        if not group_keys:
            return ValidationReport.skip(self.name, "Bootstrap requires group_keys.")

        # Prepare grouping columns
        group_cols = list(group_keys)
        if time_col:
            group_cols = group_cols + [time_col]

        # If route missing, derive it
        work = df.copy()
        if "route" in group_cols and "route" not in work.columns:
            if {"estDepartureAirport", "estArrivalAirport"}.issubset(work.columns):
                work["route"] = (
                    work["estDepartureAirport"].astype("string").str.upper().fillna("<NA>")
                    + "-"
                    + work["estArrivalAirport"].astype("string").str.upper().fillna("<NA>")
                )

        # Keep only necessary columns
        work = work[group_cols + ["obs_duration_min"]].dropna()

        rng = ctx.rng()
        n_boot = ctx.bootstrap_n

        rows = []

        grouped = work.groupby(group_cols, dropna=False)

        for key, g in grouped:
            x = g["obs_duration_min"].to_numpy(dtype=float)
            if len(x) < 5:
                rows.append((*((key,) if not isinstance(key, tuple) else key), np.nan, np.nan, np.nan))
                continue

            boot_vals = []
            for _ in range(n_boot):
                sample = rng.choice(x, size=len(x), replace=True)
                boot_vals.append(np.mean(sample))

            lo = float(np.quantile(boot_vals, self.ci_low))
            hi = float(np.quantile(boot_vals, self.ci_high))
            rows.append((*((key,) if not isinstance(key, tuple) else key), lo, hi, hi - lo))

        ci_df = pd.DataFrame(rows, columns=group_cols + ["ci_low", "ci_high", "ci_width"])

        merged = result.metric_df.merge(ci_df, on=group_cols, how="left")

        rel_width = (merged["ci_width"].abs() /
                    (merged[value_col].abs() + 1e-9))

        frac_bad = float((rel_width > 0.25).mean())

        status = "PASS"
        summary = f"Bootstrap CI computed (n_boot={n_boot})."

        if frac_bad > 0.30:
            status = "WARN"
            summary = "Large uncertainty for many groups."

        return ValidationReport(
            name=self.name,
            status=status,
            summary=summary,
            details={
                "bootstrap_n": n_boot,
                "frac_wide_ci": frac_bad,
            },
            tables={"bootstrap_ci": merged},
        )
