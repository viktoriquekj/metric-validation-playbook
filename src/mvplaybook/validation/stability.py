from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator
from mvplaybook.validation.utils import ensure_datetime_utc


@dataclass(frozen=True)
class StabilityValidator(Validator):
    """
    Checks:
      - stability over time (CV of metric per group)
      - rank stability between adjacent periods (Spearman corr)
    """
    min_periods: int = 5

    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        self._validate_metric_result(result)

        time_col = result.spec.time_col
        if not time_col or time_col not in result.metric_df.columns:
            return ValidationReport.skip(self.name, "Metric has no time_col; stability over time not applicable.")

        gk = result.spec.group_keys
        value_col = result.spec.value_col

        mdf = result.metric_df.copy()
        mdf[time_col] = ensure_datetime_utc(mdf[time_col])

        # Need enough time points
        n_periods = mdf[time_col].nunique(dropna=True)
        if n_periods < self.min_periods:
            return ValidationReport.skip(self.name, f"Not enough time periods (have {n_periods}, need {self.min_periods}).")

        # CV per group
        grp = mdf.groupby(gk, dropna=False)[value_col]
        stats = grp.agg(mean="mean", std="std").reset_index()
        stats["cv"] = stats["std"] / (stats["mean"].abs() + 1e-9)

        frac_unstable = float((stats["cv"] > 0.30).mean())  # heuristic

        # Rank stability: Spearman between consecutive periods (aggregate across groups)
        periods = sorted([p for p in mdf[time_col].dropna().unique()])
        corrs = []
        for i in range(1, len(periods)):
            a = mdf[mdf[time_col] == periods[i - 1]][gk + [value_col]]
            b = mdf[mdf[time_col] == periods[i]][gk + [value_col]]
            joined = a.merge(b, on=gk, suffixes=("_prev", "_cur"))
            if len(joined) >= 5:
                corr = joined[value_col + "_prev"].corr(joined[value_col + "_cur"], method="spearman")
                if pd.notna(corr):
                    corrs.append(float(corr))

        avg_rank_corr = float(np.mean(corrs)) if corrs else np.nan

        status = "PASS"
        summary = f"Stability computed across {n_periods} periods."

        if frac_unstable > 0.30 or (pd.notna(avg_rank_corr) and avg_rank_corr < 0.7):
            status = "WARN"
            summary = "Metric shows instability: high CV or low rank correlation over time."

        return ValidationReport(
            name=self.name,
            status=status,  # type: ignore
            summary=summary,
            details={"n_periods": int(n_periods), "frac_cv_gt_0.30": frac_unstable, "avg_rank_spearman": avg_rank_corr},
            tables={"cv_by_group": stats},
        )
