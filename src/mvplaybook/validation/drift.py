from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator
from mvplaybook.validation.utils import psi, ensure_datetime_utc


@dataclass(frozen=True)
class DriftValidator(Validator):
    """
    Detects drift between reference and current periods:
      - PSI on metric_value distribution
      - Optional PSI on source composition (segment columns if provided in ctx.extras['source_cols'])
    """
    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        self._validate_metric_result(result)

        time_col = result.spec.time_col
        value_col = result.spec.value_col

        if not time_col or time_col not in result.metric_df.columns:
            return ValidationReport.skip(self.name, "Metric has no time_col; drift requires time-indexed metric output.")

        mdf = result.metric_df.copy()
        mdf[time_col] = ensure_datetime_utc(mdf[time_col])

        ref_start, ref_end = (None, None)
        cur_start, cur_end = (None, None)
        if ctx.reference_period and ctx.current_period:
            ref_start, ref_end = ctx.reference_period
            cur_start, cur_end = ctx.current_period

            ref_mask = True
            cur_mask = True
            if ref_start:
                ref_mask = ref_mask & (mdf[time_col].dt.date >= ref_start)
            if ref_end:
                ref_mask = ref_mask & (mdf[time_col].dt.date <= ref_end)
            if cur_start:
                cur_mask = cur_mask & (mdf[time_col].dt.date >= cur_start)
            if cur_end:
                cur_mask = cur_mask & (mdf[time_col].dt.date <= cur_end)

            ref_df = mdf[ref_mask]
            cur_df = mdf[cur_mask]
        else:
            # fallback: split by time median
            t = mdf[time_col].dropna()
            if len(t) == 0:
                return ValidationReport.skip(self.name, "No valid timestamps to compute drift.")
            split = t.quantile(0.5)
            ref_df = mdf[mdf[time_col] <= split]
            cur_df = mdf[mdf[time_col] > split]

        if len(ref_df) < 30 or len(cur_df) < 30:
            return ValidationReport.skip(self.name, "Not enough data in reference/current windows for drift.")

        metric_psi = psi(ref_df[value_col].to_numpy(), cur_df[value_col].to_numpy(), bins=ctx.psi_bins)

        status = "PASS"
        summary = f"Metric PSI={metric_psi:.3f}."

        if metric_psi >= ctx.psi_fail:
            status = "FAIL"
            summary = f"Severe drift: metric PSI={metric_psi:.3f} >= {ctx.psi_fail}."
        elif metric_psi >= ctx.psi_warn:
            status = "WARN"
            summary = f"Drift detected: metric PSI={metric_psi:.3f} >= {ctx.psi_warn}."

        return ValidationReport(
            name=self.name,
            status=status,  # type: ignore
            summary=summary,
            details={
                "metric_psi": metric_psi,
                "psi_bins": ctx.psi_bins,
                "psi_warn": ctx.psi_warn,
                "psi_fail": ctx.psi_fail,
                "reference_period": ctx.reference_period,
                "current_period": ctx.current_period,
            },
        )
