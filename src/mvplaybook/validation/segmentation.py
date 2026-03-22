from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator
from mvplaybook.validation.utils import ensure_datetime_utc


@dataclass(frozen=True)
class SegmentationBiasValidator(Validator):
    """
    Simpson-style check:
      - compute overall change between reference and current periods
      - compute change within each segment value
      - flag if overall change direction disagrees with most segments
    """
    min_segment_n: int = 30

    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        self._validate_metric_result(result)

        seg_cols = ctx.extras.get("segment_cols", [])
        if not seg_cols:
            return ValidationReport.skip(self.name, "No segment_cols provided in ctx.extras['segment_cols'].")

        time_col = result.spec.time_col
        value_col = result.spec.value_col
        gk = result.spec.group_keys

        if not time_col or time_col not in result.metric_df.columns:
            return ValidationReport.skip(self.name, "Metric has no time_col; segmentation bias check requires time series.")

        if not (ctx.reference_period and ctx.current_period):
            return ValidationReport.skip(self.name, "Need ctx.reference_period and ctx.current_period for this check.")

        mdf = result.metric_df.copy()
        mdf[time_col] = ensure_datetime_utc(mdf[time_col])

        # Helper to filter by period
        def _period_mask(start, end):
            mask = pd.Series(True, index=mdf.index)
            if start:
                mask &= (mdf[time_col].dt.date >= start)
            if end:
                mask &= (mdf[time_col].dt.date <= end)
            return mask

        ref_mask = _period_mask(*ctx.reference_period)
        cur_mask = _period_mask(*ctx.current_period)

        ref = mdf[ref_mask]
        cur = mdf[cur_mask]
        if len(ref) < 30 or len(cur) < 30:
            return ValidationReport.skip(self.name, "Not enough metric rows in ref/current.")

        overall_change = cur[value_col].mean() - ref[value_col].mean()
        overall_sign = np.sign(overall_change)

        findings = []
        contradictions = 0
        checked = 0

        # Segment on raw df (more natural), then recompute metric for each segment value
        for seg_col in seg_cols:
            if seg_col not in df.columns:
                continue
            for seg_val, seg_df in df.groupby(seg_col, dropna=False):
                if len(seg_df) < self.min_segment_n:
                    continue
                # recompute metric within this segment
                seg_res = metric.compute(seg_df, ctx)
                seg_res.validate()
                smdf = seg_res.metric_df.copy()
                if time_col not in smdf.columns:
                    continue
                smdf[time_col] = ensure_datetime_utc(smdf[time_col])
                sref = smdf[_period_mask(*ctx.reference_period)]
                scur = smdf[_period_mask(*ctx.current_period)]
                if len(sref) < 5 or len(scur) < 5:
                    continue
                change = scur[value_col].mean() - sref[value_col].mean()
                sign = np.sign(change)
                checked += 1
                if overall_sign != 0 and sign != 0 and sign != overall_sign:
                    contradictions += 1
                findings.append({"segment_col": seg_col, "segment_val": str(seg_val), "change": float(change)})

        if checked == 0:
            return ValidationReport.skip(self.name, "No segments had enough data for segmentation check.")

        frac_contra = contradictions / checked

        status = "PASS"
        summary = f"Segmentation check run across {checked} segment groups."

        if frac_contra > 0.30:
            status = "WARN"
            summary = f"Potential Simpson’s paradox: {frac_contra:.0%} of segment groups disagree with overall trend."

        return ValidationReport(
            name=self.name,
            status=status,  # type: ignore
            summary=summary,
            details={"overall_change": float(overall_change), "checked_segments": checked, "contradiction_rate": frac_contra},
            tables={"segment_changes": pd.DataFrame(findings).sort_values("change", ascending=False)},
        )
