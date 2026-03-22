from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator


@dataclass(frozen=True)
class RobustnessValidator(Validator):
    """
    Robustness = does the metric hold under outlier-handling or alternative aggregation?

    Requires:
      - ctx.extras['robustness_variants'] (list of dicts)
      - metric.compute_variant()
    """
    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        variants = ctx.extras.get("robustness_variants")
        if not variants:
            return ValidationReport.skip(self.name, "No robustness_variants provided in ctx.extras.")

        # Reuse SensitivityValidator logic: robustness is just a different semantic set of variants
        # We'll run same comparison and report.
        try:
            base = result.metric_df.copy()
            base_val = result.spec.value_col
            base_key = result.spec.group_keys + ([result.spec.time_col] if result.spec.time_col else [])
        except Exception as e:
            return ValidationReport.skip(self.name, f"Cannot run robustness due to metric spec issue: {e}")

        rows = []
        for v in variants:
            try:
                vres = metric.compute_variant(df, ctx, v)
            except NotImplementedError:
                return ValidationReport.skip(self.name, "Metric does not support compute_variant().")
            vres.validate()
            vdf = vres.metric_df.copy()
            merged = base.merge(vdf, on=base_key, suffixes=("_base", "_var"), how="inner")
            if len(merged) == 0:
                continue
            rel = (merged[base_val + "_var"] - merged[base_val + "_base"]).abs() / (merged[base_val + "_base"].abs() + 1e-9)
            rows.append({"variant": str(v), "mean_rel_diff": float(rel.mean()), "n_aligned": int(len(merged))})

        if not rows:
            return ValidationReport.skip(self.name, "No robustness variant results aligned.")

        out = pd.DataFrame(rows).sort_values("mean_rel_diff", ascending=False)

        status = "PASS"
        summary = "Robustness check completed."
        if out["mean_rel_diff"].max() > 0.10:
            status = "WARN"
            summary = "Metric is not robust: it changes materially under robustness variants."

        return ValidationReport(
            name=self.name,
            status=status,  # type: ignore
            summary=summary,
            tables={"robustness_summary": out},
            details={"max_mean_rel_diff": float(out["mean_rel_diff"].max())},
        )
