from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator


@dataclass(frozen=True)
class SensitivityValidator(Validator):
    """
    Sensitivity = do conclusions change under reasonable definition tweaks?

    Requires:
      - metric.compute_variant(df, ctx, variant)
      - ctx.extras['sensitivity_variants'] = list of dicts
    """
    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        self._validate_metric_result(result)

        variants = ctx.extras.get("sensitivity_variants")
        if not variants:
            return ValidationReport.skip(self.name, "No sensitivity_variants provided in ctx.extras.")

        try:
            base = result.metric_df.copy()
            base_val = result.spec.value_col
            base_key = result.spec.group_keys + ([result.spec.time_col] if result.spec.time_col else [])
        except Exception as e:
            return ValidationReport.skip(self.name, f"Cannot run sensitivity due to metric spec issue: {e}")

        if len(base_key) == 0:
            return ValidationReport.skip(self.name, "Sensitivity check needs group_keys/time_col to align results.")

        # Compute variants
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

            diff = merged[base_val + "_var"] - merged[base_val + "_base"]
            rel = diff.abs() / (merged[base_val + "_base"].abs() + 1e-9)

            rows.append({
                "variant": str(v),
                "mean_abs_diff": float(diff.abs().mean()),
                "mean_rel_diff": float(rel.mean()),
                "p90_rel_diff": float(np.quantile(rel, 0.90)),
                "n_aligned": int(len(merged)),
            })

        if not rows:
            return ValidationReport.skip(self.name, "No variant results aligned with base metric output.")

        out = pd.DataFrame(rows).sort_values("mean_rel_diff", ascending=False)

        status = "PASS"
        summary = "Sensitivity check completed."
        if out["p90_rel_diff"].max() > 0.20:
            status = "WARN"
            summary = "Metric is sensitive: large changes under definition perturbations."

        return ValidationReport(
            name=self.name,
            status=status,  # type: ignore
            summary=summary,
            tables={"sensitivity_summary": out},
            details={"max_p90_rel_diff": float(out["p90_rel_diff"].max())},
        )
