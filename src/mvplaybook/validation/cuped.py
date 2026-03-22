from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric
from mvplaybook.validation.base import Validator
from mvplaybook.validation.utils import ensure_datetime_utc


@dataclass(frozen=True)
class CUPEDValidator(Validator):
    """
    CUPED demo:
      - requires time series metric output
      - requires ctx.extras['cuped'] config:
          {
            "pre_period": ["YYYY-MM-DD", "YYYY-MM-DD"],
            "post_period": ["YYYY-MM-DD", "YYYY-MM-DD"]
          }
      - computes variance reduction of post metric mean after CUPED adjustment
    """
    def run(self, df: pd.DataFrame, metric: Metric, result: MetricResult, ctx: RunContext) -> ValidationReport:
        cfg = ctx.extras.get("cuped")
        if not cfg:
            return ValidationReport.skip(self.name, "No cuped config in ctx.extras['cuped'].")

        time_col = result.spec.time_col
        if not time_col or time_col not in result.metric_df.columns:
            return ValidationReport.skip(self.name, "CUPED requires time series metric output (time_col).")

        pre = cfg.get("pre_period")
        post = cfg.get("post_period")
        if not pre or not post:
            return ValidationReport.skip(self.name, "cuped config needs pre_period and post_period.")

        pre_start = pd.to_datetime(pre[0], utc=True)
        pre_end = pd.to_datetime(pre[1], utc=True)
        post_start = pd.to_datetime(post[0], utc=True)
        post_end = pd.to_datetime(post[1], utc=True)

        mdf = result.metric_df.copy()
        mdf[time_col] = ensure_datetime_utc(mdf[time_col])
        val = result.spec.value_col
        gk = result.spec.group_keys

        pre_df = mdf[(mdf[time_col] >= pre_start) & (mdf[time_col] <= pre_end)]
        post_df = mdf[(mdf[time_col] >= post_start) & (mdf[time_col] <= post_end)]

        if len(pre_df) < 30 or len(post_df) < 30:
            return ValidationReport.skip(self.name, "Not enough data in pre/post windows for CUPED demo.")

        # covariate X = pre-period mean per group; outcome Y = post-period mean per group
        X = pre_df.groupby(gk, dropna=False)[val].mean().rename("X").reset_index()
        Y = post_df.groupby(gk, dropna=False)[val].mean().rename("Y").reset_index()
        XY = X.merge(Y, on=gk, how="inner")
        if len(XY) < 10:
            return ValidationReport.skip(self.name, "Not enough aligned groups for CUPED.")

        x = XY["X"].to_numpy()
        y = XY["Y"].to_numpy()

        var_x = np.var(x, ddof=1)
        if var_x <= 0:
            return ValidationReport.skip(self.name, "Variance of pre-period covariate is zero; CUPED not applicable.")

        theta = np.cov(y, x, ddof=1)[0, 1] / var_x
        y_adj = y - theta * (x - np.mean(x))

        var_y = float(np.var(y, ddof=1))
        var_y_adj = float(np.var(y_adj, ddof=1))
        vr = 1.0 - (var_y_adj / (var_y + 1e-12))

        status = "PASS"
        summary = f"CUPED variance reduction ≈ {vr:.1%}."

        if vr < 0.05:
            status = "WARN"
            summary = f"Low CUPED gain: variance reduction ≈ {vr:.1%}."

        XY_out = XY.copy()
        XY_out["Y_adj"] = y_adj
        XY_out["theta"] = theta

        return ValidationReport(
            name=self.name,
            status=status,  # type: ignore
            summary=summary,
            details={"theta": float(theta), "var_y": var_y, "var_y_adj": var_y_adj, "variance_reduction": float(vr)},
            tables={"cuped_groups": XY_out},
        )
