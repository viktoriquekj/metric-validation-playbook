from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, ValidationReport
from mvplaybook.metrics.base import Metric


@dataclass(frozen=True)
class Validator(ABC):
    """
    Base class for all validation checks.

    Pipeline usage:
      report = validator.run(df, metric, metric_result, ctx)

    Design principles:
      - Validators are metric-agnostic. They operate on MetricResult.metric_df
        and/or raw df if needed.
      - Validators return ValidationReport with status + details.
      - Validators should never raise on 'bad metric' (use FAIL/WARN),
        unless it's a programmer error (missing columns, invalid assumptions).
    """

    name: str

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        metric: Metric,
        result: MetricResult,
        ctx: RunContext,
    ) -> ValidationReport:
        raise NotImplementedError

    # ---- helper: ensure metric result is valid before running checks ----
    def _validate_metric_result(self, result: MetricResult) -> None:
        result.validate()
