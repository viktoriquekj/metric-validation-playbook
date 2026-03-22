from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.core.result import MetricResult, MetricSpec


@dataclass(frozen=True)
class Metric(ABC):
    """
    A Metric defines *what we measure* and how we compute it.

    Pipeline usage:
      result = metric.compute(df, ctx)
      result.validate()

    Validators will use:
      - result.metric_df (tidy)
      - result.spec (definition + group keys + value column)
      - result.artifacts (optional extras)
    """

    name: str
    description: str

    # What defines the unit of analysis
    group_keys: List[str] = field(default_factory=list)

    # Optional time column used for time-based metrics (e.g., daily/weekly)
    time_col: Optional[str] = None

    # Standard output column names
    value_col: str = "metric_value"
    count_col: str = "n"

    # Any extra spec info (weights, thresholds, guardrails, etc.)
    spec_extras: Dict[str, Any] = field(default_factory=dict)

    # ---------- Schema contract ----------
    @property
    @abstractmethod
    def required_columns(self) -> Sequence[str]:
        """Columns required in the input df to compute this metric."""
        raise NotImplementedError

    # ---------- Computation ----------
    @abstractmethod
    def compute(self, df: pd.DataFrame, ctx: RunContext) -> MetricResult:
        """Compute the metric and return MetricResult."""
        raise NotImplementedError

    # ---------- Helpers ----------
    def validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"[Metric:{self.name}] Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)[:50]}..."
            )

    def make_spec(self, ctx: RunContext) -> MetricSpec:
        """Build a MetricSpec snapshot for reproducible reporting."""
        # Prefer ctx.time_col if metric time_col not explicitly set
        time_col = self.time_col #or ctx.time_col
        return MetricSpec(
            name=self.name,
            description=self.description,
            group_keys=list(self.group_keys),
            time_col=time_col,
            value_col=self.value_col,
            count_col=self.count_col,
            extras=dict(self.spec_extras),
        )

    def _coerce_time_col(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        """Ensure time column is datetime-like for time metrics."""
        return pd.to_datetime(df[time_col], utc=True, errors="coerce")

    def compute_variant(self, df: pd.DataFrame, ctx: RunContext, variant: Dict[str, Any]) -> MetricResult:
        """
        Optional: compute the metric with a modified definition (variant).
        Default: not supported → validators that need it should SKIP.
        """
        raise NotImplementedError("This metric does not support variants.")



