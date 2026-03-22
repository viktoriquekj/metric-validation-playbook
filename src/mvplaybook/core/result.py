from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from enum import Enum



@dataclass(frozen=True)
class MetricSpec:
    """
    A reproducible snapshot of what the metric is.

    Stored in MetricResult so reports can include:
    - definition
    - group keys
    - value column name
    - guardrails
    """
    name: str
    description: str
    group_keys: List[str]
    time_col: Optional[str]
    value_col: str = "metric_value"
    count_col: str = "n"
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """
    Standard output of Metric.compute().

    metric_df should be tidy and validator-friendly:
      columns:
        - group keys (MetricSpec.group_keys)
        - optional time column (MetricSpec.time_col) if metric is time-based
        - metric value column (MetricSpec.value_col)
        - count column (MetricSpec.count_col) at minimum (support / sample size)
      plus any additional columns the metric wants to include.

    artifacts: extra objects (optional) like:
      - intermediate aggregations
      - distribution summaries
      - thresholds used
    """
    metric_df: pd.DataFrame
    spec: MetricSpec
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Fail fast if metric_df does not match the spec contract."""
        missing = []
        for c in self.spec.group_keys:
            if c not in self.metric_df.columns:
                missing.append(c)

        if self.spec.time_col is not None and self.spec.time_col not in self.metric_df.columns:
            missing.append(self.spec.time_col)

        for required in [self.spec.value_col, self.spec.count_col]:
            if required not in self.metric_df.columns:
                missing.append(required)

        if missing:
            raise ValueError(
                f"MetricResult.metric_df missing required columns: {sorted(set(missing))}. "
                f"Have: {list(self.metric_df.columns)}"
            )

        # Basic sanity: count should be non-negative
        if (self.metric_df[self.spec.count_col] < 0).any():
            raise ValueError("MetricResult contains negative counts, which is invalid.")
        



class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class ValidationReport:
    """
    Standard output for any Validator.

    status:
      - PASS: check is good
      - WARN: potential issue / investigate
      - FAIL: serious issue / metric not trustworthy for intended use
      - SKIP: not applicable / insufficient data

    summary: one-liner for the final report.
    details: structured results (numbers, tables, flags).
    tables: optional DataFrames (kept in-memory or saved by ReportBuilder).
    figures: optional list of saved figure paths (strings or Paths).
    warnings: any additional messages.
    """
    name: str
    status: CheckStatus
    summary: str

    details: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_ok(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN)

    @staticmethod
    def pass_(name: str, summary: str, **kwargs) -> "ValidationReport":
        return ValidationReport(name=name, status=CheckStatus.PASS, summary=summary, **kwargs)

    @staticmethod
    def warn(name: str, summary: str, **kwargs) -> "ValidationReport":
        return ValidationReport(name=name, status=CheckStatus.WARN, summary=summary, **kwargs)

    @staticmethod
    def fail(name: str, summary: str, **kwargs) -> "ValidationReport":
        return ValidationReport(name=name, status=CheckStatus.FAIL, summary=summary, **kwargs)

    @staticmethod
    def skip(name: str, summary: str, **kwargs) -> "ValidationReport":
        return ValidationReport(name=name, status=CheckStatus.SKIP, summary=summary, **kwargs)

