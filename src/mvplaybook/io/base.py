from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from mvplaybook.core.context import RunContext


@dataclass(frozen=True)
class DatasetMetadata:
    """Lightweight dataset description returned by adapters."""
    name: str
    time_col: Optional[str]
    segment_cols: List[str]
    id_cols: List[str]
    notes: Dict[str, Any]


class DatasetAdapter(ABC):
    """
    DatasetAdapter isolates dataset-specific logic:
      - reading raw files
      - schema validation
      - basic type normalization (e.g., parse datetimes)
      - providing metadata: time_col, segment_cols

    Pipeline will do:
      df = adapter.load(ctx)
      adapter.validate(df, ctx)
      meta = adapter.metadata(ctx)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short adapter name used in configs (e.g., 'opensky', 'faers')."""
        raise NotImplementedError

    @abstractmethod
    def load(self, ctx: RunContext) -> pd.DataFrame:
        """Load raw data and return a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def required_columns(self) -> List[str]:
        """Columns required for this dataset adapter to function."""
        raise NotImplementedError

    def validate(self, df: pd.DataFrame, ctx: RunContext) -> None:
        """Default schema validation: required columns exist."""
        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{self.name}] Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)[:50]}..."
            )

    def metadata(self, ctx: RunContext) -> DatasetMetadata:
        """Default metadata; adapters can override."""
        return DatasetMetadata(
            name=self.name,
            time_col=ctx.time_col,
            segment_cols=[],
            id_cols=[],
            notes={},
        )

    # ---- helpers ----
    def _resolve_path(self, ctx: RunContext, path_like: str | Path) -> Path:
        """Resolve paths relative to project_root."""
        p = Path(path_like)
        if p.is_absolute():
            return p
        return (ctx.project_root / p).resolve()
