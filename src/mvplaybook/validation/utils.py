from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)

    ref_p = np.clip(ref_p, eps, 1)
    cur_p = np.clip(cur_p, eps, 1)

    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))


def ensure_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")
