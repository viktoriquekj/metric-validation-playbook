from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.io.base import DatasetAdapter, DatasetMetadata


@dataclass(frozen=True)
class OpenSkyAdapter(DatasetAdapter):
    """
    Loads OpenSky 'flights' (arrival/departure) dataset you built.

    Expects the config to provide:
      - extras['data_path'] : str (csv/parquet path relative to repo root or absolute)

    Recommended standard columns (based on your snippet):
      kind, icao24, callsign, firstSeen, lastSeen,
      estDepartureAirport, estArrivalAirport,
      firstSeen_utc, lastSeen_utc, date_utc,
      flight_key, obs_duration_min, callsign_prefix
    """

    @property
    def name(self) -> str:
        return "opensky"

    def required_columns(self) -> List[str]:
        return [
            "kind",
            "icao24",
            "callsign",
            "firstSeen_utc",
            "lastSeen_utc",
            "estDepartureAirport",
            "estArrivalAirport",
            "date_utc",
            "flight_key",
            "obs_duration_min",
            "callsign_prefix",
        ]

    def load(self, ctx: RunContext) -> pd.DataFrame:
        data_path = ctx.extras.get("data_path")
        if not data_path:
            raise ValueError(
                "[opensky] Missing ctx.extras['data_path']. "
                "Provide it in config dict/YAML under 'data_path'."
            )
        path = self._resolve_path(ctx, data_path)

        # Basic loader: infer by suffix
        if path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        elif path.suffix.lower() in [".csv"]:
            df = pd.read_csv(path)
        else:
            raise ValueError(f"[opensky] Unsupported file type: {path.suffix} for {path}")

        # Normalize dtypes
        # Ensure datetime columns are parsed
        for col in ["firstSeen_utc", "lastSeen_utc"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

        # Ensure date_utc is date-like (keep as string or datetime64 - either is fine,
        # but standardize to datetime64[ns, UTC] normalized to date boundary)
        if "date_utc" in df.columns:
            # If it's already a date string 'YYYY-MM-DD' this will parse nicely
            df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce").dt.floor("D")

        # Standardize airports/callsign_prefix to uppercase strings
        for col in ["estDepartureAirport", "estArrivalAirport", "callsign_prefix", "kind"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.upper()

        # obs_duration_min numeric
        if "obs_duration_min" in df.columns:
            df["obs_duration_min"] = pd.to_numeric(df["obs_duration_min"], errors="coerce")

        # If ctx.time_col not set, set a sensible default for this dataset
        if ctx.time_col is None:
            # Note: RunContext is frozen, so we don't mutate it. Pipeline can enforce time_col.
            pass

        return df

    def metadata(self, ctx: RunContext) -> DatasetMetadata:
        # Provide sensible defaults for segmentation
        segment_cols = ["callsign_prefix", "kind", "estDepartureAirport", "estArrivalAirport"]
        id_cols = ["flight_key", "icao24", "callsign"]

        return DatasetMetadata(
            name=self.name,
            time_col=ctx.time_col or "firstSeen_utc",
            segment_cols=segment_cols,
            id_cols=id_cols,
            notes={"airport_focus": "EKCH"},
        )
