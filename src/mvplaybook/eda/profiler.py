
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from mvplaybook.core.context import RunContext
from mvplaybook.eda.plots import barplot_counts_and_rates


TypeSpec = Union[str, Any]  # str specs or pandas/numpy dtype objects


@dataclass(frozen=True)
class EDAArtifacts:
    """Return object for EDA functions that produce multiple artifacts."""
    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, Path]
    notes: Dict[str, Any]


class EDAProfiler:
    """
    General, reusable EDA utilities.
    Design principles:
      - Works on any DataFrame (dataset-agnostic)
      - Produces tables + optional plots
      - Does NOT mutate the original df unless explicitly requested

    Typical usage in pipeline/notebook:
      df = adapter.load(ctx)
      EDAProfiler.overview(df)

      df = EDAProfiler.cast_types(df, type_map)  # returns a new df

      dup = EDAProfiler.duplicates_report(df, subset_cols=[...])
      miss = EDAProfiler.missingness_report(df, cols=[...])
      grp = EDAProfiler.missingness_by_group(df, group_cols=[...], target_col="...")
    """

    # -------- 1) Overview --------
    @staticmethod
    def overview(df: pd.DataFrame, n_head: int = 5, print_out: bool = True) -> Dict[str, Any]:
        """
        Prints and/or returns:
          - shape
          - columns + dtypes
          - head
        """
        info = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(n_head),
        }

        if print_out:
            print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
            print("\nColumns & dtypes:")
            display_df = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
            print(display_df.to_string(index=False))
            print("\nHead:")
            try:
                # in notebooks, this looks nicer
                from IPython.display import display
                display(df.head(n_head))
            except Exception:
                print(df.head(n_head))

        return 

    # -------- 2) Type casting --------
    @staticmethod
    def cast_types(
        df: pd.DataFrame,
        type_map: Dict[str, TypeSpec],
        copy: bool = True,
        errors: str = "coerce",
    ) -> pd.DataFrame:
        """
        Cast columns based on a declarative type_map.
        Supported string specs:
          - "datetime"        -> pd.to_datetime(utc=False)
          - "datetime_utc"    -> pd.to_datetime(utc=True)
          - "date"            -> datetime_utc floored to day (UTC)
          - "int" / "float"   -> pd.to_numeric
          - "string"          -> pandas string dtype
          - "string_upper"    -> pandas string + upper()
          - "category"        -> category dtype
          - "bool"            -> bool (best-effort)

        Returns a new df by default.
        """
        out = df.copy() if copy else df

        for col, spec in type_map.items():
            if col not in out.columns:
                raise ValueError(f"cast_types: column '{col}' not in dataframe.")

            if isinstance(spec, str):
                s = spec.lower().strip()

                if s in ("datetime", "datetime64", "timestamp"):
                    out[col] = pd.to_datetime(out[col], utc=False, errors=errors)

                elif s in ("datetime_utc", "timestamp_utc"):
                    out[col] = pd.to_datetime(out[col], utc=True, errors=errors)

                elif s == "date":
                    out[col] = pd.to_datetime(out[col], utc=True, errors=errors).dt.floor("D")

                elif s in ("int", "int64", "integer"):
                    out[col] = pd.to_numeric(out[col], errors=errors).astype("Int64")

                elif s in ("float", "float64", "double"):
                    out[col] = pd.to_numeric(out[col], errors=errors).astype("Float64")

                elif s == "string":
                    out[col] = out[col].astype("string")

                elif s == "string_upper":
                    out[col] = out[col].astype("string").str.upper()

                elif s == "category":
                    out[col] = out[col].astype("category")

                elif s in ("bool", "boolean"):
                    # best-effort: common encodings
                    series = out[col]
                    if series.dtype == bool:
                        out[col] = series
                    else:
                        out[col] = series.astype("string").str.lower().map(
                            {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
                        ).astype("boolean")

                else:
                    raise ValueError(f"cast_types: unsupported type spec '{spec}' for column '{col}'.")

            else:
                # If user passed an actual dtype/class
                out[col] = out[col].astype(spec)

        return out

    # -------- 3a) Duplicates --------
    @staticmethod
    def duplicates_report(
        df: pd.DataFrame,
        subset_cols: Sequence[str],
        keep: str = "first",
        sample_n: int = 20,
    ) -> Dict[str, Any]:
        """
        Returns:
          - duplicate_mask
          - n_duplicates
          - n_unique_by_col (for subset cols)
          - duplicated_rows_sample
        """
        for c in subset_cols:
            if c not in df.columns:
                raise ValueError(f"duplicates_report: column '{c}' not in dataframe.")

        dup_mask = df.duplicated(subset=list(subset_cols), keep=keep)
        n_dup = int(dup_mask.sum())

        n_unique = {c: int(df[c].nunique(dropna=False)) for c in subset_cols}

        dup_rows = df.loc[dup_mask].head(sample_n)

        return {
            "subset_cols": list(subset_cols),
            "n_rows": int(len(df)),
            "n_duplicates": n_dup,
            "duplicate_rate": float(n_dup / max(len(df), 1)),
            "n_unique_by_col": n_unique,
            "duplicated_rows_sample": dup_rows,
        }

    # -------- 3b) Missingness (overall) --------
    @staticmethod
    def missingness_report(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with:
          column, null_count, null_rate
        """
        use_cols = list(cols) if cols is not None else list(df.columns)
        for c in use_cols:
            if c not in df.columns:
                raise ValueError(f"missingness_report: column '{c}' not in dataframe.")

        n = len(df)
        null_count = df[use_cols].isna().sum()
        out = pd.DataFrame({
            "column": null_count.index,
            "null_count": null_count.values.astype(int),
            "null_rate": (null_count.values / max(n, 1)),
        }).sort_values("null_rate", ascending=False)

        return out.reset_index(drop=True)

    # -------- 3c) Missingness by group (plots or tables) --------
    @staticmethod
    def missingness_by_group(
        df: pd.DataFrame,
        group_cols: Sequence[str],
        target_col: str,
        ctx: Optional[RunContext] = None,
        max_categories_for_plot: int = 12,
        save_prefix: str = "missingness_by_group",
    ) -> EDAArtifacts:
        """
        For each group column (and also for the combined group key), compute:
          - null_count(target_col) by group
          - null_rate(target_col) by group (null_count / group_size)

        Behavior:
          - If group has <= max_categories_for_plot unique values: create two-panel bar chart.
          - If > max_categories_for_plot: return a table instead (no plot).

        If ctx is provided, plots are saved under ctx.figures_dir().
        Returns EDAArtifacts with tables + figure paths.
        """
        for c in group_cols:
            if c not in df.columns:
                raise ValueError(f"missingness_by_group: group column '{c}' not in dataframe.")
        if target_col not in df.columns:
            raise ValueError(f"missingness_by_group: target column '{target_col}' not in dataframe.")

        tables: Dict[str, pd.DataFrame] = {}
        figures: Dict[str, Path] = {}
        notes: Dict[str, Any] = {"target_col": target_col, "group_cols": list(group_cols)}

        def _summary_for_groupkey(group_key: pd.Series, label: str) -> pd.DataFrame:
            tmp = pd.DataFrame({"_group": group_key, "_is_null": df[target_col].isna()})
            g = tmp.groupby("_group", dropna=False)["_is_null"].agg(
                null_count="sum",
                group_size="count",
            ).reset_index()
            g["null_count"] = g["null_count"].astype(int)
            g["group_size"] = g["group_size"].astype(int)
            g["null_rate"] = g["null_count"] / g["group_size"].clip(lower=1)
            g = g.rename(columns={"_group": label})
            return g.sort_values("null_count", ascending=False).reset_index(drop=True)

        # 1) Individual group columns
        for gc in group_cols:
            table = _summary_for_groupkey(df[gc].astype("string").fillna("<NA>"), gc)
            tables[f"group::{gc}"] = table

            save_path = None
            if ctx is not None:
                ctx.ensure_output_dirs()
                save_path = ctx.figures_dir() / f"{save_prefix}__{gc}__target_{target_col}.png"

            fig_path, mode = barplot_counts_and_rates(
                df=table,
                x_col=gc,
                count_col="null_count",
                rate_col="null_rate",
                title_left=f"Null count of '{target_col}' by {gc}",
                title_right=f"Null rate of '{target_col}' by {gc}",
                xlabel=str(gc),
                max_categories_for_plot=max_categories_for_plot,
                save_path=save_path,
            )

            if mode == "plot" and fig_path is not None:
                figures[f"fig::{gc}"] = fig_path
            else:
                notes[f"note::{gc}"] = f"Too many categories ({table[gc].nunique()}) → returned table only."

        # 2) Combined group key (optional but useful)
        if len(group_cols) > 1:
            combined = df[list(group_cols)].astype("string").fillna("<NA>").agg(" | ".join, axis=1)
            combo_name = " | ".join(group_cols)
            table = _summary_for_groupkey(combined, combo_name)
            tables[f"group::COMBINED({combo_name})"] = table

            save_path = None
            if ctx is not None:
                ctx.ensure_output_dirs()
                save_path = ctx.figures_dir() / f"{save_prefix}__COMBINED__target_{target_col}.png"

            fig_path, mode = barplot_counts_and_rates(
                df=table,
                x_col=combo_name,
                count_col="null_count",
                rate_col="null_rate",
                title_left=f"Null count of '{target_col}' by {combo_name}",
                title_right=f"Null rate of '{target_col}' by {combo_name}",
                xlabel=combo_name,
                max_categories_for_plot=max_categories_for_plot,
                save_path=save_path,
            )

            if mode == "plot" and fig_path is not None:
                figures["fig::COMBINED"] = fig_path
            else:
                notes["note::COMBINED"] = f"Too many categories ({table[combo_name].nunique()}) → returned table only."

        return EDAArtifacts(tables=tables, figures=figures, notes=notes)
