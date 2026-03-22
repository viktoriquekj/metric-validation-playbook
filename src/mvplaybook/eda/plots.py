

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def barplot_counts_and_rates(
    df: pd.DataFrame,
    x_col: str,
    count_col: str,
    rate_col: str,
    title_left: str,
    title_right: str,
    xlabel: str,
    ylabel_left: str = "Null count",
    ylabel_right: str = "Null rate",
    max_categories_for_plot: int = 12,
    save_path: Optional[Path] = None,
) -> Tuple[Optional[Path], str]:
    """
    Creates a two-panel figure:
      left: counts
      right: rates
    If number of categories > max_categories_for_plot, returns (None, 'table') and does NOT plot.

    Returns:
      (saved_figure_path_or_None, mode) where mode is 'plot' or 'table'
    """
    n_cats = df[x_col].nunique(dropna=False)
    if n_cats > max_categories_for_plot:
        return None, "table"

    # Sort by count desc for readability
    plot_df = df.sort_values(count_col, ascending=False)

    # Create figure
    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.bar(plot_df[x_col].astype(str), plot_df[count_col])
    ax1.set_title(title_left)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_left)
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(plot_df[x_col].astype(str), plot_df[rate_col])
    ax2.set_title(title_right)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_right)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return save_path, "plot"

    return None, "plot"
