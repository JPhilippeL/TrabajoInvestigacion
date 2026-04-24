"""
@file egnn_metrics.py
@author Mohamed EL BOUKHIARI
@brief Utility functions for EGNN metrics handling.
@details
This file centralizes helper functions related to metric reading and formatting.
"""

from __future__ import annotations

import os
import pandas as pd


def load_metrics_summary(results_dir: str) -> dict:
    """
    @brief Load the metrics_summary.csv file from the results directory.
    @param results_dir Results directory path.
    @return Dictionary containing mean and std values for RMSE, Pearson and Spearman.
    """
    summary_path = os.path.join(results_dir, "metrics_summary.csv")
    df = pd.read_csv(summary_path)

    metrics = {}
    for _, row in df.iterrows():
        metrics[row["Metric"]] = {
            "mean": row["Mean"],
            "std": row["Std"],
        }

    return metrics
