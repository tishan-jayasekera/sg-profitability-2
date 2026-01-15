from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.clean import add_revenue_keys
from src.utils import get_logger, truthy_flag


def aggregate_revenue(df: pd.DataFrame, truthy_values: list) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate revenue to job-month with filtering for exclusions."""
    logger = get_logger()
    df = add_revenue_keys(df)
    df["is_excluded"] = df["Excluded"].apply(lambda v: truthy_flag(v, truthy_values))

    excluded_count = int(df["is_excluded"].sum())
    filtered = df[~df["is_excluded"]].copy()
    logger.info("Revenue rows excluded: %s", excluded_count)

    meta_cols = [
        "Source",
        "Account Manager",
        "Client",
        "Industry",
        "Category",
        "Department",
        "Client Group",
        "FY",
    ]

    def _mode(series: pd.Series) -> str:
        series = series.dropna()
        if series.empty:
            return ""
        return str(series.mode().iloc[0])

    meta_agg = {col: _mode for col in meta_cols if col in filtered.columns}

    revenue_monthly = (
        filtered.groupby(["job_no", "month_key"], dropna=False)
        .agg({"Amount": "sum", **meta_agg})
        .reset_index()
        .rename(columns={"Amount": "revenue_monthly"})
    )

    qa = {"excluded_rows": excluded_count}
    return revenue_monthly, qa
