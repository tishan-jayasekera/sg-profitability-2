from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import coalesce


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df))


def build_fact_table(
    allocated: pd.DataFrame,
    quote_task: pd.DataFrame,
) -> pd.DataFrame:
    """Combine allocation results with quotation data and compute profitability metrics."""
    allocated = allocated.copy()
    quote_task = quote_task.copy()

    merged = allocated.merge(
        quote_task,
        on=["job_no", "task_name"],
        how="left",
        indicator=True,
        suffixes=("", "_quote"),
    )

    merged["is_unquoted_task"] = merged["_merge"].ne("both")
    merged = merged.drop(columns=["_merge"])

    merged["quoted_time"] = merged["quoted_time"].fillna(0)
    merged["quoted_amount"] = merged["quoted_amount"].fillna(0)

    merged["gross_profit"] = merged["revenue_allocated"] - merged["total_cost"]
    merged["margin"] = np.where(
        merged["revenue_allocated"] != 0,
        merged["gross_profit"] / merged["revenue_allocated"],
        0,
    )

    merged["quote_hour_variance"] = merged["total_hours"] - merged["quoted_time"]

    total_task_hours = (
        merged.groupby(["job_no", "task_name"], dropna=False)["total_hours"]
        .sum()
        .reset_index()
        .rename(columns={"total_hours": "total_hours_task"})
    )
    merged = merged.merge(total_task_hours, on=["job_no", "task_name"], how="left")
    merged["quote_amount_allocated"] = np.where(
        merged["total_hours_task"] > 0,
        merged["quoted_amount"] * (merged["total_hours"] / merged["total_hours_task"]),
        0,
    )
    merged["quote_amount_variance"] = merged["revenue_allocated"] - merged["quote_amount_allocated"]

    merged["is_unallocated_row"] = merged.get("is_unallocated_row", False)
    merged["is_unworked_task"] = (merged["quoted_time"] > 0) & (merged["total_hours"] <= 0)

    merged["client"] = coalesce(_col(merged, "client"), _col(merged, "Client"))
    merged["category"] = coalesce(_col(merged, "job_category"), _col(merged, "Category"), _col(merged, "category"))
    merged["department"] = coalesce(_col(merged, "department"), _col(merged, "Department"))

    return merged


def build_job_month_summary(fact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics at job-month level."""
    summary = (
        fact.groupby(["job_no", "month_key"], dropna=False)
        .agg(
            revenue_monthly=("revenue_monthly", "first"),
            revenue_allocated=("revenue_allocated", "sum"),
            cost_month=("total_cost", "sum"),
            hours_month=("total_hours", "sum"),
        )
        .reset_index()
    )
    summary["gp_month"] = summary["revenue_allocated"] - summary["cost_month"]
    summary["margin_month"] = np.where(
        summary["revenue_allocated"] != 0,
        summary["gp_month"] / summary["revenue_allocated"],
        0,
    )
    return summary


def build_job_total_summary(fact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics at job level."""
    summary = (
        fact.groupby(["job_no"], dropna=False)
        .agg(
            revenue_allocated=("revenue_allocated", "sum"),
            total_cost=("total_cost", "sum"),
            total_hours=("total_hours", "sum"),
            quoted_time=("quoted_time", "sum"),
            quoted_amount=("quoted_amount", "sum"),
        )
        .reset_index()
    )
    summary["gross_profit"] = summary["revenue_allocated"] - summary["total_cost"]
    summary["margin"] = np.where(
        summary["revenue_allocated"] != 0,
        summary["gross_profit"] / summary["revenue_allocated"],
        0,
    )
    summary["utilization_vs_quote"] = np.where(
        summary["quoted_time"] != 0,
        summary["total_hours"] / summary["quoted_time"],
        0,
    )
    return summary


def build_quote_vs_actual_summary(fact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics at job-task level across months."""
    summary = (
        fact.groupby(["job_no", "task_name"], dropna=False)
        .agg(
            total_hours=("total_hours", "sum"),
            quoted_time=("quoted_time", "first"),
            quoted_amount=("quoted_amount", "first"),
            revenue_allocated=("revenue_allocated", "sum"),
        )
        .reset_index()
    )
    summary["utilization_vs_quote"] = np.where(
        summary["quoted_time"] != 0,
        summary["total_hours"] / summary["quoted_time"],
        0,
    )
    return summary
