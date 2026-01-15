from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.utils import ensure_unique, fuzzy_task_suggestions, get_logger


def build_qa_report(
    fact: pd.DataFrame,
    revenue_monthly: pd.DataFrame,
    timesheet_task_month: pd.DataFrame,
    quote_task: pd.DataFrame,
    tolerance: float,
) -> Dict[str, object]:
    """Generate QA checks and summary metrics."""
    logger = get_logger()

    allocation_check = (
        fact.groupby(["job_no", "month_key"], dropna=False)
        .agg(revenue_monthly=("revenue_monthly", "first"), revenue_allocated=("revenue_allocated", "sum"))
        .reset_index()
    )
    allocation_check["delta"] = allocation_check["revenue_monthly"] - allocation_check["revenue_allocated"]
    allocation_check["within_tolerance"] = allocation_check["delta"].abs() <= tolerance

    allocation_ok = bool(allocation_check["within_tolerance"].all())
    max_delta = float(allocation_check["delta"].abs().max()) if not allocation_check.empty else 0.0

    unique_keys_ok = ensure_unique(fact, ["job_no", "task_name", "month_key"])

    missing_rate_count = int(timesheet_task_month.get("missing_base_rate_flag", pd.Series(dtype=bool)).sum())
    negative_hours_count = int(timesheet_task_month.get("had_negative_hours_flag", pd.Series(dtype=bool)).sum())

    mixed_dimension_cols = [c for c in timesheet_task_month.columns if c.startswith("mixed_dimension_")]
    mixed_dimension_counts = {
        col: int(timesheet_task_month[col].sum()) for col in mixed_dimension_cols
    }

    timesheet_tasks = timesheet_task_month[["job_no", "task_name"]].drop_duplicates()
    quote_tasks = quote_task[["job_no", "task_name"]].drop_duplicates()
    unmatched_timesheet = timesheet_tasks.merge(quote_tasks, on=["job_no", "task_name"], how="left", indicator=True)
    unmatched_timesheet = unmatched_timesheet[unmatched_timesheet["_merge"].ne("both")]

    suggestions = fuzzy_task_suggestions(
        unmatched_timesheet["task_name"].tolist(), quote_tasks["task_name"].tolist()
    )

    mismatch_counts = []
    top_mismatch_by_hours = []
    top_mismatch_by_revenue = []
    mismatch_revenue_by_month = []
    mixed_department_share = 0.0
    if "dept_match_status" in fact.columns:
        mismatch = fact[fact["dept_match_status"] == "MISMATCH"].copy()
        if not mismatch.empty:
            mismatch_counts = (
                mismatch.groupby(["department_actual", "department_quote"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(20)
                .to_dict(orient="records")
            )
            top_mismatch_by_hours = (
                mismatch.groupby(["job_no", "task_name"], dropna=False)
                .agg(total_hours=("total_hours", "sum"))
                .reset_index()
                .sort_values("total_hours", ascending=False)
                .head(20)
                .to_dict(orient="records")
            )
            top_mismatch_by_revenue = (
                mismatch.groupby(["job_no", "task_name"], dropna=False)
                .agg(revenue_allocated=("revenue_allocated", "sum"))
                .reset_index()
                .sort_values("revenue_allocated", ascending=False)
                .head(20)
                .to_dict(orient="records")
            )
            mismatch_month = (
                mismatch.groupby("month_key", dropna=False)["revenue_allocated"]
                .sum()
                .reset_index()
                .rename(columns={"revenue_allocated": "mismatch_revenue"})
            )
            total_month = (
                fact.groupby("month_key", dropna=False)["revenue_allocated"]
                .sum()
                .reset_index()
                .rename(columns={"revenue_allocated": "total_revenue"})
            )
            mismatch_month = mismatch_month.merge(total_month, on="month_key", how="left")
            mismatch_month["mismatch_revenue_share"] = mismatch_month.apply(
                lambda r: float(r["mismatch_revenue"] / r["total_revenue"]) if r["total_revenue"] else 0.0,
                axis=1,
            )
            mismatch_revenue_by_month = mismatch_month.to_dict(orient="records")
        if "mixed_department" in fact.columns:
            total_hours = fact["total_hours"].sum()
            mixed_hours = fact.loc[fact["mixed_department"], "total_hours"].sum()
            mixed_department_share = float(mixed_hours / total_hours) if total_hours else 0.0

    report = {
        "allocation_ok": allocation_ok,
        "allocation_max_delta": max_delta,
        "unique_keys_ok": unique_keys_ok,
        "missing_base_rate_groups": missing_rate_count,
        "negative_hours_groups": negative_hours_count,
        "mixed_dimension_counts": mixed_dimension_counts,
        "unmatched_timesheet_tasks": int(len(unmatched_timesheet)),
        "task_match_suggestions": suggestions,
        "dept_mismatch_counts": mismatch_counts,
        "dept_mismatch_top_by_hours": top_mismatch_by_hours,
        "dept_mismatch_top_by_revenue": top_mismatch_by_revenue,
        "dept_mismatch_revenue_by_month": mismatch_revenue_by_month,
        "mixed_department_share_hours": mixed_department_share,
    }

    logger.info("QA allocation ok: %s", allocation_ok)
    return report
