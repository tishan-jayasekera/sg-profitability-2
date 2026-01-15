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

    report = {
        "allocation_ok": allocation_ok,
        "allocation_max_delta": max_delta,
        "unique_keys_ok": unique_keys_ok,
        "missing_base_rate_groups": missing_rate_count,
        "negative_hours_groups": negative_hours_count,
        "mixed_dimension_counts": mixed_dimension_counts,
        "unmatched_timesheet_tasks": int(len(unmatched_timesheet)),
        "task_match_suggestions": suggestions,
    }

    logger.info("QA allocation ok: %s", allocation_ok)
    return report
