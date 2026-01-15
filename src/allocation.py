from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.utils import get_logger


def allocate_revenue(
    timesheet_task_month: pd.DataFrame,
    revenue_monthly: pd.DataFrame,
    unallocated_task_name: str,
) -> pd.DataFrame:
    """Allocate job-month revenue to job-task-month by hours share."""
    logger = get_logger()

    base = timesheet_task_month.copy()
    revenue = revenue_monthly.copy()

    base = base.merge(
        revenue,
        on=["job_no", "month_key"],
        how="left",
        suffixes=("", "_revenue"),
    )
    if "revenue_monthly" not in base.columns:
        base["revenue_monthly"] = 0
    base["revenue_monthly"] = base["revenue_monthly"].fillna(0)

    total_hours = (
        base.groupby(["job_no", "month_key"], dropna=False)["total_hours"]
        .sum()
        .reset_index()
        .rename(columns={"total_hours": "total_hours_job_month"})
    )

    base = base.merge(total_hours, on=["job_no", "month_key"], how="left")
    base["task_share"] = 0.0
    has_hours = base["total_hours_job_month"] > 0
    base.loc[has_hours, "task_share"] = (
        base.loc[has_hours, "total_hours"] / base.loc[has_hours, "total_hours_job_month"]
    )
    base["revenue_allocated"] = base["task_share"] * base["revenue_monthly"]
    base["is_quote_only_task"] = False

    revenue_only = revenue.merge(total_hours, on=["job_no", "month_key"], how="left")
    revenue_only["total_hours_job_month"] = revenue_only["total_hours_job_month"].fillna(0)
    needs_unallocated = revenue_only["total_hours_job_month"] <= 0

    if needs_unallocated.any():
        unalloc = revenue_only.loc[needs_unallocated].copy()
        unalloc["task_name"] = unallocated_task_name
        unalloc["task_name_raw"] = unallocated_task_name
        unalloc["total_hours"] = 0.0
        unalloc["billable_hours"] = 0.0
        unalloc["onshore_hours"] = 0.0
        unalloc["total_cost"] = 0.0
        unalloc["avg_base_rate"] = 0.0
        unalloc["avg_billable_rate"] = 0.0
        unalloc["distinct_staff_count"] = 0
        unalloc["task_share"] = 0.0
        unalloc["revenue_allocated"] = unalloc["revenue_monthly"]
        unalloc["is_unallocated_row"] = True

        for col in base.columns:
            if col not in unalloc.columns:
                unalloc[col] = None
        base["is_unallocated_row"] = False
        base = pd.concat([base, unalloc[base.columns]], ignore_index=True)
    else:
        base["is_unallocated_row"] = False

    logger.info("Revenue allocation rows: %s", len(base))
    return base
