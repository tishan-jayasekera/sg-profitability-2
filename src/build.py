from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src import allocation, io, metrics, qa, quotation, revenue, timesheet
from src.utils import get_logger, load_settings, read_task_name_map, write_json


@dataclass
class BuildResult:
    revenue_monthly: pd.DataFrame
    timesheet_task_month: pd.DataFrame
    quote_task: pd.DataFrame
    fact: pd.DataFrame
    job_month_summary: pd.DataFrame
    job_total_summary: pd.DataFrame
    quote_vs_actual_summary: pd.DataFrame
    qa_report: Dict[str, object]


def build_dataset(
    input_path: str | Path,
    fy: str,
    include_all_history: bool = False,
    settings_path: str | Path = "config/settings.yaml",
) -> BuildResult:
    """Run the end-to-end dataset build."""
    logger = get_logger()
    settings = load_settings(settings_path)

    sheets = io.read_excel_sheets(input_path)
    task_map = read_task_name_map("config/task_name_map.csv")

    revenue_monthly, revenue_qa = revenue.aggregate_revenue(
        sheets[io.SHEET_REVENUE], settings["exclusions"]["truthy_values"]
    )

    timesheet_task_month, timesheet_qa = timesheet.aggregate_timesheet(
        sheets[io.SHEET_TIMESHEET], task_map
    )

    quote_task, quote_qa = quotation.aggregate_quotation(
        sheets[io.SHEET_QUOTATION], task_map
    )

    if not include_all_history:
        start = pd.to_datetime(settings["months"]["fy26_start"])
        end = pd.to_datetime(settings["months"]["fy26_end"])
        revenue_monthly = revenue_monthly[
            (revenue_monthly["month_key"] >= start) & (revenue_monthly["month_key"] <= end)
        ]

    allocated = allocation.allocate_revenue(
        timesheet_task_month,
        revenue_monthly,
        settings["allocation"]["unallocated_task_name"],
    )

    allocated_keys = allocated[["job_no", "task_name"]].drop_duplicates()
    quote_keys = quote_task[["job_no", "task_name"]].drop_duplicates()
    missing_quote_tasks = quote_keys.merge(
        allocated_keys, on=["job_no", "task_name"], how="left", indicator=True
    )
    missing_quote_tasks = missing_quote_tasks[missing_quote_tasks["_merge"].ne("both")].drop(columns=["_merge"])
    if not missing_quote_tasks.empty:
        synth = missing_quote_tasks.copy()
        synth["month_key"] = pd.NaT
        synth["task_name_raw"] = synth["task_name"]
        synth["total_hours"] = 0.0
        synth["billable_hours"] = 0.0
        synth["onshore_hours"] = 0.0
        synth["total_cost"] = 0.0
        synth["avg_base_rate"] = 0.0
        synth["avg_billable_rate"] = 0.0
        synth["distinct_staff_count"] = 0
        synth["task_share"] = 0.0
        synth["revenue_monthly"] = 0.0
        synth["revenue_allocated"] = 0.0
        synth["is_unallocated_row"] = False
        for col in allocated.columns:
            if col not in synth.columns:
                synth[col] = None
        allocated = pd.concat([allocated, synth[allocated.columns]], ignore_index=True)

    fact = metrics.build_fact_table(allocated, quote_task)
    job_month_summary = metrics.build_job_month_summary(fact)
    job_total_summary = metrics.build_job_total_summary(fact)
    quote_vs_actual_summary = metrics.build_quote_vs_actual_summary(fact)

    qa_report = qa.build_qa_report(
        fact,
        revenue_monthly,
        timesheet_task_month,
        quote_task,
        settings["allocation"]["revenue_tolerance"],
    )

    processed_dir = Path(settings["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    io.save_parquet(revenue_monthly, processed_dir / "revenue_monthly.parquet")
    io.save_parquet(timesheet_task_month, processed_dir / "timesheet_task_month.parquet")
    io.save_parquet(quote_task, processed_dir / "quote_task.parquet")
    io.save_parquet(fact, processed_dir / "fact_job_task_month.parquet")
    io.save_parquet(job_month_summary, processed_dir / "job_month_summary.parquet")
    io.save_parquet(job_total_summary, processed_dir / "job_total_summary.parquet")
    io.save_parquet(quote_vs_actual_summary, processed_dir / "quote_vs_actual_summary.parquet")

    io.save_csv(revenue_monthly, processed_dir / "revenue_monthly.csv")
    io.save_csv(timesheet_task_month, processed_dir / "timesheet_task_month.csv")
    io.save_csv(quote_task, processed_dir / "quote_task.csv")
    io.save_csv(fact, processed_dir / "fact_job_task_month.csv")
    io.save_csv(job_month_summary, processed_dir / "job_month_summary.csv")
    io.save_csv(job_total_summary, processed_dir / "job_total_summary.csv")
    io.save_csv(quote_vs_actual_summary, processed_dir / "quote_vs_actual_summary.csv")

    write_json(processed_dir / "qa_report.json", qa_report)

    logger.info("Build completed for %s", fy)

    return BuildResult(
        revenue_monthly=revenue_monthly,
        timesheet_task_month=timesheet_task_month,
        quote_task=quote_task,
        fact=fact,
        job_month_summary=job_month_summary,
        job_total_summary=job_total_summary,
        quote_vs_actual_summary=quote_vs_actual_summary,
        qa_report=qa_report,
    )
