from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.clean import add_quotation_keys
from src.utils import get_logger, safe_to_numeric, standardize_department


META_COLUMNS = {
    "client": "[Job] Client",
    "job_name": "[Job] Name",
    "job_category": "[Job] Category",
    "job_status": "[Job] Status",
    "job_start_date": "[Job] Start Date",
    "job_completed_date": "[Job] Completed Date",
    "department_quote": "department_quote",
    "product": "Product",
}


def _mode(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return ""
    return str(series.mode().iloc[0])


def aggregate_quotation(df: pd.DataFrame, map_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate quotation rows to job-task level."""
    logger = get_logger()
    df = add_quotation_keys(df)
    if map_df is not None and not map_df.empty:
        from src.utils import apply_task_name_map

        df = apply_task_name_map(df, map_df, job_col="job_no", task_col="task_name")

    df["quoted_time"] = safe_to_numeric(df.get("[Job Task] Quoted Time", pd.Series(dtype=float))).fillna(0)
    df["quoted_amount"] = safe_to_numeric(df.get("[Job Task] Quoted Amount", pd.Series(dtype=float))).fillna(0)
    df["invoiced_time"] = safe_to_numeric(df.get("[Job Task] Invoiced Time", pd.Series(dtype=float))).fillna(0)
    df["invoiced_amount"] = safe_to_numeric(df.get("[Job Task] Invoiced Amount", pd.Series(dtype=float))).fillna(0)

    df["department_quote"] = df.get("Department", pd.Series(dtype=str)).apply(standardize_department)

    agg_map = {
        "quoted_time": "sum",
        "quoted_amount": "sum",
        "invoiced_time": "sum",
        "invoiced_amount": "sum",
        "task_name_raw": _mode,
    }

    for out_col, src_col in META_COLUMNS.items():
        if src_col in df.columns:
            agg_map[src_col] = _mode

    grouped = (
        df.groupby(["job_no", "task_name"], dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    quote_dept_counts = (
        df.groupby(["job_no", "task_name"], dropna=False)["department_quote"]
        .nunique()
        .reset_index()
        .rename(columns={"department_quote": "quote_mixed_department"})
    )
    quote_dept_counts["quote_mixed_department"] = quote_dept_counts["quote_mixed_department"] > 1
    grouped = grouped.merge(quote_dept_counts, on=["job_no", "task_name"], how="left")

    rename_map = {v: k for k, v in META_COLUMNS.items() if v in grouped.columns}
    grouped = grouped.rename(columns=rename_map)

    qa = {"quote_rows": len(df)}
    logger.info("Quotation rows processed: %s", len(df))
    return grouped, qa
