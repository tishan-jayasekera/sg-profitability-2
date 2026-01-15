from __future__ import annotations

from typing import Dict

import pandas as pd

from src.utils import standardize_job_no, standardize_task_name, to_month_start


def add_revenue_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize revenue keys and month column."""
    df = df.copy()
    df["job_no"] = df["Job Number"].apply(standardize_job_no)
    df["month_key"] = to_month_start(df["Month"])
    return df


def add_timesheet_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timesheet keys and month column."""
    df = df.copy()
    df["job_no"] = df["[Job] Job No."].apply(standardize_job_no)
    df["task_name_raw"] = df["[Job Task] Name"].apply(standardize_task_name)
    df["task_name"] = df["task_name_raw"].apply(standardize_task_name)
    df["month_key"] = to_month_start(df["Month Key"])
    return df


def add_quotation_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize quotation keys."""
    df = df.copy()
    df["job_no"] = df["[Job] Job No."].apply(standardize_job_no)
    df["task_name_raw"] = df["[Job Task] Name"].apply(standardize_task_name)
    df["task_name"] = df["task_name_raw"].apply(standardize_task_name)
    return df


def apply_dimension_modes(df: pd.DataFrame, dimensions: Dict[str, str]) -> pd.DataFrame:
    """Apply dimension columns to dataframe."""
    df = df.copy()
    for dest, src in dimensions.items():
        df[dest] = df[src]
    return df
