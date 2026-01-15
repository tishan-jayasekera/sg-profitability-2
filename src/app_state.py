from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from src.build import build_dataset
from src.utils import get_logger, load_settings


@st.cache_data(show_spinner=False)
def load_processed(processed_dir: str) -> Dict[str, pd.DataFrame]:
    """Load processed parquet datasets."""
    processed = Path(processed_dir)
    datasets = {
        "revenue_monthly": pd.read_parquet(processed / "revenue_monthly.parquet"),
        "timesheet_task_month": pd.read_parquet(processed / "timesheet_task_month.parquet"),
        "quote_task": pd.read_parquet(processed / "quote_task.parquet"),
        "fact": pd.read_parquet(processed / "fact_job_task_month.parquet"),
        "job_month_summary": pd.read_parquet(processed / "job_month_summary.parquet"),
        "job_total_summary": pd.read_parquet(processed / "job_total_summary.parquet"),
        "quote_vs_actual_summary": pd.read_parquet(processed / "quote_vs_actual_summary.parquet"),
    }
    return datasets


def get_data(
    data_source: str,
    input_path: str,
    fy: str,
    include_all_history: bool,
    settings_path: str = "config/settings.yaml",
) -> Dict[str, pd.DataFrame]:
    """Load data from processed outputs or rebuild from Excel."""
    settings = load_settings(settings_path)
    processed_dir = settings["processed_dir"]
    logger = get_logger()

    if data_source == "Rebuild from Excel":
        logger.info("Rebuilding datasets from Excel")
        result = build_dataset(
            input_path=input_path,
            fy=fy,
            include_all_history=include_all_history,
            settings_path=settings_path,
        )
        return {
            "revenue_monthly": result.revenue_monthly,
            "timesheet_task_month": result.timesheet_task_month,
            "quote_task": result.quote_task,
            "fact": result.fact,
            "job_month_summary": result.job_month_summary,
            "job_total_summary": result.job_total_summary,
            "quote_vs_actual_summary": result.quote_vs_actual_summary,
        }

    try:
        return load_processed(processed_dir)
    except FileNotFoundError:
        st.error(\"Processed data not found. Run the build script or choose 'Rebuild from Excel'.\")\n        st.stop()


def sidebar_base_controls(settings_path: str = "config/settings.yaml") -> Dict[str, object]:
    """Render base sidebar controls and return selections."""
    settings = load_settings(settings_path)
    st.sidebar.title("Filters")

    data_source = st.sidebar.selectbox(
        "Data source",
        ["Processed parquet", "Rebuild from Excel"],
        index=0,
    )
    include_all_history = st.sidebar.checkbox("Include all history", value=False)
    input_path = st.sidebar.text_input("Excel path", value=settings["raw_input_default"])
    fy = st.sidebar.text_input("FY label", value=settings["fy_default"])

    return {
        "data_source": data_source,
        "include_all_history": include_all_history,
        "input_path": input_path,
        "fy": fy,
    }


def sidebar_filters(fact: pd.DataFrame, base: Dict[str, object]) -> Dict[str, object]:
    """Render sidebar controls and return filter selections."""

    month_series = pd.to_datetime(fact["month_key"], errors="coerce")
    min_month = month_series.min()
    max_month = month_series.max()
    if pd.isna(min_month) or pd.isna(max_month):
        min_month = pd.Timestamp("2025-07-01")
        max_month = pd.Timestamp("2026-01-01")

    start_date, end_date = st.sidebar.date_input(
        "Month range",
        value=(min_month.date(), max_month.date()),
        min_value=min_month.date(),
        max_value=max_month.date(),
    )

    department_options = sorted({d for d in fact.get("department", pd.Series(dtype=str)).dropna().unique() if d != ""})
    client_options = sorted({c for c in fact.get("client", pd.Series(dtype=str)).dropna().unique() if c != ""})
    category_options = sorted({c for c in fact.get("category", pd.Series(dtype=str)).dropna().unique() if c != ""})
    job_options = sorted({j for j in fact["job_no"].dropna().unique() if j != ""})

    departments = st.sidebar.multiselect("Department", department_options)
    clients = st.sidebar.multiselect("Client", client_options)
    categories = st.sidebar.multiselect("Category", category_options)
    jobs = st.sidebar.multiselect("Job No", job_options)

    billable_only = st.sidebar.checkbox("Billable-only", value=False)
    onshore_only = st.sidebar.checkbox("Onshore-only", value=False)

    return {
        "data_source": base["data_source"],
        "include_all_history": base["include_all_history"],
        "input_path": base["input_path"],
        "fy": base["fy"],
        "start_date": pd.Timestamp(start_date),
        "end_date": pd.Timestamp(end_date),
        "departments": departments,
        "clients": clients,
        "categories": categories,
        "jobs": jobs,
        "billable_only": billable_only,
        "onshore_only": onshore_only,
    }


def apply_filters(fact: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    """Apply filters to fact table."""
    filtered = fact.copy()
    month_series = pd.to_datetime(filtered["month_key"], errors="coerce")
    month_mask = month_series.isna() | (
        (month_series >= filters["start_date"]) & (month_series <= filters["end_date"])
    )
    filtered = filtered[month_mask]

    if filters["departments"]:
        filtered = filtered[filtered["department"].isin(filters["departments"])]
    if filters["clients"]:
        filtered = filtered[filtered["client"].isin(filters["clients"])]
    if filters["categories"]:
        filtered = filtered[filtered["category"].isin(filters["categories"])]
    if filters["jobs"]:
        filtered = filtered[filtered["job_no"].isin(filters["jobs"])]
    if filters["billable_only"]:
        filtered = filtered[filtered["billable_hours"] > 0]
    if filters["onshore_only"]:
        filtered = filtered[filtered["onshore_hours"] > 0]

    return filtered
