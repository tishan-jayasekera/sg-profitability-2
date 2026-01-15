from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.clean import add_timesheet_keys
from src.utils import get_logger, safe_to_numeric, standardize_department


DIMENSION_COLUMNS = {
    "function": "Function",
    "category": "[Category] Category",
    "role": "Role",
    "task": "Task",
    "deliverable": "Deliverable",
}


def _mode(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return ""
    return str(series.mode().iloc[0])


def _mixed(series: pd.Series) -> bool:
    return series.dropna().nunique() > 1


def _weighted_mode_info(values: pd.Series, weights: pd.Series) -> Tuple[str, bool, float, str, float]:
    data = pd.DataFrame({"value": values, "weight": weights})
    data = data.dropna(subset=["value"])
    data = data[data["value"].astype(str).str.strip() != ""]
    if data.empty:
        return "", False, 0.0, "", 0.0
    summary = data.groupby("value", dropna=False)["weight"].sum().reset_index()
    summary = summary[summary["weight"] > 0].sort_values("weight", ascending=False)
    if summary.empty:
        return "", False, 0.0, "", 0.0
    total = summary["weight"].sum()
    top_value = str(summary.iloc[0]["value"])
    top_weight = float(summary.iloc[0]["weight"])
    mixed = len(summary) > 1
    second_value = ""
    second_share = 0.0
    if mixed:
        second_value = str(summary.iloc[1]["value"])
        second_share = float(summary.iloc[1]["weight"]) / total if total else 0.0
    top_share = top_weight / total if total else 0.0
    return top_value, mixed, top_share, second_value, second_share


def _weighted_mode(values: pd.Series, weights: pd.Series) -> Tuple[str, bool]:
    top_value, mixed, _, _, _ = _weighted_mode_info(values, weights)
    return top_value, mixed


def aggregate_timesheet(df: pd.DataFrame, map_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate timesheet rows to job-task-month level."""
    logger = get_logger()
    df = add_timesheet_keys(df)
    if map_df is not None and not map_df.empty:
        from src.utils import apply_task_name_map

        df = apply_task_name_map(df, map_df, job_col="job_no", task_col="task_name")

    df["hours_raw"] = safe_to_numeric(df["[Time] Time"])
    negative_hours = int((df["hours_raw"] < 0).sum())
    df["hours"] = df["hours_raw"].clip(lower=0).fillna(0)
    df["had_negative_hours"] = df["hours_raw"] < 0

    df["base_rate"] = safe_to_numeric(df["[Task] Base Rate"]).fillna(0)
    df["billable_rate"] = safe_to_numeric(df["[Task] Billable Rate"]).fillna(0)

    df["missing_base_rate"] = df["base_rate"] <= 0
    missing_base_rate = int(df["missing_base_rate"].sum())

    df["billable_flag"] = df["Billable?"].astype(str).str.upper().str.strip().eq("YES")
    df["onshore_flag"] = df["Onshore"].astype(str).str.strip().isin(["1", "TRUE", "YES"])
    df["department_actual"] = df.get("Department", pd.Series(dtype=str)).apply(standardize_department)

    df["cost"] = df["hours"] * df["base_rate"]
    df["billable_amount_actual"] = df["hours"] * df["billable_rate"]

    agg_map = {
        "hours": "sum",
        "cost": "sum",
        "billable_amount_actual": "sum",
        "billable_flag": "sum",
        "onshore_flag": "sum",
        "[Staff] Name": pd.Series.nunique,
    }

    grouped = df.groupby(["job_no", "task_name", "month_key"], dropna=False)
    numeric = grouped.agg(agg_map).reset_index()
    numeric = numeric.rename(
        columns={
            "hours": "total_hours",
            "cost": "total_cost",
            "billable_amount_actual": "billable_amount_actual",
            "billable_flag": "billable_hours",
            "onshore_flag": "onshore_hours",
            "[Staff] Name": "distinct_staff_count",
        }
    )
    numeric["billable_hours"] = grouped.apply(lambda g: g.loc[g["billable_flag"], "hours"].sum()).values
    numeric["onshore_hours"] = grouped.apply(lambda g: g.loc[g["onshore_flag"], "hours"].sum()).values
    numeric["missing_base_rate_flag"] = grouped.apply(lambda g: g["missing_base_rate"].any()).values
    numeric["had_negative_hours_flag"] = grouped.apply(lambda g: g["had_negative_hours"].any()).values

    base_rate_hours = grouped.apply(lambda g: g.loc[g["base_rate"] > 0, "hours"].sum()).values
    billable_rate_hours = grouped.apply(lambda g: g.loc[g["billable_rate"] > 0, "hours"].sum()).values
    numeric["avg_base_rate"] = np.where(
        base_rate_hours > 0, numeric["total_cost"] / base_rate_hours, 0
    )
    numeric["avg_billable_rate"] = np.where(
        billable_rate_hours > 0, numeric["billable_amount_actual"] / billable_rate_hours, 0
    )

    dims = {"task_name_raw": grouped["task_name_raw"].agg(_mode).values}
    dept_info = grouped.apply(lambda g: _weighted_mode_info(g["department_actual"], g["hours"]))
    dept_df = pd.DataFrame(
        dept_info.tolist(),
        columns=[
            "department_actual",
            "mixed_department",
            "dept_top_share",
            "dept_second",
            "dept_second_share",
        ],
    )
    dims.update({col: dept_df[col].values for col in dept_df.columns})

    for out_col, src_col in DIMENSION_COLUMNS.items():
        if src_col not in df.columns:
            continue
        mode_info = grouped.apply(lambda g: _weighted_mode(g[src_col], g["hours"]))
        mode_df = pd.DataFrame(mode_info.tolist(), columns=[out_col, f"mixed_dimension_{out_col}"])
        dims[out_col] = mode_df[out_col].values
        dims[f"mixed_dimension_{out_col}"] = mode_df[f"mixed_dimension_{out_col}"].values

    dim_df = pd.DataFrame(dims)
    timesheet_task_month = pd.concat([numeric.reset_index(drop=True), dim_df], axis=1)

    qa = {
        "negative_hours": negative_hours,
        "missing_base_rate_rows": missing_base_rate,
    }
    logger.info("Timesheet rows with negative hours: %s", negative_hours)
    return timesheet_task_month, qa
