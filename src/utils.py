from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


LOGGER_NAME = "sg_profitability"


def get_logger() -> logging.Logger:
    """Create or return a module-level logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_settings(path: str | Path) -> Dict[str, Any]:
    """Load YAML settings from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names in-place."""
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def normalize_whitespace(value: Any) -> str:
    """Collapse whitespace to single spaces and strip ends."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return " ".join(text.split()).strip()


def standardize_job_no(value: Any) -> str:
    """Standardize job number keys."""
    return normalize_whitespace(value).upper()


def standardize_task_name(value: Any) -> str:
    """Standardize task names for joins."""
    return normalize_whitespace(value)


def truthy_flag(value: Any, truthy_values: Iterable[Any]) -> bool:
    """Evaluate whether a value should be treated as truthy by configuration."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    str_val = str(value).strip().upper()
    for item in truthy_values:
        if isinstance(item, str):
            if str_val == item.strip().upper():
                return True
        else:
            if value == item:
                return True
    return False


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def to_month_start(dt_series: pd.Series) -> pd.Series:
    """Normalize a datetime series to first-of-month."""
    dates = pd.to_datetime(dt_series, errors="coerce")
    return dates.values.astype("datetime64[M]")


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted average with safe guards."""
    mask = weights > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.average(values[mask], weights=weights[mask]))


def read_task_name_map(path: str | Path) -> pd.DataFrame:
    """Load task name mapping configuration."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["job_no", "from_task", "to_task"])
    return pd.read_csv(path, dtype=str).fillna("")


def apply_task_name_map(
    df: pd.DataFrame, map_df: pd.DataFrame, job_col: str, task_col: str
) -> pd.DataFrame:
    """Apply task name mapping to a dataframe."""
    if map_df.empty:
        return df

    df = df.copy()
    map_df = map_df.copy()
    map_df["job_no"] = map_df["job_no"].fillna("").apply(standardize_job_no)
    map_df["from_task"] = map_df["from_task"].apply(standardize_task_name)
    map_df["to_task"] = map_df["to_task"].apply(standardize_task_name)

    global_map = {
        row["from_task"]: row["to_task"]
        for _, row in map_df[map_df["job_no"] == ""].iterrows()
        if row["from_task"]
    }
    job_specific = {}
    for _, row in map_df[map_df["job_no"] != ""].iterrows():
        if row["from_task"]:
            job_specific.setdefault(row["job_no"], {})[row["from_task"]] = row["to_task"]

    def _map(row: pd.Series) -> str:
        job = row[job_col]
        task = row[task_col]
        mapped = job_specific.get(job, {}).get(task)
        if mapped:
            return mapped
        return global_map.get(task, task)

    df[task_col] = df.apply(_map, axis=1)
    return df


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def coalesce(*series_list: pd.Series) -> pd.Series:
    """Return the first non-null value across a set of Series."""
    if not series_list:
        raise ValueError("No series provided to coalesce")
    result = series_list[0]
    for series in series_list[1:]:
        result = result.combine_first(series)
    return result


def ensure_unique(df: pd.DataFrame, keys: List[str]) -> bool:
    """Return True if dataframe has unique keys."""
    return df.duplicated(subset=keys).sum() == 0


def fuzzy_task_suggestions(
    source_tasks: Iterable[str], target_tasks: Iterable[str], limit: int = 10
) -> List[Dict[str, Any]]:
    """Return fuzzy match suggestions between two task lists."""
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return []

    suggestions = []
    target_list = list({t for t in target_tasks if t})
    for task in list({t for t in source_tasks if t})[:limit]:
        match = process.extractOne(task, target_list, scorer=fuzz.WRatio)
        if match:
            suggestions.append({"task": task, "candidate": match[0], "score": match[1]})
    return suggestions
