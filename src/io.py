from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils import get_logger, normalize_columns


SHEET_REVENUE = "Monthly Revenue"
SHEET_TIMESHEET = "Timesheet Data"
SHEET_QUOTATION = "Quotation Data"


def read_excel_sheets(path: str | Path) -> Dict[str, pd.DataFrame]:
    """Read required sheets from the Excel file."""
    logger = get_logger()
    excel_path = Path(path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    logger.info("Loading Excel: %s", excel_path)
    sheets = pd.read_excel(
        excel_path,
        sheet_name=[SHEET_REVENUE, SHEET_TIMESHEET, SHEET_QUOTATION],
        engine="openpyxl",
    )
    return {name: normalize_columns(df) for name, df in sheets.items()}


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe to parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataframe to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
