from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.app_state import apply_filters, get_data, sidebar_base_controls, sidebar_filters
from src.utils import load_settings


st.header("Data QA")

base = sidebar_base_controls()

data = get_data(
    data_source=base["data_source"],
    input_path=base["input_path"],
    fy=base["fy"],
    include_all_history=base["include_all_history"],
)

filters = sidebar_filters(data["fact"], base)
filtered_fact = apply_filters(data["fact"], filters)

settings = load_settings("config/settings.yaml")
qa_path = Path(settings["processed_dir"]) / "qa_report.json"

if qa_path.exists():
    qa_report = json.loads(qa_path.read_text(encoding="utf-8"))
    st.subheader("QA Summary")
    st.json(qa_report)
else:
    st.warning("QA report not found. Run the build script to generate it.")

def _flag_col(name: str) -> bool:
    return name in filtered_fact.columns


mask = False
if _flag_col("missing_base_rate_flag"):
    mask = mask | filtered_fact["missing_base_rate_flag"]
if _flag_col("had_negative_hours_flag"):
    mask = mask | filtered_fact["had_negative_hours_flag"]
if _flag_col("is_unallocated_row"):
    mask = mask | filtered_fact["is_unallocated_row"]
if _flag_col("is_unquoted_task"):
    mask = mask | filtered_fact["is_unquoted_task"]

flags = filtered_fact[mask] if mask is not False else filtered_fact.iloc[0:0]

st.subheader("Flagged Rows")
if flags.empty:
    st.write("No flagged rows in the current filter window.")
else:
    st.dataframe(flags, use_container_width=True)
