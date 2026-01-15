from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import apply_filters, get_data, sidebar_base_controls, sidebar_filters


st.header("Job Drilldown")

base = sidebar_base_controls()

data = get_data(
    data_source=base["data_source"],
    input_path=base["input_path"],
    fy=base["fy"],
    include_all_history=base["include_all_history"],
)

filters = sidebar_filters(data["fact"], base)
filtered_fact = apply_filters(data["fact"], filters)

job_options = sorted({j for j in filtered_fact["job_no"].dropna().unique() if j != ""})
job_no = st.selectbox("Select Job", job_options if job_options else [""])

job_fact = filtered_fact[filtered_fact["job_no"] == job_no]

monthly = (
    job_fact.groupby("month_key", dropna=False)
    .agg(revenue=("revenue_allocated", "sum"), cost=("total_cost", "sum"))
    .reset_index()
)
monthly["month_key"] = pd.to_datetime(monthly["month_key"], errors="coerce")
monthly = monthly.dropna(subset=["month_key"]).sort_values("month_key")
monthly["gp"] = monthly["revenue"] - monthly["cost"]

fig = px.line(
    monthly,
    x="month_key",
    y=["revenue", "cost", "gp"],
    markers=True,
    title="Monthly Revenue, Cost, and GP",
)
st.plotly_chart(fig, use_container_width=True)

summary = (
    job_fact.groupby(["task_name"], dropna=False)
    .agg(
        total_hours=("total_hours", "sum"),
        total_cost=("total_cost", "sum"),
        revenue_allocated=("revenue_allocated", "sum"),
        gross_profit=("gross_profit", "sum"),
        quoted_time=("quoted_time", "first"),
        quote_hour_variance=("quote_hour_variance", "sum"),
        is_unquoted_task=("is_unquoted_task", "first"),
        is_unworked_task=("is_unworked_task", "first"),
        is_unallocated_row=("is_unallocated_row", "first"),
    )
    .reset_index()
)

st.subheader("Task Table")

st.dataframe(summary, use_container_width=True)

unallocated_months = job_fact[job_fact["is_unallocated_row"]][["month_key", "revenue_allocated"]]
if not unallocated_months.empty:
    st.subheader("Months with Revenue but No Hours")
    st.dataframe(unallocated_months, use_container_width=True)
