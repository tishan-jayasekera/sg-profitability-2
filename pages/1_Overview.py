from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import apply_filters, get_data, sidebar_base_controls, sidebar_filters


st.header("Overview")

base = sidebar_base_controls()

data = get_data(
    data_source=base["data_source"],
    input_path=base["input_path"],
    fy=base["fy"],
    include_all_history=base["include_all_history"],
)

filters = sidebar_filters(data["fact"], base)
filtered_fact = apply_filters(data["fact"], filters)

revenue = filtered_fact["revenue_allocated"].sum()
cost = filtered_fact["total_cost"].sum()
profit = revenue - cost
margin = profit / revenue if revenue else 0

jobs = filtered_fact["job_no"].nunique()
tasks = filtered_fact["task_name"].nunique()
unallocated = filtered_fact.loc[
    filtered_fact["is_unallocated_row"], "revenue_allocated"
].sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Revenue", f"{revenue:,.0f}")
col2.metric("Total Cost", f"{cost:,.0f}")
col3.metric("Gross Profit", f"{profit:,.0f}")
col4.metric("Margin", f"{margin:.1%}")
col5.metric("Jobs", f"{jobs}")
col6.metric("Unallocated Revenue", f"{unallocated:,.0f}")

monthly = (
    filtered_fact.groupby("month_key", dropna=False)
    .agg(revenue=("revenue_allocated", "sum"), cost=("total_cost", "sum"))
    .reset_index()
)
monthly["month_key"] = pd.to_datetime(monthly["month_key"], errors="coerce")
monthly = monthly.dropna(subset=["month_key"]).sort_values("month_key")

fig = px.line(
    monthly,
    x="month_key",
    y=["revenue", "cost"],
    markers=True,
    title="Revenue vs Cost by Month",
)
st.plotly_chart(fig, use_container_width=True)

job_gp = (
    filtered_fact.groupby("job_no", dropna=False)
    .agg(gp=("gross_profit", "sum"))
    .reset_index()
)
job_gp = job_gp.sort_values("gp", ascending=False)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top 20 Jobs by GP")
    st.dataframe(job_gp.head(20), use_container_width=True)

with col_right:
    st.subheader("Bottom 20 Jobs by GP")
    st.dataframe(job_gp.tail(20), use_container_width=True)

unquoted = (
    filtered_fact[filtered_fact["is_unquoted_task"]]
    .groupby("job_no", dropna=False)
    .agg(unquoted_hours=("total_hours", "sum"))
    .reset_index()
    .sort_values("unquoted_hours", ascending=False)
)

overruns = (
    filtered_fact.groupby(["job_no", "task_name"], dropna=False)
    .agg(quote_hour_variance=("quote_hour_variance", "sum"))
    .reset_index()
    .sort_values("quote_hour_variance", ascending=False)
)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Jobs with Largest Unquoted Hours")
    st.dataframe(unquoted.head(15), use_container_width=True)

with col_right:
    st.subheader("Tasks with Biggest Overruns vs Quote")
    st.dataframe(overruns.head(15), use_container_width=True)
