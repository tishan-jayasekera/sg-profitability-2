from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.app_state import apply_filters, get_data, sidebar_base_controls, sidebar_filters


st.header("Task Reconciliation")

base = sidebar_base_controls()

data = get_data(
    data_source=base["data_source"],
    input_path=base["input_path"],
    fy=base["fy"],
    include_all_history=base["include_all_history"],
)

filters = sidebar_filters(data["fact"], base)
filtered_fact = apply_filters(data["fact"], filters)

job_options = ["All Jobs"] + sorted({j for j in filtered_fact["job_no"].dropna().unique() if j != ""})
job_no = st.selectbox("Job Filter", job_options)

if job_no != "All Jobs":
    filtered_fact = filtered_fact[filtered_fact["job_no"] == job_no]

summary = (
    filtered_fact.groupby(["job_no", "task_name"], dropna=False)
    .agg(
        actual_hours=("total_hours", "sum"),
        quoted_time=("quoted_time", "first"),
        quoted_amount=("quoted_amount", "first"),
        revenue_allocated=("revenue_allocated", "sum"),
    )
    .reset_index()
)

fig = px.bar(
    summary,
    x="task_name",
    y=["quoted_time", "actual_hours"],
    barmode="group",
    title="Quoted vs Actual Hours",
)
st.plotly_chart(fig, use_container_width=True)

fig_amount = px.bar(
    summary,
    x="task_name",
    y=["quoted_amount", "revenue_allocated"],
    barmode="group",
    title="Quoted Amount vs Allocated Revenue",
)
st.plotly_chart(fig_amount, use_container_width=True)

st.subheader("Filtered Tasks")

st.dataframe(summary, use_container_width=True)

csv_data = summary.to_csv(index=False)
st.download_button(
    "Download Summary CSV",
    data=csv_data,
    file_name="task_reconciliation.csv",
    mime="text/csv",
)
