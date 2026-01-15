from __future__ import annotations

import streamlit as st

from src.app_state import apply_filters, get_data, sidebar_base_controls, sidebar_filters


st.set_page_config(page_title="SG Profitability", layout="wide")

st.title("SG Profitability")

base = sidebar_base_controls()

data = get_data(
    data_source=base["data_source"],
    input_path=base["input_path"],
    fy=base["fy"],
    include_all_history=base["include_all_history"],
)

filters = sidebar_filters(data["fact"], base)
filtered_fact = apply_filters(data["fact"], filters)

st.markdown(
    "Use the pages in the left navigation for detailed views. The filters apply across pages."
)

col1, col2, col3, col4 = st.columns(4)

revenue = filtered_fact["revenue_allocated"].sum()
cost = filtered_fact["total_cost"].sum()
profit = revenue - cost
margin = profit / revenue if revenue else 0

col1.metric("Total Revenue", f"{revenue:,.0f}")
col2.metric("Total Cost", f"{cost:,.0f}")
col3.metric("Gross Profit", f"{profit:,.0f}")
col4.metric("Margin", f"{margin:.1%}")
