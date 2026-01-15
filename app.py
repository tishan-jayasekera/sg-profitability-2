from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import (
    METRIC_DEFINITIONS,
    analyze_overrun_causes,
    apply_filters,
    calculate_overall_metrics,
    compute_builder_task_stats,
    compute_department_summary,
    compute_job_summary,
    compute_monthly_by_department,
    compute_monthly_summary,
    compute_product_summary,
    compute_reconciliation_totals,
    compute_task_summary,
    diagnose_job_margin,
    generate_insights,
    get_available_departments,
    get_available_fiscal_years,
    get_available_products,
    get_loss_making_jobs,
    get_underquoted_jobs,
    get_unquoted_tasks,
    prepare_fact_for_analysis,
)
from src.app_state import get_data, sidebar_base_controls


st.set_page_config(page_title="Job Profitability Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
  <style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  :root {
    --bg-1: #f8f4ef;
    --bg-2: #eef4f7;
    --ink-1: #1e1a18;
    --ink-2: #5a534f;
    --accent-1: #e4572e;
    --accent-2: #2e86ab;
    --accent-3: #2ecc71;
    --card: rgba(255, 255, 255, 0.85);
    --border: rgba(30, 26, 24, 0.08);
    --shadow: 0 10px 30px rgba(30, 26, 24, 0.08);
  }
  html, body, [class*="css"] {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    color: var(--ink-1);
  }
  .stApp {
    background: radial-gradient(1100px 600px at 10% -10%, #fff2e8 0%, transparent 60%),
          radial-gradient(800px 400px at 90% 10%, #e9f3ff 0%, transparent 60%),
          linear-gradient(180deg, var(--bg-1), var(--bg-2));
  }
  .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3.5rem;
  }
  h1, h2, h3, h4 {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
    letter-spacing: -0.02em;
  }
  .hero {
    padding: 1.4rem 1.8rem;
    border-radius: 18px;
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }
  .hero-sub {
    color: var(--ink-2);
    font-size: 1rem;
    margin-bottom: 0.75rem;
  }
  .pill {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(46, 134, 171, 0.12);
    color: #1f5e78;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.4rem;
  }
  .callout {
    padding: 0.85rem 1rem;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    margin: 0.6rem 0 1rem 0;
  }
  .callout-title {
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: var(--accent-2);
  }
  .callout-text {
    color: var(--ink-2);
    font-size: 0.95rem;
  }
  div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 0.9rem 0.9rem 0.7rem 0.9rem;
    border-radius: 14px;
    box-shadow: var(--shadow);
  }
  div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: var(--ink-2);
  }
  div[data-testid="stMetricValue"] {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
  }
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.01em;
  }
  .stExpander, .stDataFrame {
    background: var(--card);
    border-radius: 12px;
    border: 1px solid var(--border);
  }
  </style>
  """,
    unsafe_allow_html=True,
)


def fmt_currency(val: float) -> str:
    if pd.isna(val) or val == 0:
        return "$0"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:,.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:,.1f}K"
    return f"${val:,.0f}"


def fmt_pct(val: float) -> str:
    return f"{val:.1f}%" if pd.notna(val) else "N/A"


def fmt_rate(val: float) -> str:
    return f"${val:,.0f}/hr" if pd.notna(val) and val > 0 else "N/A"


def hero(title: str, subtitle: str, pills: list[str]) -> None:
    pill_html = "".join([f"<span class='pill'>{p}</span>" for p in pills])
    st.markdown(
        f"""
    <div class="hero">
      <div class="hero-title">{title}</div>
      <div class="hero-sub">{subtitle}</div>
      <div>{pill_html}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def callout(title: str, body: str) -> None:
    st.markdown(
        f"""
    <div class="callout">
      <div class="callout-title">{title}</div>
      <div class="callout-text">{body}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def callout_list(title: str, items: list[str]) -> None:
    items_html = "".join([f"<li>{i}</li>" for i in items])
    st.markdown(
        f"""
    <div class="callout">
      <div class="callout-title">{title}</div>
      <div class="callout-text">
        <ul style="margin: 0.2rem 0 0.2rem 1.2rem;">{items_html}</ul>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def metric_explainer(title: str, keys: list[str]) -> None:
    lines = []
    for key in keys:
        defn = METRIC_DEFINITIONS.get(key)
        if not defn:
            continue
        lines.append(f"**{defn['name']}** - `{defn['formula']}` - {defn['desc']}")
    if lines:
        with st.expander(title):
            st.markdown("\n\n".join(lines))


def apply_chart_theme() -> None:
    def theme():
        return {
            "config": {
                "background": "rgba(0,0,0,0)",
                "axis": {
                    "labelColor": "#4f4946",
                    "titleColor": "#1e1a18",
                    "gridColor": "#ece7e1",
                    "domainColor": "#d7cfc6",
                    "labelFont": "IBM Plex Sans",
                    "titleFont": "Space Grotesk",
                    "labelFontSize": 12,
                    "titleFontSize": 13,
                },
                "legend": {
                    "labelFont": "IBM Plex Sans",
                    "titleFont": "Space Grotesk",
                    "labelColor": "#4f4946",
                    "titleColor": "#1e1a18",
                },
                "title": {
                    "font": "Space Grotesk",
                    "fontSize": 16,
                    "color": "#1e1a18",
                },
                "view": {"stroke": "transparent"},
            }
        }

    alt.themes.register("profit_theme", theme)
    alt.themes.enable("profit_theme")


@st.cache_data(show_spinner=False)
def compute_summaries(df_filtered: pd.DataFrame):
    dept_summary = compute_department_summary(df_filtered)
    product_summary = compute_product_summary(df_filtered)
    job_summary = compute_job_summary(df_filtered)
    task_summary = compute_task_summary(df_filtered)
    monthly_summary = compute_monthly_summary(df_filtered)
    monthly_by_dept = compute_monthly_by_department(df_filtered)
    metrics = calculate_overall_metrics(job_summary)
    causes = analyze_overrun_causes(task_summary)
    insights = generate_insights(job_summary, dept_summary, monthly_summary, task_summary)
    return (
        dept_summary,
        product_summary,
        job_summary,
        task_summary,
        monthly_summary,
        monthly_by_dept,
        metrics,
        causes,
        insights,
    )


def main() -> None:
    apply_chart_theme()
    hero(
        "Job Profitability Analysis",
        "Revenue = Quoted Amount | Benchmark = Expected Quote (Quoted Hours x Billable Rate)",
        ["Pricing Discipline", "Margin Health", "Scope Control"],
    )

    callout(
        "How to read the dashboard",
        "Start with Executive Summary for topline health, then use Monthly Trends for seasonality, "
        "Drill-Down for root causes, and Job Diagnosis for single-job explanations.",
    )

    st.sidebar.header("Data & Filters")
    base = sidebar_base_controls()

    data = get_data(
        data_source=base["data_source"],
        input_path=base["input_path"],
        fy=base["fy"],
        include_all_history=base["include_all_history"],
    )

    fact = prepare_fact_for_analysis(data["fact"])

    fy_list = get_available_fiscal_years(fact)
    if not fy_list:
        st.error("No fiscal year data found")
        st.stop()

    selected_fy = st.sidebar.selectbox(
        "Fiscal Year",
        fy_list,
        index=len(fy_list) - 1,
        format_func=lambda x: f"FY{str(x)[-2:]}",
    )

    dept_list = get_available_departments(fact)
    selected_dept = st.sidebar.selectbox("Department", ["All Departments"] + dept_list)
    dept_filter = None if selected_dept == "All Departments" else selected_dept

    st.sidebar.markdown("---")
    exclude_sg = st.sidebar.checkbox("Exclude SG Allocation", value=False)
    billable_only = st.sidebar.checkbox("Billable tasks only", value=False)

    df_filtered, recon = apply_filters(
        fact,
        exclude_sg_allocation=exclude_sg,
        billable_only=billable_only,
        fiscal_year=selected_fy,
        department=dept_filter,
    )

    if len(df_filtered) == 0:
        st.error("No data after applying filters.")
        st.stop()

    recon = compute_reconciliation_totals(df_filtered, recon)

    (
        dept_summary,
        product_summary,
        job_summary,
        task_summary,
        monthly_summary,
        monthly_by_dept,
        metrics,
        causes,
        insights,
    ) = compute_summaries(df_filtered)

    st.sidebar.markdown("---")
    st.sidebar.metric("Records", f"{recon['final_records']:,}")
    st.sidebar.metric("Jobs", f"{metrics['total_jobs']:,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Executive Summary", "Monthly Trends", "Drill-Down", "Job Diagnosis", "Smart Quote Builder"]
    )

    with tab1:
        st.header(f"FY{str(selected_fy)[-2:]} Executive Summary")
        callout_list(
            "Executive Summary explainer",
            [
                "All KPIs aggregate filtered jobs and tasks",
                "Margin % uses Quoted Amount as the denominator",
                "Quote Gap % uses Expected Quote as the denominator",
            ],
        )

        if insights["headline"]:
            for h in insights["headline"]:
                st.markdown(h)

        st.markdown("---")

        st.subheader("Revenue & Margin")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue (Quoted)", fmt_currency(metrics["total_quoted_amount"]))
        c2.metric("Base Cost", fmt_currency(metrics["total_base_cost"]))
        c3.metric("Margin", fmt_currency(metrics["margin"]), delta=fmt_pct(metrics["margin_pct"]))
        c4.metric("Margin %", fmt_pct(metrics["margin_pct"]))

        st.subheader("Quoting Sanity Check")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Amount", fmt_currency(metrics["total_quoted_amount"]))
        c2.metric("Expected Quote", fmt_currency(metrics["total_expected_quote"]))
        gap = metrics["quote_gap"]
        gap_label = "Above" if gap >= 0 else "Below"
        c3.metric(f"Quote Gap ({gap_label})", fmt_currency(gap), delta=f"{metrics['quote_gap_pct']:+.0f}%")
        c4.metric("Underquoted Jobs", f"{metrics['jobs_underquoted']} / {metrics['total_jobs']}")

        st.subheader("Rate Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted Rate/Hr", fmt_rate(metrics["avg_quoted_rate_hr"]))
        c2.metric("Billable Rate/Hr", fmt_rate(metrics["avg_billable_rate_hr"]))
        c3.metric("Effective Rate/Hr", fmt_rate(metrics["avg_effective_rate_hr"]))
        c4.metric("Cost Rate/Hr", fmt_rate(metrics["avg_cost_rate_hr"]))

        st.subheader("Performance Flags")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jobs at Loss", f"{metrics['jobs_at_loss']} / {metrics['total_jobs']}")
        c2.metric("Hour Overruns", f"{metrics['jobs_over_budget']}")
        c3.metric("Underquoted Jobs", str(metrics["jobs_underquoted"]))
        c4.metric("Scope Creep Tasks", str(causes["scope_creep"]["count"]))

        st.subheader("Margin Bridge")
        bridge_data = pd.DataFrame(
            [
                {"Step": "1. Revenue (Quoted)", "Amount": metrics["total_quoted_amount"], "Color": "Revenue"},
                {"Step": "2. Base Cost", "Amount": -metrics["total_base_cost"], "Color": "Cost"},
                {"Step": "3. Margin", "Amount": metrics["margin"], "Color": "Margin"},
            ]
        )
        bridge_chart = (
            alt.Chart(bridge_data)
            .mark_bar(size=45, cornerRadiusEnd=4)
            .encode(
                x=alt.X("Step:N", sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="~s")),
                color=alt.Color(
                    "Color:N",
                    scale=alt.Scale(domain=["Revenue", "Cost", "Margin"], range=["#2e86ab", "#e4572e", "#2ecc71"]),
                ),
                tooltip=["Step", alt.Tooltip("Amount:Q", format="$,.0f")],
            )
            .properties(height=300)
        )
        st.altair_chart(bridge_chart, use_container_width=True)

    with tab2:
        st.header(f"Monthly Trends - FY{str(selected_fy)[-2:]}")
        if len(monthly_summary) == 0:
            st.warning("No monthly data available.")
        else:
            callout_list(
                "Monthly trend explainer",
                [
                    "Values are aggregated by month after filters",
                    "Quote Gap % uses Expected Quote as the denominator",
                    "Hours Variance % compares actual to quoted hours",
                ],
            )

            trend_metric = st.selectbox(
                "Select Metric",
                ["actual_margin_pct", "quote_gap_pct", "quoted_amount", "hours_variance_pct", "effective_rate_hr"],
                format_func=lambda x: {
                    "actual_margin_pct": "Margin %",
                    "quote_gap_pct": "Quote Gap % (Quoting Accuracy)",
                    "quoted_amount": "Revenue (Quoted Amount)",
                    "hours_variance_pct": "Hours Variance %",
                    "effective_rate_hr": "Effective Rate/Hr",
                }.get(x, x),
            )

            format_map = {
                "actual_margin_pct": ".1f",
                "quote_gap_pct": ".1f",
                "hours_variance_pct": ".1f",
                "quoted_amount": "$,.0f",
                "effective_rate_hr": "$,.0f",
            }
            metric_format = format_map.get(trend_metric, ",.1f")
            trend_chart = (
                alt.Chart(monthly_summary)
                .mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=3)
                .encode(
                    x=alt.X("Calendar_Month:N", sort=list(monthly_summary["Calendar_Month"]), axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y(f"{trend_metric}:Q", axis=alt.Axis(format=metric_format)),
                    color=alt.value("#2e86ab"),
                    tooltip=["Calendar_Month", alt.Tooltip(f"{trend_metric}:Q", format=metric_format)],
                )
                .properties(height=350)
            )
            st.altair_chart(trend_chart, use_container_width=True)

            if selected_dept == "All Departments" and len(monthly_by_dept) > 0:
                st.subheader("Margin % by Department")
                dept_trend = (
                    alt.Chart(monthly_by_dept)
                    .mark_line(point=alt.OverlayMarkDef(size=40))
                    .encode(
                        x=alt.X(
                            "Calendar_Month:N",
                            sort=list(monthly_summary["Calendar_Month"]),
                            axis=alt.Axis(labelAngle=-45),
                        ),
                        y=alt.Y("actual_margin_pct:Q", title="Margin %"),
                        color="department_reporting:N",
                        tooltip=[
                            "Calendar_Month",
                            "department_reporting",
                            alt.Tooltip("actual_margin_pct:Q", format=".0f"),
                        ],
                    )
                    .properties(height=350)
                )
                st.altair_chart(dept_trend, use_container_width=True)

    with tab3:
        st.header("Hierarchical Analysis")
        callout_list(
            "Drill-down explainer",
            [
                "Each level inherits the filters above",
                "Use Margin % to spot weak performers",
                "Use Quote Gap to spot pricing issues",
            ],
        )

        if len(dept_summary) > 0:
            st.subheader("Department Scoreboard")
            dept_metrics = dept_summary.copy()
            dept_metrics["Margin_Band"] = np.where(
                dept_metrics["margin_pct"] >= 35,
                "Healthy",
                np.where(dept_metrics["margin_pct"] < 20, "At Risk", "Watch"),
            )
            dept_scatter = (
                alt.Chart(dept_metrics)
                .mark_circle(size=240)
                .encode(
                    x=alt.X("quote_gap_pct:Q", title="Quote Gap %"),
                    y=alt.Y("margin_pct:Q", title="Margin %"),
                    color=alt.Color(
                        "Margin_Band:N",
                        scale=alt.Scale(domain=["Healthy", "Watch", "At Risk"], range=["#2ecc71", "#f4d35e", "#e4572e"]),
                    ),
                    size=alt.Size("job_count:Q", title="# Jobs", scale=alt.Scale(range=[200, 1200])),
                    tooltip=[
                        "department_reporting",
                        alt.Tooltip("job_count:Q", format=",.0f", title="# Jobs"),
                        alt.Tooltip("margin_pct:Q", format=".1f", title="Margin %"),
                        alt.Tooltip("quote_gap_pct:Q", format=".1f", title="Quote Gap %"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(dept_scatter, use_container_width=True)

        st.markdown("---")
        st.subheader("Level 2: Product Performance")
        sel_dept_drill = st.selectbox("Filter by Department", ["All"] + sorted(dept_summary["department_reporting"].unique().tolist()))
        prod_f = product_summary if sel_dept_drill == "All" else product_summary[product_summary["department_reporting"] == sel_dept_drill]

        if len(prod_f) > 0:
            prod_chart = (
                alt.Chart(prod_f)
                .mark_bar(size=16, cornerRadiusEnd=3)
                .encode(
                    y=alt.Y("product:N", sort="-x"),
                    x=alt.X("margin_pct:Q", title="Margin %", axis=alt.Axis(format="~s")),
                    color=alt.condition(alt.datum.margin_pct < 20, alt.value("#e74c3c"), alt.value("#2ecc71")),
                    tooltip=["product", "department_reporting", alt.Tooltip("margin_pct:Q", format=".1f")],
                )
                .properties(height=320)
            )
            st.altair_chart(prod_chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Level 3: Job Performance")
        sel_prod = st.selectbox("Filter by Product", ["All"] + sorted(prod_f["product"].unique().tolist()))
        jobs_f = job_summary.copy()
        if sel_dept_drill != "All":
            jobs_f = jobs_f[jobs_f["department_reporting"] == sel_dept_drill]
        if sel_prod != "All":
            jobs_f = jobs_f[jobs_f["product"] == sel_prod]

        c1, c2, c3, c4 = st.columns(4)
        show_loss = c1.checkbox("Loss only")
        show_underquoted = c2.checkbox("Underquoted")
        show_overrun = c3.checkbox("Hour Overrun")
        sort_by = c4.selectbox("Sort", ["margin", "quote_gap", "hours_variance_pct", "margin_pct"])

        if show_loss:
            jobs_f = jobs_f[jobs_f["is_loss"]]
        if show_underquoted:
            jobs_f = jobs_f[jobs_f["is_underquoted"]]
        if show_overrun:
            jobs_f = jobs_f[jobs_f["is_overrun"]]

        jobs_disp = jobs_f.sort_values(sort_by, ascending=sort_by in ["margin", "quote_gap"]).head(25)

        if len(job_summary) > 0:
            job_chart = (
                alt.Chart(job_summary)
                .mark_circle(size=110)
                .encode(
                    x=alt.X("quote_gap_pct:Q", title="Quote Gap %"),
                    y=alt.Y("margin_pct:Q", title="Margin %"),
                    color=alt.condition(alt.datum.margin_pct < 20, alt.value("#e4572e"), alt.value("#2ecc71")),
                    tooltip=[
                        "job_name",
                        "department_reporting",
                        "product",
                        alt.Tooltip("margin_pct:Q", format=".1f", title="Margin %"),
                        alt.Tooltip("quote_gap_pct:Q", format=".1f", title="Quote Gap %"),
                        alt.Tooltip("hours_variance_pct:Q", format=".0f", title="Hours Var %"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(job_chart, use_container_width=True)

        if len(jobs_disp) > 0:
            cols = [
                "job_no",
                "job_name",
                "client",
                "Calendar_Month",
                "quoted_amount",
                "expected_quote",
                "quote_gap",
                "base_cost",
                "margin",
                "margin_pct",
                "hours_variance_pct",
            ]
            st.dataframe(
                jobs_disp[cols].style.format(
                    {
                        "quoted_amount": "${:,.0f}",
                        "expected_quote": "${:,.0f}",
                        "quote_gap": "${:,.0f}",
                        "base_cost": "${:,.0f}",
                        "margin": "${:,.0f}",
                        "margin_pct": "{:.1f}%",
                        "hours_variance_pct": "{:+.0f}%",
                    }
                ),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No jobs match filters.")

    with tab4:
        st.header("Job Diagnosis Tool")
        st.markdown("*Understand why a specific job performed the way it did*")
        all_jobs = job_summary.apply(
            lambda r: f"{r['job_no']} - {str(r['job_name'])[:40]} ({r['client']})", axis=1
        ).tolist()

        selected_job = st.selectbox("Select a Job to Diagnose", ["-- Select --"] + all_jobs)

        if selected_job != "-- Select --":
            job_no = selected_job.split(" - ")[0]
            job_row = job_summary[job_summary["job_no"] == job_no].iloc[0]
            job_tasks = task_summary[task_summary["job_no"] == job_no]

            diagnosis = diagnose_job_margin(job_row, job_tasks)

            st.subheader(f"{job_row['job_name']}")
            st.caption(f"Client: {job_row['client']} | {job_row['Calendar_Month']}")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Revenue", fmt_currency(job_row["quoted_amount"]))
            c2.metric("Cost", fmt_currency(job_row["base_cost"]))
            c3.metric("Margin", fmt_currency(job_row["margin"]))
            c4.metric("Quote Gap", fmt_currency(job_row["quote_gap"]))
            c5.metric("Hours Var", f"{job_row['hours_variance_pct']:+.0f}%")

            st.markdown("---")
            st.subheader("Diagnosis")
            st.markdown(f"**Summary:** {diagnosis['summary']}")

            if diagnosis["issues"]:
                st.markdown("**Issues Identified:**")
                for issue in diagnosis["issues"]:
                    st.markdown(f"- {issue}")

            if diagnosis["root_causes"]:
                st.markdown("**Root Causes:**")
                for cause in diagnosis["root_causes"]:
                    st.markdown(f"- {cause}")

            if diagnosis["recommendations"]:
                st.markdown("**Recommendations:**")
                for rec in diagnosis["recommendations"]:
                    st.markdown(f"- {rec}")

            if len(job_tasks) > 0:
                st.markdown("---")
                st.subheader("Task Analysis")
                unquoted_tasks = job_tasks[job_tasks["is_unquoted"]]
                overrun_tasks = job_tasks[job_tasks["is_overrun"] & ~job_tasks["is_unquoted"]]
                underquoted_tasks = job_tasks[job_tasks["quote_gap"] < 0]

                if len(unquoted_tasks) > 0:
                    st.markdown("**Unquoted Tasks (Scope Creep):**")
                    for _, t in unquoted_tasks.iterrows():
                        st.markdown(f"- {t['task_name']}: {t['actual_hours']:.0f} hrs, ${t['base_cost']:,.0f} cost")

                if len(overrun_tasks) > 0:
                    st.markdown("**Hour Overruns:**")
                    for _, t in overrun_tasks.iterrows():
                        st.markdown(f"- {t['task_name']}: {t['hours_variance']:+.0f} hrs over")

                if len(underquoted_tasks) > 0:
                    st.markdown("**Underquoted Tasks:**")
                    for _, t in underquoted_tasks.iterrows():
                        st.markdown(f"- {t['task_name']}: ${abs(t['quote_gap']):,.0f} below internal rates")

    with tab5:
        st.header("Smart Quote Builder")
        callout_list(
            "How this works",
            [
                "Select a department and product to anchor historical tasks",
                "Pick tasks, adjust proposed hours, and add custom lines",
                "Quote uses standard billable rates for pricing",
            ],
        )

        c1, c2, c3 = st.columns(3)
        builder_dept = c1.selectbox("Department", ["All Departments"] + dept_list, key="b_dept")
        builder_dept_filter = None if builder_dept == "All Departments" else builder_dept
        builder_fy = c2.selectbox(
            "Reference Fiscal Year",
            fy_list,
            index=len(fy_list) - 1,
            key="b_fy",
            format_func=lambda x: f"FY{str(x)[-2:]}",
        )

        base_filtered, _ = apply_filters(
            fact,
            exclude_sg_allocation=exclude_sg,
            billable_only=billable_only,
            fiscal_year=builder_fy,
            department=builder_dept_filter,
        )

        products = get_available_products(base_filtered, builder_dept_filter)
        builder_product = c3.selectbox("Product", products if products else ["None"], key="b_prod")

        if builder_product == "None" or len(base_filtered) == 0:
            st.info("No data available for the selected context.")
        else:
            task_stats = compute_builder_task_stats(
                fact=base_filtered,
                timesheet_task_month=data["timesheet_task_month"],
                department=builder_dept_filter,
                product=builder_product,
            )

            if len(task_stats) == 0:
                st.warning("No tasks found for this product and fiscal year.")
            else:
                st.subheader("Historical Task Library")
                st.dataframe(
                    task_stats[[
                        "task_name",
                        "Frequency_Pct",
                        "Avg_Quoted_Hours",
                        "Avg_Actual_Hours",
                        "Billable_Rate_Hr",
                        "Cost_Rate_Hr",
                    ]].style.format(
                        {
                            "Frequency_Pct": "{:.0f}%",
                            "Avg_Quoted_Hours": "{:,.1f}",
                            "Avg_Actual_Hours": "{:,.1f}",
                            "Billable_Rate_Hr": "${:,.0f}",
                            "Cost_Rate_Hr": "${:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    height=320,
                )

                default_tasks = task_stats[task_stats["Frequency_Pct"] >= 75]["task_name"].tolist()
                selected_tasks = st.multiselect(
                    "Select tasks to include",
                    task_stats["task_name"].tolist(),
                    default=default_tasks,
                )

                if selected_tasks:
                    builder_df = task_stats[task_stats["task_name"].isin(selected_tasks)].copy()
                    builder_df["Proposed_Hours"] = builder_df["Avg_Actual_Hours"].round(1)
                    builder_df = builder_df[[
                        "task_name",
                        "Frequency_Pct",
                        "Avg_Quoted_Hours",
                        "Avg_Actual_Hours",
                        "Proposed_Hours",
                        "Billable_Rate_Hr",
                        "Cost_Rate_Hr",
                        "Total_Actual_Hours",
                    ]]

                    st.subheader("Quote Builder")
                    edited = st.data_editor(
                        builder_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "task_name": st.column_config.TextColumn("Task"),
                            "Frequency_Pct": st.column_config.NumberColumn("Frequency %", format="%.0f"),
                            "Avg_Quoted_Hours": st.column_config.NumberColumn("Avg Quoted Hrs", format="%.1f"),
                            "Avg_Actual_Hours": st.column_config.NumberColumn("Avg Actual Hrs", format="%.1f"),
                            "Proposed_Hours": st.column_config.NumberColumn("Proposed Hrs", min_value=0.0, step=0.5),
                            "Billable_Rate_Hr": st.column_config.NumberColumn("Billable Rate/Hr", format="$%.0f"),
                            "Cost_Rate_Hr": st.column_config.NumberColumn("Cost Rate/Hr", format="$%.0f"),
                        },
                        disabled=[
                            "task_name",
                            "Frequency_Pct",
                            "Avg_Quoted_Hours",
                            "Avg_Actual_Hours",
                            "Billable_Rate_Hr",
                            "Cost_Rate_Hr",
                            "Total_Actual_Hours",
                        ],
                    )

                    if "custom_items" not in st.session_state:
                        st.session_state["custom_items"] = pd.DataFrame(
                            columns=["Task_Name", "Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr"]
                        )

                    custom_items = st.data_editor(
                        st.session_state["custom_items"],
                        use_container_width=True,
                        num_rows="dynamic",
                        hide_index=True,
                        column_config={
                            "Task_Name": st.column_config.TextColumn("Task"),
                            "Proposed_Hours": st.column_config.NumberColumn("Proposed Hrs", min_value=0.0, step=0.5),
                            "Billable_Rate_Hr": st.column_config.NumberColumn("Billable Rate/Hr", min_value=0.0, format="$%.0f"),
                            "Cost_Rate_Hr": st.column_config.NumberColumn("Cost Rate/Hr", min_value=0.0, format="$%.0f"),
                        },
                    )
                    st.session_state["custom_items"] = custom_items

                    custom_clean = custom_items.copy()
                    for col in ["Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr"]:
                        custom_clean[col] = pd.to_numeric(custom_clean[col], errors="coerce")
                    custom_clean["Task_Name"] = custom_clean["Task_Name"].fillna("").astype(str).str.strip()
                    valid_custom = custom_clean.dropna(
                        subset=["Proposed_Hours", "Billable_Rate_Hr", "Cost_Rate_Hr"]
                    )
                    valid_custom = valid_custom[
                        (valid_custom["Proposed_Hours"] > 0) & (valid_custom["Task_Name"] != "")
                    ]

                    edited["Revenue"] = edited["Proposed_Hours"] * edited["Billable_Rate_Hr"]
                    edited["Cost"] = edited["Proposed_Hours"] * edited["Cost_Rate_Hr"]

                    custom_revenue = (valid_custom["Proposed_Hours"] * valid_custom["Billable_Rate_Hr"]).sum()
                    custom_cost = (valid_custom["Proposed_Hours"] * valid_custom["Cost_Rate_Hr"]).sum()

                    total_revenue = edited["Revenue"].sum() + custom_revenue
                    total_cost = edited["Cost"].sum() + custom_cost
                    margin = total_revenue - total_cost
                    margin_pct = (margin / total_revenue * 100) if total_revenue > 0 else 0

                    total_proposed_hours = edited["Proposed_Hours"].sum() + valid_custom["Proposed_Hours"].sum()
                    total_hist_actual = edited["Total_Actual_Hours"].sum()

                    st.subheader("Final Recommendation")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Recommended Quote", fmt_currency(total_revenue))
                    col_b.metric("Projected Margin %", fmt_pct(margin_pct))
                    col_c.metric("Total Proposed Hours", f"{total_proposed_hours:,.1f}")

                    if total_hist_actual > 0 and total_proposed_hours < total_hist_actual * 0.8:
                        st.warning(
                            "Proposed hours are >20% below historical actual hours for these tasks. "
                            "Consider increasing scope or validating assumptions."
                        )

    st.markdown("---")
    st.caption(
        f"Job Profitability Analysis | FY{str(selected_fy)[-2:]} | {selected_dept} | "
        f"{recon['final_records']:,} records | Revenue = Quoted Amount"
    )


if __name__ == "__main__":
    main()
