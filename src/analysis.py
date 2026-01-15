from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


METRIC_DEFINITIONS = {
    "Quoted_Amount": {"name": "Quoted Amount", "formula": "quoted_amount", "desc": "Revenue from quote"},
    "Expected_Quote": {"name": "Expected Quote", "formula": "quoted_hours x billable_rate_hr", "desc": "Benchmark quote"},
    "Base_Cost": {"name": "Base Cost", "formula": "actual_hours x cost_rate_hr", "desc": "Internal cost"},
    "Quoted_Margin": {"name": "Quoted Margin", "formula": "quoted_amount - base_cost", "desc": "Quoted margin"},
    "Actual_Margin": {"name": "Actual Margin", "formula": "billable_value - base_cost", "desc": "Actual margin"},
    "Quote_Gap": {"name": "Quote Gap", "formula": "quoted_amount - expected_quote", "desc": "Pricing gap"},
    "Effective_Rate_Hr": {"name": "Effective Rate/Hr", "formula": "quoted_amount / actual_hours", "desc": "Revenue per hour"},
    "Billable_Rate_Hr": {"name": "Billable Rate/Hr", "formula": "avg billable rate", "desc": "Standard billing rate"},
    "Cost_Rate_Hr": {"name": "Cost Rate/Hr", "formula": "avg base rate", "desc": "Internal cost rate"},
}


def _month_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_key"] = pd.to_datetime(df["month_key"], errors="coerce")
    df["Calendar_Month"] = df["month_key"].dt.strftime("%b %Y")
    fy = df["month_key"].dt.year + (df["month_key"].dt.month >= 7).astype(int)
    df["Fiscal_Year"] = fy
    df["FY_Label"] = df["Fiscal_Year"].apply(lambda v: f"FY{str(int(v))[-2:]}" if pd.notna(v) else "Unknown")
    fy_month = df["month_key"].dt.month.apply(lambda m: m - 6 if m >= 7 else m + 6)
    df["FY_Month"] = fy_month
    df["FY_Month_Label"] = df["FY_Month"].apply(
        lambda m: ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"][int(m) - 1]
        if pd.notna(m) and 1 <= int(m) <= 12
        else "Unknown"
    )
    return df


def apply_filters(
    df: pd.DataFrame,
    exclude_sg_allocation: bool = False,
    billable_only: bool = False,
    fiscal_year: Optional[int] = None,
    department: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    recon = {
        "raw_records": len(df),
        "excluded_sg_allocation": 0,
        "excluded_non_billable": 0,
        "excluded_other_fy": 0,
        "excluded_other_dept": 0,
        "final_records": 0,
    }
    df_f = df.copy()

    if exclude_sg_allocation:
        mask = df_f["task_name"].astype(str).str.strip().eq("Social Garden Invoice Allocation")
        recon["excluded_sg_allocation"] = int(mask.sum())
        df_f = df_f[~mask]

    if billable_only:
        mask = (df_f["billable_rate_hr"] > 0) & (df_f["cost_rate_hr"] > 0)
        recon["excluded_non_billable"] = int((~mask).sum())
        df_f = df_f[mask]

    if fiscal_year is not None:
        mask = df_f["Fiscal_Year"] == fiscal_year
        recon["excluded_other_fy"] = int((~mask).sum())
        df_f = df_f[mask]

    if department is not None:
        mask = df_f["department_reporting"] == department
        recon["excluded_other_dept"] = int((~mask).sum())
        df_f = df_f[mask]

    recon["final_records"] = len(df_f)
    return df_f, recon


def compute_reconciliation_totals(df: pd.DataFrame, recon: Dict[str, int]) -> Dict[str, object]:
    recon["totals"] = {
        "sum_quoted_hours": float(df["quoted_hours"].sum()),
        "sum_actual_hours": float(df["actual_hours"].sum()),
        "sum_quoted_amount": float(df["quoted_amount"].sum()),
        "sum_expected_quote": float(df["expected_quote"].sum()),
        "sum_billable_value": float(df["billable_value"].sum()),
        "sum_base_cost": float(df["total_cost"].sum()),
        "avg_quoted_rate_hr": float(df[df["quoted_rate_hr"] > 0]["quoted_rate_hr"].mean())
        if len(df[df["quoted_rate_hr"] > 0]) > 0
        else 0.0,
        "avg_billable_rate_hr": float(df["billable_rate_hr"].mean()),
        "avg_cost_rate_hr": float(df["cost_rate_hr"].mean()),
        "unique_jobs": int(df["job_no"].nunique()),
        "unique_products": int(df["product"].nunique()),
        "unique_departments": int(df["department_reporting"].nunique()),
    }
    return recon


def get_available_fiscal_years(df: pd.DataFrame) -> List[int]:
    return sorted([int(y) for y in df["Fiscal_Year"].dropna().unique() if pd.notna(y)])


def get_available_departments(df: pd.DataFrame) -> List[str]:
    return sorted([d for d in df["department_reporting"].dropna().unique() if d])


def get_available_products(df: pd.DataFrame, department: Optional[str] = None) -> List[str]:
    if department:
        prods = df[df["department_reporting"] == department]["product"].dropna().unique()
    else:
        prods = df["product"].dropna().unique()
    return sorted([p for p in prods if p])


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_Sort"] = df["month_key"].dt.to_period("M")

    g = df.groupby(["Month_Sort", "Calendar_Month", "Fiscal_Year", "FY_Month"]).agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
        job_count=("job_no", pd.Series.nunique),
    ).reset_index()

    g = g.sort_values("Month_Sort").reset_index(drop=True)

    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["quoted_margin_pct"] = np.where(g["quoted_amount"] > 0, g["quoted_margin"] / g["quoted_amount"] * 100, 0)
    g["actual_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)

    g["quoted_rate_hr"] = np.where(g["quoted_hours"] > 0, g["quoted_amount"] / g["quoted_hours"], 0)
    g["effective_rate_hr"] = np.where(g["actual_hours"] > 0, g["quoted_amount"] / g["actual_hours"], 0)
    g["cost_rate_hr"] = np.where(g["actual_hours"] > 0, g["base_cost"] / g["actual_hours"], 0)

    g["hours_variance"] = g["actual_hours"] - g["quoted_hours"]
    g["hours_variance_pct"] = np.where(g["quoted_hours"] > 0, g["hours_variance"] / g["quoted_hours"] * 100, 0)
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)

    return g


def compute_monthly_by_department(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_Sort"] = df["month_key"].dt.to_period("M")

    g = df.groupby(["Month_Sort", "Calendar_Month", "department_reporting"]).agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
        job_count=("job_no", pd.Series.nunique),
    ).reset_index()

    g = g.sort_values(["Month_Sort", "department_reporting"]).reset_index(drop=True)
    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["actual_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)

    return g


def compute_monthly_by_product(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_Sort"] = df["month_key"].dt.to_period("M")

    g = df.groupby(["Month_Sort", "Calendar_Month", "department_reporting", "product"]).agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
        job_count=("job_no", pd.Series.nunique),
    ).reset_index()

    g = g.sort_values(["Month_Sort", "department_reporting", "product"]).reset_index(drop=True)
    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["actual_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)

    return g


def compute_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("department_reporting").agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
        job_count=("job_no", pd.Series.nunique),
        product_count=("product", pd.Series.nunique),
    ).reset_index()

    g["margin"] = g["quoted_amount"] - g["base_cost"]
    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["margin_pct"] = np.where(g["quoted_amount"] > 0, g["quoted_margin"] / g["quoted_amount"] * 100, 0)
    g["billable_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)
    g["quoted_rate_hr"] = np.where(g["quoted_hours"] > 0, g["quoted_amount"] / g["quoted_hours"], 0)
    g["billable_rate_hr"] = np.where(g["actual_hours"] > 0, g["billable_value"] / g["actual_hours"], 0)
    g["cost_rate_hr"] = np.where(g["actual_hours"] > 0, g["base_cost"] / g["actual_hours"], 0)
    g["hours_variance"] = g["actual_hours"] - g["quoted_hours"]
    g["hours_variance_pct"] = np.where(g["quoted_hours"] > 0, g["hours_variance"] / g["quoted_hours"] * 100, 0)
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)

    return g


def compute_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["department_reporting", "product"]).agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
        job_count=("job_no", pd.Series.nunique),
    ).reset_index()

    g["margin"] = g["quoted_amount"] - g["base_cost"]
    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["margin_pct"] = np.where(g["quoted_amount"] > 0, g["quoted_margin"] / g["quoted_amount"] * 100, 0)
    g["billable_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)
    g["quoted_rate_hr"] = np.where(g["quoted_hours"] > 0, g["quoted_amount"] / g["quoted_hours"], 0)
    g["billable_rate_hr"] = np.where(g["actual_hours"] > 0, g["billable_value"] / g["actual_hours"], 0)
    g["cost_rate_hr"] = np.where(g["actual_hours"] > 0, g["base_cost"] / g["actual_hours"], 0)
    g["hours_variance"] = g["actual_hours"] - g["quoted_hours"]
    g["hours_variance_pct"] = np.where(g["quoted_hours"] > 0, g["hours_variance"] / g["quoted_hours"] * 100, 0)
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)

    return g


def compute_job_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(
        [
            "department_reporting",
            "product",
            "job_no",
            "job_name",
            "client",
            "job_status",
            "Calendar_Month",
            "Fiscal_Year",
            "FY_Label",
        ]
    ).agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
    ).reset_index()

    g["margin"] = g["quoted_amount"] - g["base_cost"]
    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["margin_pct"] = np.where(g["quoted_amount"] > 0, g["quoted_margin"] / g["quoted_amount"] * 100, 0)
    g["billable_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)
    g["quoted_rate_hr"] = np.where(g["quoted_hours"] > 0, g["quoted_amount"] / g["quoted_hours"], 0)
    g["billable_rate_hr"] = np.where(g["actual_hours"] > 0, g["billable_value"] / g["actual_hours"], 0)
    g["cost_rate_hr"] = np.where(g["actual_hours"] > 0, g["base_cost"] / g["actual_hours"], 0)
    g["effective_rate_hr"] = np.where(g["actual_hours"] > 0, g["quoted_amount"] / g["actual_hours"], 0)
    g["hours_variance"] = g["actual_hours"] - g["quoted_hours"]
    g["hours_variance_pct"] = np.where(g["quoted_hours"] > 0, g["hours_variance"] / g["quoted_hours"] * 100, np.where(g["actual_hours"] > 0, 100, 0))
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)
    g["is_overrun"] = g["hours_variance"] > 0
    g["is_loss"] = g["margin"] < 0
    g["is_underquoted"] = g["quote_gap"] < 0

    return g


def compute_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(
        [
            "department_reporting",
            "product",
            "job_no",
            "job_name",
            "task_name",
            "Fiscal_Year",
            "FY_Label",
            "Calendar_Month",
        ]
    ).agg(
        quoted_hours=("quoted_hours", "sum"),
        quoted_amount=("quoted_amount", "sum"),
        actual_hours=("actual_hours", "sum"),
        billable_value=("billable_value", "sum"),
        base_cost=("total_cost", "sum"),
        expected_quote=("expected_quote", "sum"),
        billable_rate_hr=("billable_rate_hr", "mean"),
        cost_rate_hr=("cost_rate_hr", "mean"),
        quoted_rate_hr=("quoted_rate_hr", "mean"),
    ).reset_index()

    g["margin"] = g["quoted_amount"] - g["base_cost"]
    g["quoted_margin"] = g["quoted_amount"] - g["base_cost"]
    g["actual_margin"] = g["billable_value"] - g["base_cost"]
    g["margin_variance"] = g["actual_margin"] - g["quoted_margin"]
    g["margin_pct"] = np.where(g["quoted_amount"] > 0, g["quoted_margin"] / g["quoted_amount"] * 100, 0)
    g["billable_margin_pct"] = np.where(g["billable_value"] > 0, g["actual_margin"] / g["billable_value"] * 100, 0)
    g["hours_variance"] = g["actual_hours"] - g["quoted_hours"]
    g["hours_variance_pct"] = np.where(g["quoted_hours"] > 0, g["hours_variance"] / g["quoted_hours"] * 100, np.where(g["actual_hours"] > 0, 100, 0))
    g["quote_gap"] = g["quoted_amount"] - g["expected_quote"]
    g["quote_gap_pct"] = np.where(g["expected_quote"] > 0, g["quote_gap"] / g["expected_quote"] * 100, 0)
    g["is_unquoted"] = (g["quoted_hours"] == 0) & (g["actual_hours"] > 0)
    g["is_overrun"] = g["hours_variance"] > 0

    return g


def generate_insights(
    job_summary: pd.DataFrame,
    dept_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
) -> Dict[str, List[str]]:
    insights = {
        "headline": [],
        "margin_drivers": [],
        "quoting_accuracy": [],
        "department_performance": [],
        "trends": [],
        "action_items": [],
    }

    total_quoted = job_summary["quoted_amount"].sum()
    total_billable = job_summary["billable_value"].sum()
    total_cost = job_summary["base_cost"].sum()
    total_profit = total_quoted - total_cost
    overall_margin = (total_profit / total_quoted * 100) if total_quoted > 0 else 0
    realization = (total_billable / total_quoted * 100) if total_quoted > 0 else 0

    if realization < 90:
        insights["headline"].append(
            f"Revenue realization at {realization:.0f}% — billing below quoted amounts"
        )
    elif realization > 110:
        insights["headline"].append(
            f"Strong revenue realization at {realization:.0f}% — exceeding quotes"
        )

    if overall_margin < 20:
        insights["headline"].append(f"Overall margin critically low at {overall_margin:.1f}%")
    elif overall_margin < 35:
        insights["headline"].append(f"Overall margin below target at {overall_margin:.1f}%")
    else:
        insights["headline"].append(f"Healthy overall margin at {overall_margin:.1f}%")

    loss_jobs = job_summary[job_summary["is_loss"]]
    if len(loss_jobs) > 0:
        total_losses = loss_jobs["margin"].sum()
        insights["headline"].append(
            f"{len(loss_jobs)} jobs running at a loss, totaling ${abs(total_losses):,.0f}"
        )

    if len(dept_summary) > 0:
        worst_dept = dept_summary.loc[dept_summary["billable_margin_pct"].idxmin()]
        best_dept = dept_summary.loc[dept_summary["billable_margin_pct"].idxmax()]
        if worst_dept["billable_margin_pct"] < 15:
            insights["margin_drivers"].append(
                f"{worst_dept['department_reporting']} dragging margins at {worst_dept['billable_margin_pct']:.1f}%"
            )
        if best_dept["billable_margin_pct"] > 40:
            insights["margin_drivers"].append(
                f"{best_dept['department_reporting']} leading with {best_dept['billable_margin_pct']:.1f}% margin"
            )

    underquoted = job_summary[job_summary["quote_gap_pct"] < 0]
    if len(underquoted) > 0:
        insights["quoting_accuracy"].append(
            f"{len(underquoted)} jobs underquoted vs internal benchmark"
        )

    unquoted_tasks = task_summary[task_summary["is_unquoted"]]
    if len(unquoted_tasks) > 0:
        unquoted_cost = unquoted_tasks["base_cost"].sum()
        unquoted_hours = unquoted_tasks["actual_hours"].sum()
        insights["quoting_accuracy"].append(
            f"{len(unquoted_tasks)} unquoted tasks detected — {unquoted_hours:,.0f} hours at ${unquoted_cost:,.0f} cost"
        )

    if len(monthly_summary) >= 3:
        recent = monthly_summary.tail(3)
        margin_trend = recent["actual_margin_pct"].values
        if len(margin_trend) >= 3:
            if margin_trend[-1] > margin_trend[-3] + 5:
                insights["trends"].append(
                    f"Margins improving — up {margin_trend[-1] - margin_trend[-3]:.1f}pp over last 3 months"
                )
            elif margin_trend[-1] < margin_trend[-3] - 5:
                insights["trends"].append(
                    f"Margins declining — down {margin_trend[-3] - margin_trend[-1]:.1f}pp over last 3 months"
                )

    if len(loss_jobs) > 0:
        top_loss = loss_jobs.nsmallest(3, "margin")
        for _, job in top_loss.iterrows():
            insights["action_items"].append(
                f"Review {job['job_name']} ({job['job_no']}) — ${job['margin']:,.0f} margin"
            )

    return insights


def calculate_overall_metrics(js: pd.DataFrame) -> Dict[str, float]:
    n = len(js)
    if n == 0:
        return {"total_jobs": 0}

    q = js["quoted_amount"].sum()
    b = js["billable_value"].sum()
    c = js["base_cost"].sum()
    p = q - c
    hq = js["quoted_hours"].sum()
    ha = js["actual_hours"].sum()
    eq = js["expected_quote"].sum()

    quoted_margin = q - c
    actual_margin = b - c

    metrics = {
        "total_jobs": int(n),
        "total_quoted_amount": float(q),
        "total_expected_quote": float(eq),
        "total_billable_value": float(b),
        "total_base_cost": float(c),
        "total_profit": float(p),
        "margin": float(quoted_margin),
        "margin_pct": float((quoted_margin / q * 100) if q > 0 else 0),
        "overall_actual_margin": float(actual_margin),
        "overall_margin_variance": float(actual_margin - quoted_margin),
        "overall_billable_margin_pct": float((actual_margin / b * 100) if b > 0 else 0),
        "revenue_realization_pct": float((b / q * 100) if q > 0 else 0),
        "avg_quoted_rate_hr": float((q / hq) if hq > 0 else 0),
        "avg_billable_rate_hr": float((b / ha) if ha > 0 else 0),
        "avg_cost_rate_hr": float((c / ha) if ha > 0 else 0),
        "total_hours_quoted": float(hq),
        "total_hours_actual": float(ha),
        "hours_variance": float(ha - hq),
        "hours_variance_pct": float(((ha - hq) / hq * 100) if hq > 0 else 0),
    }
    metrics["quote_gap"] = float(q - eq)
    metrics["quote_gap_pct"] = float(((q - eq) / eq * 100) if eq > 0 else 0)
    metrics["avg_effective_rate_hr"] = float((q / ha) if ha > 0 else 0)
    metrics["jobs_over_budget"] = int(js["is_overrun"].sum())
    metrics["jobs_at_loss"] = int(js["is_loss"].sum())
    metrics["jobs_underquoted"] = int(js["is_underquoted"].sum())
    metrics["overrun_rate"] = float((js["is_overrun"].sum() / n * 100) if n > 0 else 0)
    metrics["loss_rate"] = float((js["is_loss"].sum() / n * 100) if n > 0 else 0)

    return metrics


def analyze_overrun_causes(ts: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    unq = ts[ts["is_unquoted"]]
    ovr = ts[(ts["is_overrun"]) & (~ts["is_unquoted"])]
    return {
        "scope_creep": {
            "count": int(len(unq)),
            "cost": float(unq["base_cost"].sum()),
            "hours": float(unq["actual_hours"].sum()),
        },
        "underestimation": {
            "count": int(len(ovr)),
            "excess_hours": float(ovr["hours_variance"].sum()),
        },
    }


def get_top_overruns(js: pd.DataFrame, n: int = 10, by: str = "hours_variance") -> pd.DataFrame:
    return js.nlargest(n, by)


def get_loss_making_jobs(js: pd.DataFrame) -> pd.DataFrame:
    return js[js["is_loss"]].sort_values("margin")


def get_unquoted_tasks(ts: pd.DataFrame) -> pd.DataFrame:
    return ts[ts["is_unquoted"]].sort_values("base_cost", ascending=False)


def get_underquoted_jobs(js: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return js[js["quote_gap"] < threshold].sort_values("quote_gap")


def get_premium_jobs(js: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    return js[js["quote_gap"] > threshold].sort_values("quote_gap", ascending=False)


def diagnose_job_margin(job_row: pd.Series, job_tasks: pd.DataFrame) -> Dict[str, object]:
    diagnosis = {"summary": "", "issues": [], "root_causes": [], "recommendations": []}

    margin = job_row["actual_margin"]
    margin_pct = job_row["billable_margin_pct"]
    quote_gap = job_row["quote_gap"]
    hours_var_pct = job_row["hours_variance_pct"]

    if margin < 0:
        diagnosis["summary"] = "Job is running at a loss"
        diagnosis["issues"].append("Negative margin")
    elif margin_pct < 20:
        diagnosis["summary"] = "Job has low profitability"
        diagnosis["issues"].append("Low margin percentage")
    else:
        diagnosis["summary"] = "Job is profitable"

    if quote_gap < -1000:
        diagnosis["issues"].append("Significantly underquoted")
        diagnosis["root_causes"].append("Pricing below internal rates")
        diagnosis["recommendations"].append("Review pricing strategy for similar jobs")
    elif quote_gap > 1000:
        diagnosis["issues"].append("Premium pricing applied")

    if hours_var_pct > 50:
        diagnosis["issues"].append("Major scope overrun")
        diagnosis["root_causes"].append("Underestimated effort requirements")
        diagnosis["recommendations"].append("Improve effort estimation")
    elif hours_var_pct < -20:
        diagnosis["issues"].append("Significant underrun")

    if len(job_tasks) > 0:
        unquoted_tasks = job_tasks[job_tasks["is_unquoted"]]
        if len(unquoted_tasks) > 0:
            diagnosis["issues"].append(f"{len(unquoted_tasks)} unquoted tasks")
            diagnosis["root_causes"].append("Scope changes not quoted")
            diagnosis["recommendations"].append("Implement change order process")

    return diagnosis


def compute_builder_task_stats(
    fact: pd.DataFrame,
    timesheet_task_month: pd.DataFrame,
    department: Optional[str] = None,
    product: Optional[str] = None,
) -> pd.DataFrame:
    fact_filter = fact.copy()
    if department:
        fact_filter = fact_filter[fact_filter["department_reporting"] == department]
    if product:
        fact_filter = fact_filter[fact_filter["product"] == product]

    keys = fact_filter[["job_no", "task_name"]].drop_duplicates()
    if keys.empty:
        return pd.DataFrame()

    ts = timesheet_task_month.copy()
    ts = ts.merge(keys, on=["job_no", "task_name"], how="inner")

    total_jobs = ts["job_no"].nunique()
    stats = (
        ts.groupby("task_name")
        .agg(
            Jobs_With_Task=("job_no", "nunique"),
            Avg_Quoted_Hours=("total_hours", "mean"),
            Avg_Actual_Hours=("total_hours", "mean"),
            Billable_Rate_Hr=("avg_billable_rate", "mean"),
            Cost_Rate_Hr=("avg_base_rate", "mean"),
            Total_Actual_Hours=("total_hours", "sum"),
        )
        .reset_index()
    )
    stats["Frequency_Pct"] = np.where(total_jobs > 0, (stats["Jobs_With_Task"] / total_jobs) * 100, 0)
    stats = stats.sort_values(["Frequency_Pct", "Avg_Actual_Hours"], ascending=False)
    return stats


def prepare_fact_for_analysis(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()
    df = _month_fields(df)
    df["department_reporting"] = df.get("department_reporting", df.get("department", ""))
    df["product"] = df.get("product", "")
    df["job_name"] = df.get("job_name", "")
    df["client"] = df.get("client", "")
    df["job_status"] = df.get("job_status", "")
    return df
