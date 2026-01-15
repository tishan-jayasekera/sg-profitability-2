# SG Profitability (FY26)

Lean MVP for building a unified job profitability dataset from revenue, timesheet, and quotation Excel sheets, with a Streamlit analytics app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build dataset

```bash
python scripts/build_dataset.py --input data/raw/Quoted_Task_Report_FY26.xlsx --fy FY26
```

Outputs:
- `data/processed/revenue_monthly.parquet`
- `data/processed/timesheet_task_month.parquet`
- `data/processed/quote_task.parquet`
- `data/processed/fact_job_task_month.parquet`
- `data/processed/job_month_summary.parquet`
- `data/processed/job_total_summary.parquet`
- `data/processed/qa_report.json`

## Run Streamlit

```bash
streamlit run app.py
```

## Notes
- Place the Excel in `data/raw/Quoted_Task_Report_FY26.xlsx`.
- Customize task mappings in `config/task_name_map.csv`.
- Business rules and assumptions live in `docs/context.md`.
