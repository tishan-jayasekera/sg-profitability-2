# Context and Runbook

## Overview
The source Excel has three relevant sheets with mismatched grains:
- Monthly Revenue: job-month revenue (first-of-month date).
- Timesheet Data: daily entries at job-task level (FY26 only), aggregated to job-task-month.
- Quotation Data: job-task statement of work with quoted hours/amounts and job metadata.

The pipeline cleans keys, aggregates each sheet, allocates job-month revenue to job-task-month by hours share, and reconciles against quotes. Outputs are stored in `data/processed/` as Parquet and a QA JSON report.

## Join Keys and Cleaning
- `job_no`: string, stripped, uppercased.
- `task_name`: stripped, whitespace collapsed.
- `month_key`:
  - revenue: `Month` (already first-of-month).
  - timesheet: `Month Key` (already first-of-month).
- Task mapping: optional overrides in `config/task_name_map.csv` with `job_no` (optional), `from_task`, `to_task`. Mapping is applied before joins. Fuzzy matching suggestions are generated for QA but not auto-applied.

## Revenue Aggregation
- Filter out rows where `Excluded` is truthy (values in `config/settings.yaml`).
- Group by `(job_no, month_key)` and sum `Amount` as `revenue_monthly`.
- Negative revenue is allowed and allocates proportionally.

## Timesheet Aggregation
- Group by `(job_no, task_name, month_key)`.
- Derived columns:
  - `hours`: numeric time; invalid or negative values are coerced to 0 and flagged.
  - `cost`: hours * base rate; missing base rate defaults to 0 and flagged.
- Aggregates:
  - total_hours, billable_hours, onshore_hours
  - total_cost
  - weighted avg base/billable rates
  - distinct staff count
- Dimension columns (Department, Function, Category, Role, Task, Deliverable) use mode; if multiple distinct values are present, a `mixed_dimension_*` flag is raised.

## Revenue Allocation
For each job-month:
- Compute total hours across tasks.
- If total hours > 0, allocate `revenue_monthly` to tasks by hours share.
- If total hours = 0, create a synthetic task row `__UNALLOCATED__` with all revenue.

## Quotation Aggregation
Aggregate quotation rows by `(job_no, task_name)`:
- `quoted_time`, `quoted_amount`, `invoiced_time`, `invoiced_amount` summed.
- Keep job metadata: Client, Job Name, Category, Status, Start/Completed dates, Department, Product.

## Final Fact Table
Grain: `(job_no, task_name, month_key)`
- `revenue_allocated` and `total_cost` to compute `gross_profit` and `margin`.
- Quote fields joined at job-task level:
- `is_unquoted_task` when no quote match.
- `is_unworked_task` when quote exists but no timesheet.
- `is_unallocated_row` for the synthetic allocation row.
- For quoted tasks with no timesheet rows, a synthetic row is added with `month_key` as null.

## QA Checks
- Per job-month: sum of `revenue_allocated` equals `revenue_monthly` within tolerance.
- Fact table keys are unique.
- Flag missing rates, negative hours, mixed dimensions, and task-name mismatches.

## Known Limitations
- Timesheet coverage is FY26 only (Jul-2025 to Jan-2026 in the provided file).
- Revenue can be limited to FY26 months unless "include all history" is enabled in the app.
