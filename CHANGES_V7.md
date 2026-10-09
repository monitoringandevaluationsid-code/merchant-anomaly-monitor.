# V7 — Single Merchant Dropdown

This release replaces the separate Merchant filter and merchant dropdown with one searchable **Select merchant** control.

- **All merchants** is the first choice and default. It aggregates Captured daily sales and transaction counts for the selected Original or New portfolio. It displays the dates on which one or more individual merchants triggered past-only historical alerts, without assigning a portfolio-level anomaly classification. The chart, date investigation and five business-category aggregates follow the chosen date.
- Individual merchant choices retain the earlier Isolation Forest / other available model classification, selected severe anomaly date, category drill-down and downloads.
- `Flagged dates only` continues to restrict the date picker.
- Transaction Explorer works for either the current merchant or All merchants; broad transaction selections remain subject to the 100,000-row display/export cap. Narrow the date/category to export.
- All 12 prepared datasets and evaluator prediction files are **unchanged**. Model Evaluator scores are not recalculated.

## Update existing GitHub
1. At repository root, replace **app.py** with this release and commit to `main`.
2. Optionally add `CHANGES_V7.md` and revised README.
3. Do not reupload `outputs/` unless files are missing.
4. Reopen Streamlit Cloud and choose **All merchants**, then select **Rachel Sheppard** for 13 February 2025. Repeat on New portfolio.

## Validation
Verified 105 merchants and 90 dates per portfolio. Aggregated captured-sales totals and alert counts reconcile. A live browser interaction test is still required after GitHub deployment.
