# Upload examples and interpretation

Choose **section 6 (Upload & Detect)** on the single-page dashboard.

- For directly named records, upload `templates/sample_transactions_readable.csv`. There is no need for lookup CSVs.
- For coded records, upload `templates/sample_transactions_coded.csv` together with the three sample lookup CSVs (`sample_merchant_lookup.csv`, `sample_business_lookup.csv`, `sample_status_lookup.csv`).
- `Transaction_ID` is optional, but any supplied values must be unique.
- `Status` must be supplied directly, or resolved through a status lookup. Only rows resolved as **Captured** contribute to sales. If the file truly contains only Captured records and lacks any status field, tick the explicit confirmation checkbox.

The bundled examples are **artificial demonstration data** unrelated to the university evaluator. They deliberately contain extreme sales changes to help illustrate charts. Their classifications/accuracy cannot be compared to the university evaluator's hidden merchant labels.

Outputs of Upload & Detect:

1. `merchant_predictions.csv`: one merchant and a binary `Prediction` (1 anomalous, 0 normal).
2. `merchant_risk_scores.csv`: rank, Isolation Forest score and feature values.
3. `daily_sales_anomaly_monitor.csv`: Captured sales, count, past-only expected values and day flags.
4. `flagged_merchant_days.csv`: anomaly-day dates, severity and relative differences.
5. `business_category_daily.csv`: category actual/expected and category deviation.
6. `captured_transactions.csv`: valid Captured rows and resolved labels.
7. `README_RESULTS.txt`: analysis parameters and caveats.
