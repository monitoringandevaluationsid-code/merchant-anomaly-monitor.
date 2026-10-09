# Merchant Sales Anomaly Monitor — GitHub & Streamlit V8

**BSc (Hons) Data Science and Artificial Intelligence | Big Data Analytics | Synthetic merchant portfolios.**

## What this repository contains

One main Python application, `app.py`, supports both the assignment's **single-page user interface** and local **batch modelling**.

1. **Merchant investigation:** choose Original or New portfolio, choose the available classifier and merchant, inspect 90-day Captured sales against a past-only expectation, and see red markers for unusual dates. A flagged date drives the *same-date* business-category drill-down.
2. **Transaction Explorer (Section 5):** view individual Captured transactions from both portfolios using merchant, date and business-category slicers; download filtered transactions.
3. **Upload & Detect (Section 6):** upload a different `.csv` or `.csv.gz`, add optional merchant/business/status lookups, run a freshly fitted Isolation Forest, and download predicted anomalies, flagged days, category analysis and captured transactions in a ZIP.
4. **Model comparison:** Original Isolation Forest; New Isolation Forest, Random Forest, Gradient Boosting, Logistic Regression. The original supervised predictions are not included because no evaluated copies were supplied.

The **initial Excel/PivotTable wrangling** is documented in [EXCEL_PREPARATION.md](EXCEL_PREPARATION.md).

## Update your existing GitHub repository

Repository: `monitoringandevaluationsid-code/merchant-anomaly-monitor` — branch `main`.

1. Extract the release ZIP **locally**. Do not upload the ZIP itself into the GitHub repository.
2. At repository root, replace `app.py` and `requirements.txt`, and add `README.md`, `EXCEL_PREPARATION.md`, `GITHUB_DEPLOYMENT.md`, `UPLOAD_EXAMPLES.md`, `validate_deployment.py` and `.gitignore`.
3. Open the existing **lowercase plural `outputs/`** folder and upload **all 12 files from the local `outputs/` folder**. Keep the supplied canonical names.
4. Optionally add `templates/` to demo Upload & Detect. Commit changes to `main`.
5. Your existing Streamlit Cloud app should redeploy. Main app path remains `app.py`. Select both portfolios to check the Transaction Explorer and select `templates/sample_transactions_readable.csv` for an upload test.

**Upload size:** every individual file in this package is under GitHub's 25 MB browser-upload limit. Both large raw transaction sources are **excluded** from the GitHub outputs; only their **Captured-only compressed CSVs** are included. Do not upload `Transactions.rar` or the 94 MB Excel workbook.

## Key reconciled values

| Portfolio | Captured transactions | Captured sales (USD) | Merchant-days | Category-days | Period |
|---|---:|---:|---:|---:|---|
| Original | 482,835 | 158,371,987.40 | 9,450 | 47,250 | Jan–Mar 2025 |
| New | 481,350 | 157,979,217.37 | 9,450 | 47,250 | Jan–Mar 2025 |

Uploaded evaluator-ready predictions: Original IF **11** flagged; New IF **13**, RF **13**, GB **13**, LR **15**. These counts reconcile with saved predictions. External evaluator scores originate from the university's recorded results; fresh uploads do **not** have those labels or scores.

## Local run

```bash
python -m pip install -r requirements.txt
python validate_deployment.py
streamlit run app.py
```

For a fresh batch run from raw inputs (keep in local `raw_data/`):

```bash
python app.py --mode batch --data-dir raw_data --output-dir regenerated_outputs
```

Compare any newly trained predictions **merchant-by-merchant** against saved evaluated versions before using the university Model Evaluator metrics.

## Data safety

Only synthetic course data should be published in this public academic repository. The Streamlit upload facility does not deliberately commit files to GitHub, but a public cloud service may still retain or log data; do not upload real bank or personally confidential datasets.


## V8 — unified merchant selector (9 Oct 2026)
The old separate **Merchant filter** selector is removed. The single **Select merchant** dropdown contains **All merchants** followed by each individual merchant. The initial All merchants selection plots combined Captured daily sales and expected sales and shows dates on which at least one individual merchant produced a historical alert. It does **not** classify the portfolio as anomalous. The date selector, category comparison, transaction counts and CSV download adapt to the selection. Selecting a named merchant restores the original per-merchant details and same-date drill-down. All V8 dataset, prediction and transaction files remain unchanged.


## Final V8 presentation notes
The single merchant dropdown defaults to All merchants. On a selected portfolio alert date, headline cards show compact USD totals while an exact-value caption preserves 2-decimal amounts. Category drill-down follows the selected date. Transaction Explorer defaults to All categories. Use the five unchanged outputs/predictions_*.csv files for the university evaluator; additional batch-generated Original supervised predictions have not been externally evaluated.

See CHANGES_V8.md and DATA_MANIFEST.json.
