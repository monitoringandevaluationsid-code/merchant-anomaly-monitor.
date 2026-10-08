# Merchant Sales Anomaly Monitor

**Single Python entry point:** `app.py`.

## Streamlit Cloud
Push this repository to GitHub. At https://share.streamlit.io select repository, branch `main`, entrypoint `app.py`. Only the original portfolio has complete prepared data in this package. The New portfolio option appears after `daily_sales_new_prepared.csv` and `business_daily_new_prepared.csv` are added to `outputs/`.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Regenerate full output from raw source CSVs
Place original `Transactions.csv`, `merchant.csv`, `business.csv`, `status.csv` and new files `Transactions_New.csv`, `merchant_New.csv`, `business_New.csv`, `status_New.csv` into `raw_data/`. Run:
```bash
python app.py --mode batch --data-dir raw_data --output-dir generated_outputs
```
Validate generated files against evaluator-used predictions *before* replacing current `outputs/`. Once validated, copy `generated_outputs/*.csv` to `outputs/`, commit and deploy. Do not put confidential raw files in a public repository.

## Important
A model is only selectable when its prediction CSV exists. Original portfolio's three supervised predictions and new portfolio's daily/category CSVs are not supplied. The new prediction files uploaded earlier flag 13/13/13/15 merchants (IF/RF/GB/LR), which conflict with historical presentation counts. Do not claim these reproduce the external evaluator until checked. Original portfolio supervised predictions are in-sample pseudo-label fits.
