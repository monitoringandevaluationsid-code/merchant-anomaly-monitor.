"""
Big Data Analytics - Merchant Anomaly Detection Tool
Unified batch and Streamlit application.

Modes of use
------------
1) Batch mode:
   python app.py --mode batch --data-dir . --output-dir outputs

   Required original files in --data-dir:
   - Transactions.csv
   - merchant.csv
   - business.csv
   - status.csv

   Optional new-dataset files in --data-dir:
   - Transactions_New.csv
   - merchant_New.csv
   - business_New.csv
   - status_New.csv

2) Streamlit dashboard:
   streamlit run app.py -- --mode app

The script performs the full workflow: data ingestion, preparation, captured-only
sales aggregation, feature engineering, Isolation Forest modelling, supervised
model comparison, evaluator-ready prediction exports, and a single-page UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
FINAL_CONTAMINATION = 0.10
ROBUST_Z_THRESHOLD = 4.0
MIN_RELATIVE_FACTOR = 1.5

FEATURE_COLUMNS = [
    "Sales_CV",
    "Transaction_CV",
    "Peak_Positive_Deviation",
    "Peak_Negative_Deviation",
    "P95_Absolute_Deviation",
    "Mean_Absolute_Deviation",
    "High_Confidence_Day_Count",
    "Max_Robust_Z",
    "Largest_Daily_Log_Change",
    "Period_Shift",
]

ORIGINAL_FILES = {
    "transactions": "Transactions.csv",
    "merchant": "merchant.csv",
    "business": "business.csv",
    "status": "status.csv",
}

NEW_FILES = {
    "transactions": "Transactions_New.csv",
    "merchant": "merchant_New.csv",
    "business": "business_New.csv",
    "status": "status_New.csv",
}


def clean_status_code(series: pd.Series) -> pd.Series:
    """Normalise status codes such as 1, '1', '01', '1.0' to two-character strings."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def locate_files(data_dir: Path, mapping: Dict[str, str]) -> Dict[str, Path]:
    paths = {key: data_dir / filename for key, filename in mapping.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    return paths


def read_source_files(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(paths["transactions"], parse_dates=["Date"], dtype={"Status_Code": "string"})
    merchant = pd.read_csv(paths["merchant"])
    business = pd.read_csv(paths["business"])
    status = pd.read_csv(paths["status"], dtype={"Status_Code": "string"})

    tx["Status_Code"] = clean_status_code(tx["Status_Code"])
    status["Status_Code"] = clean_status_code(status["Status_Code"])
    return tx, merchant, business, status


def prepare_daily_data(
    transactions: pd.DataFrame,
    merchant: pd.DataFrame,
    business: pd.DataFrame,
    status: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge references, filter captured sales and create daily merchant/business data."""
    df = (
        transactions
        .merge(merchant, on="Merchant_ID", how="left", validate="many_to_one")
        .merge(business, on="Business_ID", how="left", validate="many_to_one")
        .merge(status, on="Status_Code", how="left", validate="many_to_one")
    )

    captured = df[df["Status"].eq("Captured")].copy()

    calendar = pd.date_range(transactions["Date"].min(), transactions["Date"].max(), freq="D", name="Date")

    daily_agg = (
        captured.groupby(["Merchant_ID", "Date"], as_index=False)
        .agg(Daily_Sales=("Amount", "sum"), Transaction_Count=("Transaction_ID", "size"))
    )

    merchant_calendar = pd.MultiIndex.from_product(
        [merchant["Merchant_ID"].unique(), calendar], names=["Merchant_ID", "Date"]
    ).to_frame(index=False)

    daily = merchant_calendar.merge(daily_agg, on=["Merchant_ID", "Date"], how="left")
    daily[["Daily_Sales", "Transaction_Count"]] = daily[["Daily_Sales", "Transaction_Count"]].fillna(0)
    daily = daily.merge(merchant[["Merchant_ID", "Merchant"]], on="Merchant_ID", how="left")
    daily = daily.sort_values(["Merchant_ID", "Date"]).reset_index(drop=True)

    business_agg = (
        captured.groupby(["Merchant_ID", "Business_ID", "Date"], as_index=False)
        .agg(Business_Sales=("Amount", "sum"), Business_Transactions=("Transaction_ID", "size"))
    )

    business_calendar = pd.MultiIndex.from_product(
        [merchant["Merchant_ID"].unique(), business["Business_ID"].unique(), calendar],
        names=["Merchant_ID", "Business_ID", "Date"],
    ).to_frame(index=False)

    business_daily = business_calendar.merge(
        business_agg, on=["Merchant_ID", "Business_ID", "Date"], how="left"
    )
    business_daily[["Business_Sales", "Business_Transactions"]] = business_daily[
        ["Business_Sales", "Business_Transactions"]
    ].fillna(0)
    business_daily = (
        business_daily
        .merge(merchant[["Merchant_ID", "Merchant"]], on="Merchant_ID", how="left")
        .merge(business[["Business_ID", "Business"]], on="Business_ID", how="left")
        .sort_values(["Merchant_ID", "Business_ID", "Date"])
        .reset_index(drop=True)
    )

    return df, daily, business_daily


def add_time_series_features(daily: pd.DataFrame, business_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.copy()
    daily["Weekday"] = daily["Date"].dt.dayofweek
    g = daily.groupby("Merchant_ID", sort=False)

    daily["Prior_14_Median"] = g["Daily_Sales"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=7).median()
    )
    daily["Weekday_Median"] = daily.groupby(["Merchant_ID", "Weekday"])["Daily_Sales"].transform(
        lambda s: s.shift(1).expanding(min_periods=4).median()
    )
    daily["Expected_Sales"] = daily["Weekday_Median"].fillna(daily["Prior_14_Median"])
    daily["Log_Deviation"] = np.log((daily["Daily_Sales"] + 1) / (daily["Expected_Sales"] + 1))
    daily["Abs_Log_Deviation"] = daily["Log_Deviation"].abs()
    daily["Candidate_Extreme_Day"] = daily["Abs_Log_Deviation"].ge(np.log(MIN_RELATIVE_FACTOR))
    daily["Daily_Log_Change"] = daily.groupby("Merchant_ID")["Daily_Sales"].transform(
        lambda s: np.log1p(s).diff().abs()
    )

    parts = []
    for _, grp in daily.groupby("Merchant_ID", sort=False):
        grp = grp.copy()
        shifted = grp["Log_Deviation"].shift(1)
        grp["Residual_Median"] = shifted.rolling(28, min_periods=14).median()
        grp["Residual_MAD"] = shifted.rolling(28, min_periods=14).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        grp["Robust_Z"] = 0.6745 * (grp["Log_Deviation"] - grp["Residual_Median"]) / grp[
            "Residual_MAD"
        ].replace(0, np.nan)
        parts.append(grp)
    daily = pd.concat(parts, ignore_index=True).sort_values(["Merchant_ID", "Date"]).reset_index(drop=True)
    daily["High_Confidence_Day"] = (
        daily["Robust_Z"].abs().ge(ROBUST_Z_THRESHOLD)
        & daily["Abs_Log_Deviation"].ge(np.log(MIN_RELATIVE_FACTOR))
    )

    business_daily = business_daily.copy()
    business_daily["Weekday"] = business_daily["Date"].dt.dayofweek
    business_daily["Prior_14_Median"] = business_daily.groupby(["Merchant_ID", "Business_ID"])[
        "Business_Sales"
    ].transform(lambda s: s.shift(1).rolling(14, min_periods=7).median())
    business_daily["Weekday_Median"] = business_daily.groupby(["Merchant_ID", "Business_ID", "Weekday"])[
        "Business_Sales"
    ].transform(lambda s: s.shift(1).expanding(min_periods=4).median())
    business_daily["Expected_Business_Sales"] = business_daily["Weekday_Median"].fillna(
        business_daily["Prior_14_Median"]
    )
    business_daily["Business_Deviation"] = np.log(
        (business_daily["Business_Sales"] + 1) / (business_daily["Expected_Business_Sales"] + 1)
    )
    business_daily["Abs_Business_Deviation"] = business_daily["Business_Deviation"].abs()
    return daily, business_daily


def build_merchant_features(daily: pd.DataFrame) -> pd.DataFrame:
    merchant_features = daily.groupby("Merchant_ID").agg(
        Mean_Sales=("Daily_Sales", "mean"),
        SD_Sales=("Daily_Sales", "std"),
        Mean_Transactions=("Transaction_Count", "mean"),
        SD_Transactions=("Transaction_Count", "std"),
        Peak_Positive_Deviation=("Log_Deviation", lambda x: x.clip(lower=0).max()),
        Peak_Negative_Deviation=("Log_Deviation", lambda x: (-x.clip(upper=0)).max()),
        P95_Absolute_Deviation=("Abs_Log_Deviation", lambda x: x.quantile(0.95)),
        Mean_Absolute_Deviation=("Abs_Log_Deviation", "mean"),
        High_Confidence_Day_Count=("High_Confidence_Day", "sum"),
        Max_Robust_Z=("Robust_Z", lambda x: x.abs().max()),
    )

    merchant_features["Sales_CV"] = merchant_features["SD_Sales"] / (merchant_features["Mean_Sales"] + 1)
    merchant_features["Transaction_CV"] = merchant_features["SD_Transactions"] / (
        merchant_features["Mean_Transactions"] + 1
    )
    merchant_features["Largest_Daily_Log_Change"] = daily.groupby("Merchant_ID")["Daily_Log_Change"].max()

    ordered = daily.sort_values(["Merchant_ID", "Date"])
    first_30 = ordered.groupby("Merchant_ID").head(30).groupby("Merchant_ID")["Daily_Sales"].mean()
    last_30 = ordered.groupby("Merchant_ID").tail(30).groupby("Merchant_ID")["Daily_Sales"].mean()
    merchant_features["Period_Shift"] = np.log((last_30 + 1) / (first_30 + 1))
    merchant_features["Max_Robust_Z"] = merchant_features["Max_Robust_Z"].clip(upper=20)
    return merchant_features


def prepare_X(features: pd.DataFrame, medians: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    X = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    if medians is None:
        medians = X.median()
    X = X.fillna(medians).fillna(0)
    return X, medians


def save_submission(
    merchant_lookup: pd.DataFrame,
    feature_index: Iterable[int],
    predictions: np.ndarray,
    filename: Path,
) -> pd.DataFrame:
    pred_series = pd.Series(predictions, index=pd.Index(feature_index, name="Merchant_ID"), name="Prediction")
    output = merchant_lookup[["Merchant_ID", "Merchant"]].copy()
    output["Prediction"] = output["Merchant_ID"].map(pred_series).astype(int)
    output = output[["Merchant", "Prediction"]].sort_values("Merchant").reset_index(drop=True)

    assert output["Merchant"].notna().all()
    assert output["Merchant"].nunique() == len(output)
    assert set(output["Prediction"].unique()).issubset({0, 1})

    filename.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(filename, index=False)
    print(f"Saved {filename} | rows={len(output)} | anomalies={int(output['Prediction'].sum())}")
    return output


def train_original(data_dir: Path, output_dir: Path):
    paths = locate_files(data_dir, ORIGINAL_FILES)
    transactions, merchant, business, status = read_source_files(paths)
    merged, daily, business_daily = prepare_daily_data(transactions, merchant, business, status)
    daily, business_daily = add_time_series_features(daily, business_daily)
    merchant_features = build_merchant_features(daily)
    X, medians = prepare_X(merchant_features)

    iso_model = IsolationForest(
        n_estimators=500,
        contamination=FINAL_CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    iso_pred = (iso_model.fit_predict(X) == -1).astype(int)
    iso_score = -iso_model.score_samples(X)

    merchant_risk = merchant_features.assign(Prediction=iso_pred, Anomaly_Score=iso_score).reset_index()
    merchant_risk = merchant_risk.merge(merchant[["Merchant_ID", "Merchant"]], on="Merchant_ID", how="left")
    merchant_risk = merchant_risk.sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)
    merchant_risk["Risk_Rank"] = np.arange(1, len(merchant_risk) + 1)
    merchant_risk["Risk_Label"] = np.where(merchant_risk["Prediction"].eq(1), "Flagged", "Normal")

    output_dir.mkdir(parents=True, exist_ok=True)
    captured_tx = merged.loc[merged["Status"].eq("Captured")].copy()
    captured_tx.to_csv(output_dir / "transactions_original_captured.csv.gz", index=False, compression="gzip")
    daily.to_csv(output_dir / "daily_sales_prepared.csv", index=False)
    business_daily.to_csv(output_dir / "business_daily_prepared.csv", index=False)
    merchant_risk.to_csv(output_dir / "merchant_risk.csv", index=False)
    save_submission(merchant, merchant_features.index, iso_pred, output_dir / "predictions_original_isolation_forest.csv")
    # Supervised models are trained on original IF pseudo-labels. Original-set
    # predictions are IN-SAMPLE displays only, not independent evaluation.
    ordered_original = merchant.set_index("Merchant_ID").loc[merchant_features.index].reset_index()
    supervised_models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    for label, fitted in supervised_models.items():
        if label == "Gradient Boosting":
            fitted.fit(X, iso_pred, sample_weight=compute_sample_weight(class_weight="balanced", y=iso_pred))
        else:
            fitted.fit(X, iso_pred)
        suffix = label.lower().replace(" ", "_")
        save_submission(ordered_original, merchant_features.index, fitted.predict(X), output_dir / f"predictions_original_{suffix}.csv")

    return {
        "merchant": merchant,
        "daily": daily,
        "business_daily": business_daily,
        "features": merchant_features,
        "X": X,
        "medians": medians,
        "iso_model": iso_model,
        "iso_pred": iso_pred,
        "merchant_risk": merchant_risk,
        "supervised_models": supervised_models,
    }


def train_new_if_available(data_dir: Path, output_dir: Path, original_state: dict):
    if not all((data_dir / f).exists() for f in NEW_FILES.values()):
        print("New data files not found. Skipping Deliverable 2 prediction exports.")
        return None

    paths = locate_files(data_dir, NEW_FILES)
    transactions, merchant_new, business_new, status_new = read_source_files(paths)
    _, daily_new, business_daily_new = prepare_daily_data(transactions, merchant_new, business_new, status_new)
    daily_new, business_daily_new = add_time_series_features(daily_new, business_daily_new)
    features_new = build_merchant_features(daily_new)
    X_new, _ = prepare_X(features_new, original_state["medians"])

    X_train = original_state["X"]
    y_train = original_state["iso_pred"]
    iso_model = original_state["iso_model"]

    merchant_new_ordered = merchant_new.set_index("Merchant_ID").loc[features_new.index].reset_index()

    pred_if = (iso_model.predict(X_new) == -1).astype(int)
    save_submission(merchant_new_ordered, features_new.index, pred_if, output_dir / "predictions_new_isolation_forest.csv")

    fitted = original_state["supervised_models"]
    pred_lr = fitted["Logistic Regression"].predict(X_new)
    save_submission(merchant_new_ordered, features_new.index, pred_lr, output_dir / "predictions_new_logistic_regression.csv")
    pred_rf = fitted["Random Forest"].predict(X_new)
    save_submission(merchant_new_ordered, features_new.index, pred_rf, output_dir / "predictions_new_random_forest.csv")
    pred_gb = fitted["Gradient Boosting"].predict(X_new)
    save_submission(merchant_new_ordered, features_new.index, pred_gb, output_dir / "predictions_new_gradient_boosting.csv")

    merged_new, _, _ = prepare_daily_data(transactions, merchant_new, business_new, status_new)
    merged_new.loc[merged_new["Status"].eq("Captured")].to_csv(
        output_dir / "transactions_new_captured.csv.gz", index=False, compression="gzip")
    daily_new.to_csv(output_dir / "daily_sales_new_prepared.csv", index=False)
    business_daily_new.to_csv(output_dir / "business_daily_new_prepared.csv", index=False)
    return {
        "daily_new": daily_new,
        "business_daily_new": business_daily_new,
        "features_new": features_new,
        "X_new": X_new,
        "merchant_new": merchant_new_ordered,
        "predictions": {"isolation_forest": pred_if, "logistic_regression": pred_lr, "random_forest": pred_rf, "gradient_boosting": pred_gb},
    }


def run_batch(data_dir: Path, output_dir: Path) -> None:
    state = train_original(data_dir, output_dir)
    train_new_if_available(data_dir, output_dir, state)
    print("\nBatch run completed.")
    print(f"Outputs written to: {output_dir.resolve()}")




# ---------- User-uploaded transaction analysis (session-only; not evaluator-tested) ----------

def _adapt_upload_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept common spellings without silently guessing unrelated columns."""
    aliases = {
        'Transaction_ID': ('transactionid', 'transaction_id', 'txnid', 'txn_id'),
        'Date': ('date', 'transaction_date', 'transactiondate'),
        'Amount': ('amount', 'transaction_amount', 'salesamount', 'sales_amount'),
        'Merchant': ('merchant', 'merchant_name', 'merchantname'),
        'Merchant_ID': ('merchant_id', 'merchantid'),
        'Business': ('business', 'business_category', 'category'),
        'Business_ID': ('business_id', 'businessid'),
        'Status': ('status', 'transaction_status'),
        'Status_Code': ('status_code', 'statuscode'),
    }
    columns = {str(c).strip().lower(): c for c in frame.columns}
    changes = {}
    for target, alternatives in aliases.items():
        if target in frame.columns:
            continue
        matched = [columns[a] for a in alternatives if a in columns]
        if matched:
            changes[matched[0]] = target
    return frame.rename(columns=changes)


def _lookup_upload_values(df: pd.DataFrame, lookups: Optional[pd.DataFrame],
                          id_col: str, name_col: str, *, required=True) -> pd.Series:
    if name_col in df.columns:
        values = df[name_col].astype('string').str.strip()
        if values.isna().any() or values.eq('').any():
            raise ValueError(f'{name_col} contains blank values.')
        return values
    if id_col not in df.columns:
        if not required:
            return pd.Series(['Unspecified'] * len(df), index=df.index, dtype='string')
        raise ValueError(f'Missing {name_col} or {id_col}.')
    if lookups is None:
        raise ValueError(f'{name_col} lookup CSV is required when transactions use {id_col}.')
    lookups = _adapt_upload_column_names(lookups)
    if id_col not in lookups.columns or name_col not in lookups.columns:
        raise ValueError(f'{name_col} reference must contain {id_col} and {name_col}.')
    ref = lookups[[id_col, name_col]].copy()
    ref['_key'] = ref[id_col].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    if ref['_key'].duplicated().any():
        raise ValueError(f'Duplicate {id_col} keys in the lookup CSV.')
    keys = df[id_col].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    values = keys.map(ref.set_index('_key')[name_col]).astype('string').str.strip()
    if values.isna().any() or values.eq('').any():
        count = int(values.isna().sum() + values.eq('').sum())
        raise ValueError(f'{count:,} transaction rows have unmapped {id_col} values.')
    return values


def normalize_uploaded_transactions(transactions: pd.DataFrame,
                                    merchant_lookup: Optional[pd.DataFrame] = None,
                                    business_lookup: Optional[pd.DataFrame] = None,
                                    status_lookup: Optional[pd.DataFrame] = None,
                                    *, confirmed_captured: bool = False):
    """Strictly validate a user CSV and convert it to the official assignment schema.

    Missing statuses are never silently interpreted as Captured.
    """
    raw = _adapt_upload_column_names(transactions.copy())
    if len(raw) == 0:
        raise ValueError('The transaction file is empty.')
    if len(raw) > 2_000_000:
        raise ValueError('Maximum 2 million transaction rows per interactive analysis. Split this file.')
    for col in ('Date', 'Amount'):
        if col not in raw.columns:
            raise ValueError(f'Missing required transaction column: {col}.')
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce').dt.normalize()
    raw['Amount'] = pd.to_numeric(raw['Amount'], errors='coerce')
    if raw['Date'].isna().any() or raw['Amount'].isna().any():
        raise ValueError('Missing or invalid Date/Amount values. Correct those rows before analysis.')
    if not np.isfinite(raw['Amount'].to_numpy(dtype=float)).all():
        raise ValueError('Amount includes non-finite numbers.')
    if 'Transaction_ID' not in raw:
        raw['Transaction_ID'] = np.arange(1, len(raw) + 1, dtype=np.int64)
    if raw['Transaction_ID'].isna().any() or raw['Transaction_ID'].duplicated().any():
        raise ValueError('Transaction_ID must be nonblank and unique within the uploaded dataset.')

    raw['Merchant'] = _lookup_upload_values(raw, merchant_lookup, 'Merchant_ID', 'Merchant')
    raw['Business'] = _lookup_upload_values(raw, business_lookup, 'Business_ID', 'Business', required=False)
    if 'Status' in raw.columns:
        status = raw['Status'].astype('string').str.strip()
        if status.isna().any() or status.eq('').any():
            raise ValueError('Status contains blank values.')
    elif 'Status_Code' in raw.columns:
        status = _lookup_upload_values(raw, status_lookup, 'Status_Code', 'Status')
    elif confirmed_captured:
        status = pd.Series(['Captured'] * len(raw), index=raw.index)
    else:
        raise ValueError('Upload a Status lookup or use a Status column. If every record is Captured, confirm this explicitly.')
    raw['Status'] = status.astype('string').str.strip()
    raw['Status'] = raw['Status'].where(~raw['Status'].str.casefold().eq('captured'), 'Captured')
    raw['Merchant'] = raw['Merchant'].astype(str)
    raw['Business'] = raw['Business'].astype(str)
    stats = {'input_rows':len(raw), 'captured_rows':int(raw.Status.eq('Captured').sum()),
             'excluded_rows':int((~raw.Status.eq('Captured')).sum()),
             'status_counts':raw.Status.value_counts().to_dict(),
             'negative_amounts':int(raw.Amount.lt(0).sum())}
    if not stats['captured_rows']:
        raise ValueError('No transactions with resolved Status = Captured. Cannot calculate sales.')

    merchants = sorted(raw.Merchant.unique().tolist())
    businesses = sorted(raw.Business.unique().tolist())
    if len(merchants) < 3:
        raise ValueError('At least three distinct merchants are required for portfolio-level Isolation Forest comparison.')
    if len(merchants) > 500:
        raise ValueError('Maximum 500 merchants for interactive analysis. Use local batch mode for larger datasets.')
    dates = raw.Date.max() - raw.Date.min()
    if dates.days > 366:
        raise ValueError('The interactive analysis supports at most a 367-day calendar span. Use a shorter date range.')
    if raw.loc[raw.Status.eq('Captured'),'Date'].nunique() < 14:
        raise ValueError('At least 14 distinct dates with Captured transactions are required.')

    merchant_dim = pd.DataFrame({'Merchant_ID':np.arange(1,len(merchants)+1), 'Merchant':merchants})
    business_dim = pd.DataFrame({'Business_ID':np.arange(1,len(businesses)+1), 'Business':businesses})
    statuses = sorted(raw.Status.unique().tolist())
    status_dim = pd.DataFrame({'Status_Code':[f'{i:02d}' for i in range(1,len(statuses)+1)], 'Status':statuses})
    merchant_map = merchant_dim.set_index('Merchant')['Merchant_ID']
    business_map = business_dim.set_index('Business')['Business_ID']
    status_map = status_dim.set_index('Status')['Status_Code']
    normalized = raw[['Transaction_ID','Date','Merchant','Business','Status','Amount']].copy()
    normalized['Merchant_ID'] = normalized.Merchant.map(merchant_map).astype(int)
    normalized['Business_ID'] = normalized.Business.map(business_map).astype(int)
    normalized['Status_Code'] = normalized.Status.map(status_map)
    source = normalized[['Transaction_ID','Date','Merchant_ID','Business_ID','Status_Code','Amount']]
    return source, merchant_dim, business_dim, status_dim, stats


def analyze_uploaded_transactions(transactions: pd.DataFrame,
                                   merchant_lookup: Optional[pd.DataFrame] = None,
                                   business_lookup: Optional[pd.DataFrame] = None,
                                   status_lookup: Optional[pd.DataFrame] = None,
                                   *, contamination: float = 0.10,
                                   confirmed_captured: bool = False) -> dict:
    if not (0.01 <= contamination <= 0.30):
        raise ValueError('Contamination must be between 0.01 and 0.30.')
    source, merchants, businesses, statuses, stats = normalize_uploaded_transactions(
        transactions, merchant_lookup, business_lookup, status_lookup,
        confirmed_captured=confirmed_captured)
    merged, daily, business_daily = prepare_daily_data(source, merchants, businesses, statuses)
    daily, business_daily = add_time_series_features(daily, business_daily)
    features = build_merchant_features(daily)
    X, _ = prepare_X(features)
    model = IsolationForest(n_estimators=500, contamination=contamination, random_state=42)
    preds = (model.fit_predict(X)==-1).astype(int)
    scores = -model.score_samples(X)
    risk = features.assign(Prediction=preds, Anomaly_Score=scores).reset_index()
    risk = risk.merge(merchants[['Merchant_ID','Merchant']],on='Merchant_ID',how='left')
    risk = risk.sort_values('Anomaly_Score',ascending=False).reset_index(drop=True)
    risk.insert(0,'Risk_Rank',np.arange(1,len(risk)+1))
    risk['Classification'] = np.where(risk.Prediction.eq(1),'Anomalous','Normal')
    predictions = risk[['Merchant','Prediction']].sort_values('Merchant').reset_index(drop=True)
    day_flags = daily.loc[daily.High_Confidence_Day.astype(bool)].copy()
    day_flags['Difference_USD'] = day_flags.Daily_Sales-day_flags.Expected_Sales
    day_flags['Deviation_Pct'] = np.where(day_flags.Expected_Sales.gt(0),
                                            100*day_flags.Difference_USD/day_flags.Expected_Sales,np.nan)
    category_flags = business_daily.copy()
    category_flags['Difference_USD'] = category_flags.Business_Sales-category_flags.Expected_Business_Sales
    return {'predictions':predictions, 'risk':risk, 'daily':daily, 'business':business_daily,
            'flagged_days':day_flags,'category':category_flags,
            'captured':merged.loc[merged.Status.eq('Captured')].copy(),
            'stats':stats, 'settings':{'contamination':contamination,'n_estimators':500,
                                      'random_state':42,'independent_evaluation':False}}


def uploaded_results_zip(results: dict) -> bytes:
    import io
    import zipfile
    buf=io.BytesIO()
    outputs={'merchant_predictions.csv':results['predictions'],
             'merchant_risk_scores.csv':results['risk'],
             'daily_sales_anomaly_monitor.csv':results['daily'],
             'flagged_merchant_days.csv':results['flagged_days'],
             'business_category_daily.csv':results['category'],
             'captured_transactions.csv':results['captured']}
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,df in outputs.items():
            z.writestr(name,df.to_csv(index=False))
        z.writestr('README_RESULTS.txt',
            'Session-only Isolation Forest screening; 500 estimators, seed 42.\n'
            f'Contamination: {results["settings"]["contamination"]}\n'
            f'Input records: {results["stats"]["input_rows"]}\n'
            f'Captured records: {results["stats"]["captured_rows"]}\n'
            'Predictions are UNSUPERVISED SCREENING outputs, not verified fraud findings.\n'
            'No university Model Evaluator accuracy/F1 claims apply to these uploaded records.\n')
    return buf.getvalue()


def aggregate_portfolio_for_display(daily: pd.DataFrame, business: pd.DataFrame):
    """Create display-only, Captured-only portfolio totals without retraining models.

    Red dates indicate the presence of one or more individual merchant alerts;
    they are NOT portfolio-level anomaly classifications. Expectations are
    withheld unless every merchant/category has a valid prior-only baseline.
    """
    grp = daily.groupby('Date', sort=True)
    p_daily = grp.agg(
        Daily_Sales=('Daily_Sales', 'sum'),
        Transaction_Count=('Transaction_Count', 'sum'),
        Expected_Sales=('Expected_Sales', lambda x: x.sum(min_count=1)),
        _expected_n=('Expected_Sales', 'count'),
        _merchant_n=('Merchant', 'nunique'),
        Flagged_Merchants=('Flagged_Day', 'sum'),
    ).reset_index()
    p_daily.loc[p_daily._expected_n.lt(p_daily._merchant_n), 'Expected_Sales'] = np.nan
    p_daily['Flagged_Day'] = p_daily.Flagged_Merchants.gt(0)
    p_daily['Merchant'] = 'All merchants'
    p_daily = p_daily.drop(columns=['_expected_n','_merchant_n'])

    bgrp = business.groupby(['Date','Business'], sort=True)
    p_business = bgrp.agg(
        Business_Sales=('Business_Sales','sum'),
        Business_Transactions=('Business_Transactions','sum'),
        Expected_Business_Sales=('Expected_Business_Sales', lambda x: x.sum(min_count=1)),
        _expected_n=('Expected_Business_Sales','count'),
        _merchant_n=('Merchant','nunique'),
    ).reset_index()
    p_business.loc[p_business._expected_n.lt(p_business._merchant_n), 'Expected_Business_Sales'] = np.nan
    p_business['Merchant'] = 'All merchants'
    p_business = p_business.drop(columns=['_expected_n','_merchant_n'])
    return p_daily, p_business


def render_transaction_and_upload_sections(st, ROOT: Path, daily: pd.DataFrame,
                                           business: pd.DataFrame, merchant: str, prefix: str):
    st.divider()
    st.subheader('5. Transaction Explorer — individual Captured records')
    st.caption('Inspect and download transaction records for a merchant, date range and business category. '
               'This section is distinct from the daily aggregate above.')
    transaction_file = ROOT / f'transactions_{prefix}_captured.csv.gz'
    with st.expander('Open transaction-level slicers and records',expanded=False):
        if not transaction_file.exists():
            st.info(f'No transaction-level extract is packaged for {prefix.title()}. '
                    'The daily sales and transaction counts above remain valid. '
                    'Generate the captured .csv.gz in batch mode or use Upload & Detect below.')
        else:
            opt1,opt2,opt3=st.columns([1.2,1.5,1.3])
            merchant_choices = ['All merchants'] if merchant == 'All merchants' else ['Selected merchant: '+merchant, 'All merchants']
            choice=opt1.selectbox('Transaction merchant',merchant_choices,key='txnmerchant_'+prefix)
            min_d,max_d=daily.Date.min().date(),daily.Date.max().date()
            selected_range=opt2.date_input('Transaction date range',value=(min_d,max_d),
                                           min_value=min_d,max_value=max_d,key='txnrange_'+prefix)
            categories=['All categories']+sorted(business.Business.dropna().astype(str).unique().tolist())
            cat_pick=opt3.selectbox('Business category',categories,key='txncat_'+prefix)
            if not isinstance(selected_range,(tuple,list)) or len(selected_range)!=2:
                st.info('Choose both a start date and an end date.')
            else:
                start_d,end_d=selected_range
                @st.cache_data(show_spinner=False,max_entries=12)
                def filtered_transaction_records(path,merchant_selection,start,end,business_selection):
                    pieces=[]; total=0; cap=100000
                    for batch in pd.read_csv(path,chunksize=80000):
                        if 'Status' not in batch.columns or 'Merchant' not in batch.columns or 'Date' not in batch.columns:
                            raise ValueError('Transaction extract must contain Status, Merchant and Date columns.')
                        batch=batch.loc[batch.Status.astype(str).str.casefold().eq('captured')].copy()
                        dates=pd.to_datetime(batch.Date,errors='coerce')
                        batch=batch.loc[dates.between(pd.Timestamp(start),pd.Timestamp(end))]
                        if merchant_selection!='All merchants':
                            batch=batch.loc[batch.Merchant.eq(merchant_selection)]
                        if business_selection!='All categories':
                            batch=batch.loc[batch.Business.eq(business_selection)]
                        total+=len(batch)
                        if sum(len(x) for x in pieces)<cap:
                            keep=cap-sum(len(x) for x in pieces)
                            pieces.append(batch.head(keep))
                    return (pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(),total)
                with st.spinner('Loading filtered Captured transactions...'):
                    try:
                        chosen=merchant if choice!='All merchants' else 'All merchants'
                        filtered,total=filtered_transaction_records(str(transaction_file),chosen,str(start_d),str(end_d),cat_pick)
                    except Exception as exc:
                        st.error(f'Transaction extract could not be read: {exc}')
                        filtered,total=pd.DataFrame(),0
                a,b,c=st.columns(3)
                a.metric('Matching Captured transactions',f'{total:,}')
                b.metric('Captured sales (USD)',f'{filtered.Amount.sum():,.2f}' if total<=100000 and 'Amount' in filtered else 'N/A — too many rows')
                c.metric('Rows loaded',f'{len(filtered):,}')
                if total>100000:
                    st.warning('The matched extract exceeds the 100,000-row export cap. Narrow the merchant or date range. '
                               'The sales total is withheld to avoid reporting a partial sum.')
                if not filtered.empty:
                    st.dataframe(filtered.head(1000),hide_index=True,use_container_width=True)
                    if total<=100000:
                        st.download_button('Download filtered Captured transactions CSV',
                                           filtered.to_csv(index=False).encode('utf-8'),
                                           file_name=f'{prefix}_captured_filtered.csv',mime='text/csv')
                else:
                    st.info('No Captured transactions match the selected filters.')

    st.divider()
    st.subheader('6. Upload & Detect — analyse another transaction CSV')
    st.caption('For demonstration with synthetic or authorised data only. Uploaded data are analysed within this '
               'session and are not committed to GitHub or saved by this application code. '
               'Do not upload confidential customer data to a public Streamlit deployment.')
    with st.expander('Upload files, run Isolation Forest and download results',expanded=False):
        st.markdown('Upload a `.csv` or `.csv.gz` transaction file. If it contains numeric merchant, business or '
                    'status codes rather than readable names, add the corresponding reference CSVs.')
        with st.form('uploaded_detection_form'):
            sourcefile=st.file_uploader('Transactions (CSV / CSV.GZ)',type=['csv','gz'],key='up_tx')
            u1,u2,u3=st.columns(3)
            merchantfile=u1.file_uploader('Merchant lookup (optional)',type=['csv'],key='up_merchant')
            businessfile=u2.file_uploader('Business lookup (optional)',type=['csv'],key='up_business')
            statusfile=u3.file_uploader('Status lookup (optional)',type=['csv'],key='up_status')
            confirmed=st.checkbox('My file contains only Captured transactions and has no Status column/code',value=False)
            cont=st.slider('Isolation Forest contamination (assumed fraction)',min_value=0.01,max_value=0.30,
                           value=0.10,step=0.01)
            submitted=st.form_submit_button('Analyse uploaded transactions',type='primary')
        if submitted:
            st.session_state.pop('last_uploaded_result',None)
            if sourcefile is None:
                st.error('Select a transaction file before running the analysis.')
            else:
                try:
                    with st.spinner('Validating, aggregating and fitting a new unsupervised model...'):
                        sourcefile.seek(0)
                        tx=pd.read_csv(sourcefile, compression='gzip' if sourcefile.name.lower().endswith('.gz') else None)
                        def get_lookup(upload):
                            if upload is None:return None
                            upload.seek(0)
                            return pd.read_csv(upload)
                        result=analyze_uploaded_transactions(tx,get_lookup(merchantfile),get_lookup(businessfile),
                                                             get_lookup(statusfile),contamination=cont,
                                                             confirmed_captured=confirmed)
                        st.session_state['last_uploaded_result']=result
                    st.success('Anomaly screening complete. Results can now be reviewed and downloaded.')
                except Exception as exc:
                    st.error(f'Validation or detection failed: {exc}')
        result=st.session_state.get('last_uploaded_result')
        if result is not None:
            stats=result['stats'];risk=result['risk'];flags=result['flagged_days']
            k1,k2,k3,k4=st.columns(4)
            k1.metric('Merchants analysed',f'{len(risk):,}')
            k2.metric('Flagged by Isolation Forest',f'{int(risk.Prediction.sum()):,}')
            k3.metric('Historical flagged merchant-days',f'{len(flags):,}')
            k4.metric('Captured transactions',f'{stats["captured_rows"]:,}')
            st.caption(f'{stats["excluded_rows"]:,} non-Captured rows excluded; '
                       f'{stats["negative_amounts"]:,} negative-amount rows observed. '
                       'These classifications have no independently verified accuracy or F1.')
            st.dataframe(risk[['Merchant','Prediction','Classification','Anomaly_Score','High_Confidence_Day_Count']].head(100),
                         hide_index=True,use_container_width=True)
            selectors=sorted(risk['Merchant'].astype(str).unique().tolist())
            pick=st.selectbox('Investigate uploaded merchant',selectors,key='investigate_upload_merchant')
            m=result['daily'].loc[result['daily'].Merchant.eq(pick)].sort_values('Date')
            import plotly.graph_objects as go
            fig=go.Figure()
            fig.add_scatter(x=m.Date,y=m.Daily_Sales,mode='lines',name='Actual sales')
            fig.add_scatter(x=m.Date,y=m.Expected_Sales,mode='lines',name='Past-only expected sales',line=dict(dash='dash'))
            fl=m.loc[m.High_Confidence_Day.astype(bool)]
            if not fl.empty:
                fig.add_scatter(x=fl.Date,y=fl.Daily_Sales,mode='markers',name='Flagged day',
                                marker=dict(color='#d83445',size=11))
            fig.update_layout(height=330,margin=dict(t=12,b=20,l=10,r=10),yaxis_title='Captured sales (USD)')
            st.plotly_chart(fig,use_container_width=True)
            if not fl.empty:
                pick_date=st.selectbox('Flagged date for category investigation',
                                      fl.Date.dt.strftime('%d %b %Y').tolist(),key='upload_flagged_date')
                dt=pd.to_datetime(pick_date,format='%d %b %Y')
                cats=result['category'].loc[(result['category'].Merchant.eq(pick)) &
                                            (result['category'].Date.eq(dt))].copy()
                st.dataframe(cats[['Business','Business_Sales','Expected_Business_Sales','Difference_USD']],
                             hide_index=True,use_container_width=True)
            else:
                st.info('This merchant has no separately flagged historical days; the period-level merchant classification '
                        'may still be anomalous.')
            a,b=st.columns(2)
            with a:
                st.download_button('Download merchant predictions CSV',
                                   result['predictions'].to_csv(index=False).encode('utf-8'),
                                   file_name='uploaded_predictions_isolation_forest.csv',mime='text/csv')
            with b:
                st.download_button('Download complete anomaly-results ZIP',uploaded_results_zip(result),
                                   file_name='uploaded_anomaly_results.zip',mime='application/zip')
            st.caption('Outputs include merchant predictions, risk scores, daily sales, flagged days, '
                       'business-category data and Captured transaction records. New uploads are evaluated independently '
                       'with a newly fitted unsupervised model, not the saved university evaluator.')



def render_streamlit_app(output_dir: Path) -> None:
    """Run the assignment-required single-page Streamlit interface."""
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="Merchant Sales Anomaly Monitor", page_icon="📊", layout="wide")
    ROOT = Path(output_dir)
    if not ROOT.exists():
        ROOT = Path(__file__).resolve().parent / "outputs"
    CONFIG = {
        "Original portfolio": ("daily_sales_prepared.csv", "business_daily_prepared.csv", "original"),
        "New portfolio": ("daily_sales_new_prepared.csv", "business_daily_new_prepared.csv", "new"),
    }
    MODELS = {
        "Isolation Forest": "isolation_forest",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "Logistic Regression": "logistic_regression",
    }

    @st.cache_data(show_spinner=False)
    def read_prepared(path):
        data = pd.read_csv(path, parse_dates=["Date"])
        return data

    @st.cache_data(show_spinner=False)
    def read_predictions(path):
        p = pd.read_csv(path)
        required = {"Merchant", "Prediction"}
        if not required.issubset(p.columns):
            raise ValueError(f"{path.name} lacks {required - set(p.columns)}")
        p["Prediction"] = pd.to_numeric(p["Prediction"], errors="raise").astype(int)
        if not p["Prediction"].isin([0, 1]).all() or p["Merchant"].duplicated().any():
            raise ValueError("Predictions must be 0/1 with one row per merchant")
        return p

    st.title("Merchant Sales Anomaly Monitor")
    st.caption("Big Data Analytics | Captured transactions only | Merchant-level screening and daily historical anomalies")
    st.markdown("**Investigation sequence:** Select merchant → inspect daily sales → inspect flagged dates → investigate business category")
    portfolio = st.selectbox("Portfolio", list(CONFIG), index=0)
    daily_name, category_name, prefix = CONFIG[portfolio]
    daily_path, category_path = ROOT / daily_name, ROOT / category_name
    if not daily_path.exists() or not category_path.exists():
        st.error("Prepared files for this portfolio are not available. Run the batch pipeline and add the resulting CSVs to outputs/.")
        st.stop()
    daily = read_prepared(str(daily_path)).copy()
    business = read_prepared(str(category_path)).copy()
    flagcol = "High_Confidence_Day" if "High_Confidence_Day" in daily else "High_Confidence_Anomaly_Day"
    if flagcol not in daily:
        st.error("The daily dataset does not contain a historical-anomaly-day indicator")
        st.stop()
    daily["Flagged_Day"] = daily[flagcol].astype(str).str.strip().str.lower().isin(["true", "1"])

    available = {}
    for name, suffix in MODELS.items():
        file = ROOT / f"predictions_{prefix}_{suffix}.csv"
        if file.exists():
            available[name] = file
    if not available:
        st.error("No valid model prediction files were found for this portfolio")
        st.stop()

    # A single dropdown contains the aggregate overview and every merchant.
    # The former Merchant filter is deliberately removed to avoid two competing
    # interpretations of "All merchants".
    merchants_in_data = set(daily["Merchant"].dropna().astype(str).unique())
    merchant_names = ['All merchants'] + sorted(merchants_in_data)
    selectors = st.columns([1.3, 2.3, 1.1])
    with selectors[0]:
        model = st.selectbox('Merchant model', list(available))
    try:
        predictions = read_predictions(str(available[model]))
    except Exception as exc:
        st.error(f"Prediction file validation failed: {exc}")
        st.stop()
    predictions = predictions[predictions.Merchant.isin(merchants_in_data)].copy()
    with selectors[1]:
        merchant = st.selectbox('Select merchant', merchant_names, index=0,
                                key=f'one_merchant_selector_{prefix}',
                                help='Choose All merchants to display combined portfolio sales, or select an individual merchant.')
    with selectors[2]:
        show_dates = st.checkbox('Flagged dates only', value=False, key=f'flagged_dates_{prefix}')

    all_merchants = merchant == 'All merchants'
    if all_merchants:
        m, portfolio_business = aggregate_portfolio_for_display(daily, business)
        m = m.sort_values('Date').copy()
        flagged = m.loc[m.Flagged_Day].copy()
        model_flag = None
    else:
        m = daily.loc[daily.Merchant.eq(merchant)].sort_values('Date').copy()
        flagged = m.loc[m.Flagged_Day].copy()
        model_match = predictions.loc[predictions.Merchant.eq(merchant),'Prediction']
        model_flag = int(model_match.iloc[0]) if not model_match.empty else None
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Merchants", f"{len(predictions):,}")
    k2.metric("Flagged by selected model", f"{int(predictions.Prediction.sum()):,}")
    k3.metric("Historical flagged merchant-days", f"{int(daily.Flagged_Day.sum()):,}")
    k4.metric("Captured sales (USD)", f"{daily.Daily_Sales.sum():,.2f}")
    if all_merchants:
        st.subheader(f'1. Daily captured sales — all {len(merchants_in_data):,} merchants')
        st.markdown(f'**Portfolio overview:** {len(predictions):,} merchants · '
                    f'**{int(predictions.Prediction.sum()):,}** flagged by {model} · '
                    f'**{int(daily.Flagged_Day.sum()):,}** merchant-day alerts '
                    f'on {len(flagged)} dates')
        st.caption('Portfolio totals are summed across merchants. The red markers show dates with '
                   'one or more individual merchant alerts; they do not imply a separately '
                   'validated portfolio-level anomaly.')
    else:
        st.subheader(f'1. Daily sales pattern — {merchant}')
        label = ('Anomalous' if model_flag else 'Normal') if model_flag is not None else 'Not classified'
        st.markdown(f'**{model} classification:** {label} · **Historical flagged days:** {len(flagged)}')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=m.Date,y=m.Daily_Sales,name="Actual daily sales",mode="lines+markers",line=dict(color="#1769aa",width=2),marker=dict(size=4),hovertemplate="%{x|%d %b %Y}<br>Actual USD %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=m.Date,y=m.Expected_Sales,name="Expected sales (past observations)",mode="lines",line=dict(color="#79b1e7",dash="dash"),hovertemplate="%{x|%d %b %Y}<br>Expected USD %{y:,.2f}<extra></extra>"))
    if len(flagged):
        if all_merchants:
            fig.add_trace(go.Scatter(x=flagged.Date,y=flagged.Daily_Sales,
                customdata=flagged.Flagged_Merchants,
                name='Dates with merchant alerts',mode='markers',
                marker=dict(color='#d83445',size=11,symbol='circle'),
                hovertemplate='%{x|%d %b %Y}<br>%{customdata} merchants flagged<br>Portfolio sales USD %{y:,.2f}<extra></extra>'))
        else:
            fig.add_trace(go.Scatter(x=flagged.Date,y=flagged.Daily_Sales,name="Flagged anomaly day",mode="markers",marker=dict(color="#d83445",size=12,symbol="circle"),hovertemplate="FLAGGED · %{x|%d %b %Y}<br>Sales USD %{y:,.2f}<extra></extra>"))
    fig.update_layout(height=390,margin=dict(l=10,r=12,t=10,b=10),xaxis_title="Date",yaxis_title="Captured sales (USD)",legend=dict(orientation="h",y=-0.25),hovermode="x unified")
    fig.update_xaxes(range=[m.Date.min(),m.Date.max()])
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Red markers indicate dates flagged by a past-only deviation rule. These are distinct from full-period model classifications. Early days may lack sufficient history.")

    st.subheader("2. Flagged dates and anomaly magnitude")
    # Default to the most severe flagged date, using magnitude of the historical
    # robust z-score where available; otherwise the relative deviation magnitude.
    def severity(frame):
        z = pd.to_numeric(frame.get("Robust_Z", pd.Series(np.nan, index=frame.index)), errors="coerce").abs()
        actual = pd.to_numeric(frame["Daily_Sales"], errors="coerce")
        expected = pd.to_numeric(frame["Expected_Sales"], errors="coerce")
        rel = ((actual - expected) / expected.where(expected.gt(0))).abs()
        return rel.fillna(z).fillna(-1)

    if len(flagged):
        if all_merchants:
            # Prioritise the date with the largest combined absolute deviations
            # of individually flagged merchants, not the portfolio's net change.
            flagged_source = daily.loc[daily.Flagged_Day].copy()
            flagged_source['_impact'] = (flagged_source.Daily_Sales - flagged_source.Expected_Sales).abs()
            impacts = flagged_source.groupby('Date')['_impact'].sum()
            ranked = flagged.assign(_severity=flagged.Date.map(impacts).fillna(0)).sort_values(
                ['_severity','Date'], ascending=[False, True])
        else:
            ranked = flagged.assign(_severity=severity(flagged)).sort_values(['_severity','Date'], ascending=[False, True])
        priority_date = pd.Timestamp(ranked.iloc[0]['Date'])
        flagged_options = flagged.Date.dt.strftime('%d %b %Y').tolist()
        date_options = flagged_options if show_dates else m.Date.dt.strftime('%d %b %Y').tolist()
    else:
        priority_date = pd.Timestamp(m.iloc[-1]["Date"])
        date_options = m["Date"].dt.strftime("%d %b %Y").tolist()
    if not date_options:
        st.info("No selectable dates")
        st.stop()

    # A unique key for each merchant prevents the date selected for a previous
    # merchant from leaking into the next merchant's investigation.
    selected_label = st.selectbox(
        ('Investigate portfolio date (largest combined merchant-alert impact selected automatically)'
         if all_merchants else 'Investigate a day (most significant flagged day selected automatically)'),
        date_options,
        index=date_options.index(priority_date.strftime("%d %b %Y")),
        key=f"investigate_{prefix}_{merchant}_{'flags' if show_dates else 'all'}",
    )
    selected_date = pd.to_datetime(selected_label, format="%d %b %Y")
    obs = m.loc[m.Date.eq(selected_date)].iloc[0]
    actual = float(obs.Daily_Sales)
    expected = float(obs.Expected_Sales) if pd.notna(obs.Expected_Sales) else np.nan
    gap = actual-expected if np.isfinite(expected) else np.nan
    pct = 100*gap/expected if np.isfinite(gap) and expected>0 else np.nan
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Actual sales", f"USD {actual:,.2f}")
    c2.metric("Expected sales", f"USD {expected:,.2f}" if np.isfinite(expected) else "N/A — insufficient history")
    c3.metric("Difference", f"USD {gap:+,.2f}" if np.isfinite(gap) else "N/A")
    c4.metric("Relative difference", f"{pct:+.2f}%" if np.isfinite(pct) else "N/A")
    if bool(obs.Flagged_Day):
        if all_merchants:
            st.markdown(f'**Date status:** 🔴 {int(obs.Flagged_Merchants)} merchants have historical alerts '
                        f'· **Date:** {selected_date:%d %b %Y}')
            st.caption('Portfolio difference is a net total; it is not an independently detected '
                       'portfolio anomaly, and merchant-level increases and decreases may offset.')
        else:
            direction = 'decrease' if np.isfinite(gap) and gap < 0 else 'increase'
            st.markdown(f'**Day status:** 🔴 Flagged historical anomaly · **Direction:** {direction} '
                        f'· **Date:** {selected_date:%d %b %Y}')
    else:
        st.write("**Day status:** No historical anomaly flag")
        if not np.isfinite(expected):
            st.info("Insufficient prior observations for a reliable historical expectation on this date. Select a flagged date for the anomaly investigation.")
    if len(flagged):
        if all_merchants:
            view = flagged[['Date','Flagged_Merchants','Daily_Sales','Expected_Sales']].copy()
        else:
            view = flagged[['Date','Daily_Sales','Expected_Sales','Robust_Z']].copy()
        view['Difference_USD'] = view.Daily_Sales-view.Expected_Sales
        view['Relative_%'] = np.where(view.Expected_Sales.gt(0),100*view.Difference_USD/view.Expected_Sales,np.nan)
        view['Date'] = view.Date.dt.strftime('%d %b %Y')
        st.dataframe(view.rename(columns={'Daily_Sales':'Actual USD', 'Expected_Sales':'Expected USD',
                        'Robust_Z':'Robust z', 'Flagged_Merchants':'Merchants with alerts',
                        'Difference_USD':'Difference USD', 'Relative_%':'Relative difference %'}),
                        hide_index=True,use_container_width=True)
        if all_merchants:
            selected_alerts = daily.loc[daily.Date.eq(selected_date) & daily.Flagged_Day,
                                        ['Merchant','Daily_Sales','Expected_Sales','Robust_Z']].copy()
            if not selected_alerts.empty:
                selected_alerts['Difference_USD'] = selected_alerts.Daily_Sales-selected_alerts.Expected_Sales
                selected_alerts = selected_alerts.sort_values('Difference_USD',key=lambda v:v.abs(),ascending=False)
                st.markdown('**Individual merchant alerts on the selected date**')
                st.dataframe(selected_alerts.rename(columns={'Daily_Sales':'Actual USD',
                    'Expected_Sales':'Expected USD','Robust_Z':'Robust z',
                    'Difference_USD':'Difference USD'}), hide_index=True,use_container_width=True)
    else:
        st.info("No individual days crossed the historical-deviation threshold for this merchant.")

    st.subheader("3. Business-category drill-down")
    category_source = portfolio_business if all_merchants else business
    cat=category_source.loc[category_source.Date.eq(selected_date) &
                            category_source.Merchant.eq(merchant)].copy()
    if cat.empty:
        st.info("Business-category records unavailable for the selected day.")
    else:
        st.caption(f"{'Combined portfolio categories' if all_merchants else 'Merchant categories'} "
                   f"for {selected_date:%d %b %Y} — the same date selected above.")
        cat["Difference_USD"]=cat.Business_Sales-cat.Expected_Business_Sales
        cat["Deviation_%"]=np.where(cat.Expected_Business_Sales>0,100*cat.Difference_USD/cat.Expected_Business_Sales,np.nan)
        cat=cat.sort_values("Difference_USD",key=lambda s:s.abs(),ascending=False)
        complete = cat.loc[cat["Difference_USD"].notna()]
        if not complete.empty:
            lead = complete.iloc[0]
            sign = "shortfall" if lead["Difference_USD"] < 0 else "increase"
            st.info(f"Largest estimated category {sign}: {lead['Business']} (USD {abs(lead['Difference_USD']):,.2f}). This is a contribution for investigation, not an established cause.")
        else:
            st.warning("Category expectations are unavailable for this date due to insufficient historical data; actual sales are shown without inferred contributions.")
        fig2=go.Figure()
        fig2.add_trace(go.Bar(y=cat.Business,x=cat.Business_Sales,orientation="h",name="Actual sales",marker_color="#1f6fba"))
        fig2.add_trace(go.Bar(y=cat.Business,x=cat.Expected_Business_Sales,orientation="h",name="Expected sales",marker_color="#8ec1e9"))
        fig2.update_layout(barmode="group",height=max(260,55*len(cat)+120),margin=dict(l=10,r=15,t=15,b=15),xaxis_title="USD",yaxis_title="",yaxis=dict(autorange="reversed"),legend=dict(orientation="h",y=-0.25))
        st.plotly_chart(fig2,use_container_width=True)
        table = cat[["Business","Business_Sales","Expected_Business_Sales","Difference_USD","Deviation_%","Business_Transactions"]].rename(columns={"Business":"Category","Business_Sales":"Actual USD","Expected_Business_Sales":"Expected USD","Difference_USD":"Difference USD","Deviation_%":"Deviation %","Business_Transactions":"Captured transactions"})
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={"Actual USD":st.column_config.NumberColumn(format="%.2f"),"Expected USD":st.column_config.NumberColumn(format="%.2f"),"Difference USD":st.column_config.NumberColumn(format="%+.2f"),"Deviation %":st.column_config.NumberColumn(format="%+.2f%%")})
        st.caption('Category expectations are estimated separately and may not sum to the '
                   'overall expected sales. For All merchants, incomplete historical baselines '
                   'are withheld rather than displayed as zero. Category contributions are '
                   'investigative signals, not proven causes.')

    st.subheader("4. Transaction summary and downloadable results")
    total_transactions=int(m.Transaction_Count.sum())
    average_ticket=float(m.Daily_Sales.sum()/total_transactions) if total_transactions else 0
    z1,z2,z3=st.columns(3)
    z1.metric('Portfolio captured transactions' if all_merchants else 'Merchant captured transactions',f'{total_transactions:,}')
    z2.metric('Portfolio captured sales' if all_merchants else 'Merchant captured sales',f'USD {m.Daily_Sales.sum():,.2f}')
    z3.metric("Average captured transaction",f"USD {average_ticket:,.2f}")
    left,right=st.columns(2)
    with left:
        export_name = f'{prefix}_portfolio_daily.csv' if all_merchants else f'{prefix}_merchant_daily.csv'
        st.download_button('Download daily data — selected view',m.to_csv(index=False).encode(),
                           file_name=export_name,mime='text/csv')
    with right:
        st.download_button("Download selected model predictions",predictions[["Merchant","Prediction"]].to_csv(index=False).encode(),file_name=f"predictions_{prefix}_{MODELS[model]}.csv",mime="text/csv")
    with st.expander("Portfolio model comparison and evaluation notes"):
        st.write("Available prediction files:",", ".join(available))
        st.write("Original supervised predictions are not independently evaluated. Recorded external evaluator results should only be associated with the exact submissions that produced them; model outputs from a new run must be verified before such attribution.")
        counts=[]
        for name,path in available.items():
            p=read_predictions(str(path))
            counts.append({"Model":name,"Merchants":len(p),"Flagged":int(p.Prediction.sum())})
        st.dataframe(pd.DataFrame(counts),hide_index=True,use_container_width=True)

    render_transaction_and_upload_sections(st, ROOT, daily, business, merchant, prefix)


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description='Single-file merchant anomaly monitor')
    p.add_argument('--mode',choices=['batch','app'],default='app')
    p.add_argument('--data-dir',default='raw_data')
    p.add_argument('--output-dir',default='outputs')
    args=p.parse_args(argv)
    if args.mode=='batch': run_batch(Path(args.data_dir),Path(args.output_dir))
    else: render_streamlit_app(Path(args.output_dir))

if __name__=='__main__':
    import sys
    # Streamlit runs without CLI arguments; batch uses explicit --mode batch.
    main(sys.argv[1:])
