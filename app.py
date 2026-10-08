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




def render_streamlit_app(output_dir: Path) -> None:
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="Merchant Sales Anomaly Monitor", page_icon="📈", layout="wide")
    st.title("Merchant Sales Anomaly Monitor")
    st.caption("Captured transactions only | Merchant-level classifications and separate historical daily-deviation markers")
    st.caption("Select a dataset and model. Only models with available, validated prediction files are selectable.")

    base = Path(__file__).resolve().parent
    out = base / 'outputs'
    if not out.exists() and output_dir.exists():
        out = output_dir
    portfolios = {
        'Original portfolio': ('original', 'daily_sales_prepared.csv', 'business_daily_prepared.csv'),
        'New portfolio': ('new', 'daily_sales_new_prepared.csv', 'business_daily_new_prepared.csv'),
    }
    model_names = {'Isolation Forest': 'isolation_forest', 'Random Forest': 'random_forest',
                   'Gradient Boosting': 'gradient_boosting', 'Logistic Regression': 'logistic_regression'}
    ready = {name: params for name,params in portfolios.items() if (out / params[1]).is_file() and (out / params[2]).is_file()}
    if not ready:
        st.error(f"No prepared portfolio data in {out}. Generate outputs locally in batch mode and commit the outputs directory to GitHub.")
        st.stop()
    c1,c2,c3,c4 = st.columns([1.1,1.3,1.15,1.7])
    with c1: dataset = st.selectbox('Dataset', list(ready))
    prefix,daily_name,business_name = ready[dataset]
    available = {k:v for k,v in model_names.items() if (out / f'predictions_{prefix}_{v}.csv').is_file()}
    if not available and prefix == 'original' and (out/'merchant_risk.csv').exists():
        available={'Isolation Forest':'isolation_forest'}
    if not available:
        st.error('No prediction CSVs for selected portfolio. Generate the model outputs locally.'); st.stop()
    with c2: chosen = st.selectbox('Merchant model', list(available))
    if len(available) < 4:
        st.info(f"{dataset} currently has {len(available)} available model(s). The other models will appear once their prediction CSVs are provided.")
    if prefix == 'original' and chosen != 'Isolation Forest':
        st.warning('Original-portfolio supervised predictions are IN-SAMPLE fits to Isolation Forest pseudo-labels, not independent evaluation results.')

    @st.cache_data(show_spinner=False)
    def load_csv(path):
        return pd.read_csv(path)
    daily = load_csv(out/daily_name).copy()
    categories = load_csv(out/business_name).copy()
    pred_path=out/f'predictions_{prefix}_{available[chosen]}.csv'
    if pred_path.is_file():
        pred=load_csv(pred_path).copy()
    else:
        pred=load_csv(out/'merchant_risk.csv')[['Merchant','Prediction']].copy()
    for name, df in [('Daily',daily),('Business categories',categories)]:
        if 'Date' not in df or 'Merchant' not in df:
            st.error(f'{name} data missing Date or Merchant'); st.stop()
        df['Date']=pd.to_datetime(df['Date'],errors='coerce')
        if df['Date'].isna().any(): st.error(f'{name} contains invalid dates'); st.stop()
    flag='High_Confidence_Day'
    if flag not in daily.columns and 'High_Confidence_Anomaly_Day' in daily.columns:
        daily=daily.rename(columns={'High_Confidence_Anomaly_Day':flag})
    if flag not in daily.columns: st.error('Missing anomaly-day flag column'); st.stop()
    mapped=daily[flag].astype(str).str.lower().str.strip().map({'true':True,'false':False,'1':True,'0':False})
    if mapped.isna().any(): st.error('Invalid anomaly-day flag values'); st.stop()
    daily[flag]=mapped.astype(bool)
    if not {'Merchant','Prediction'}.issubset(pred): st.error('Prediction file missing required columns'); st.stop()
    if pred['Merchant'].isna().any() or pred['Merchant'].duplicated().any() or not pred['Prediction'].isin([0,1]).all():
        st.error('Prediction file contains duplicate names, missing values or non-binary predictions'); st.stop()
    names=set(daily['Merchant'].dropna().unique())
    if names!=set(pred['Merchant']): st.error('Merchant names in predictions and daily data do not match'); st.stop()
    pred=pred.sort_values('Merchant').reset_index(drop=True)
    with c3: filt=st.selectbox('Merchant filter',['All merchants','Flagged only'])
    merchants=pred.loc[pred['Prediction'].eq(1),'Merchant'].tolist() if filt=='Flagged only' else pred['Merchant'].tolist()
    if not merchants: st.info('No flagged merchants for selected model'); st.stop()
    with c4: selected=st.selectbox('Merchant',merchants)
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Merchants',f"{len(pred):,}")
    k2.metric('Flagged by selected model',f"{int(pred['Prediction'].sum()):,}")
    k3.metric('Historical flagged days',f"{int(daily[flag].sum()):,}")
    k4.metric('Captured sales (USD)',f"{daily['Daily_Sales'].sum():,.2f}")
    row=pred.loc[pred['Merchant'].eq(selected)].iloc[0]
    md=daily.loc[daily['Merchant'].eq(selected)].sort_values('Date').copy()
    days=md.loc[md[flag]]
    st.subheader(selected)
    st.markdown(f"**{chosen} classification:** {'Anomalous' if int(row['Prediction']) else 'Normal'} · **Historical deviation days:** {len(days)}")
    if chosen=='Isolation Forest' and prefix=='original' and (out/'merchant_risk.csv').is_file():
        score_df=load_csv(out/'merchant_risk.csv')
        if 'Anomaly_Score' in score_df:
            match=score_df.loc[score_df['Merchant'].eq(selected),'Anomaly_Score']
            if not match.empty: st.caption(f"Original Isolation Forest anomaly score: {float(match.iloc[0]):.4f} (larger = more unusual)")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=md['Date'],y=md['Daily_Sales'],mode='lines+markers',name='Actual daily sales',line={'color':'#2166ac'}))
    fig.add_trace(go.Scatter(x=md['Date'],y=md['Expected_Sales'],mode='lines',name='Historical expectation',line={'dash':'dash','color':'#727a86'}))
    fig.add_trace(go.Scatter(x=days['Date'],y=days['Daily_Sales'],mode='markers',name='Unusual day',marker={'color':'#cb2027','size':10}))
    fig.update_layout(height=420,xaxis_title='Date',yaxis_title='Sales (USD)',margin={'l':10,'r':10,'t':15,'b':15})
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Day markers use the past-only historical-deviation rule and do not change with the selected merchant-level model.')
    st.subheader('Business-category investigation')
    dates=md['Date'].dt.date.tolist()
    default=days['Date'].max().date() if len(days) else dates[-1]
    date=st.selectbox('Investigate date',dates,index=dates.index(default))
    bd=categories.loc[categories['Merchant'].eq(selected) & categories['Date'].eq(pd.Timestamp(date))].copy()
    if bd.empty: st.info('No category data for this date')
    else:
        bd['Gap_USD']=bd['Business_Sales']-bd['Expected_Business_Sales']
        bd['Absolute_Gap_USD']=bd['Gap_USD'].abs()
        bd=bd.sort_values('Absolute_Gap_USD',ascending=False)
        cols=[x for x in ['Business','Business_Sales','Expected_Business_Sales','Gap_USD','Absolute_Gap_USD'] if x in bd]
        st.dataframe(bd[cols],use_container_width=True,hide_index=True)
        st.bar_chart(bd.set_index('Business')[['Business_Sales','Expected_Business_Sales']])
        st.caption('Category expectations are calculated separately and may not sum to the merchant-level expected total.')
    st.subheader('Classification table and export')
    st.dataframe(pred,use_container_width=True,hide_index=True)
    st.download_button('Download selected model predictions',pred[['Merchant','Prediction']].to_csv(index=False).encode(),file_name=f'predictions_{prefix}_{available[chosen]}.csv',mime='text/csv')
    st.caption('For academic submission, compare regenerated predictions with the exact files uploaded to the external Model Evaluator before replacing them.')


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
