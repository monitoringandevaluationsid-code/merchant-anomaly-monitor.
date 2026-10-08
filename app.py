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




def render_streamlit_app(output_dir: Path) -> None:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.set_page_config(page_title="Merchant Sales Anomaly Monitor", page_icon="📈", layout="wide")
    st.title("Merchant Sales Anomaly Monitor")
    st.caption("Big Data Analytics | Captured sales only | Original and new merchant portfolios")
    out = Path(__file__).resolve().parent / "outputs"
    if not out.exists(): out = output_dir

    models = {"Isolation Forest": "isolation_forest", "Random Forest": "random_forest",
              "Gradient Boosting": "gradient_boosting", "Logistic Regression": "logistic_regression"}
    portfolios = {"Original portfolio": ("original", "daily_sales_prepared.csv", "business_daily_prepared.csv"),
                  "New portfolio": ("new", "daily_sales_new_prepared.csv", "business_daily_new_prepared.csv")}

    @st.cache_data(show_spinner=False)
    def read_csv(path):
        return pd.read_csv(path)

    @st.cache_data(show_spinner=False)
    def read_transactions(path):
        return pd.read_csv(path, compression="gzip", low_memory=False)

    def prepared_info(prefix, daily_fn, business_fn):
        return (out / daily_fn).is_file() and (out / business_fn).is_file()

    available_portfolios = [n for n,(prefix,d,b) in portfolios.items() if prepared_info(prefix,d,b)]
    if not available_portfolios:
        st.error("No prepared datasets found. Run batch processing locally and upload the outputs/ CSVs.")
        st.stop()

    tab_monitor, tab_compare, tab_transactions, tab_methods = st.tabs([
        "Merchant monitoring", "Original vs New / Model evaluator", "Transactions", "Methodology & data status"])

    with tab_monitor:
        c1,c2,c3,c4 = st.columns([1.2,1.3,1.1,1.6])
        with c1: portfolio = st.selectbox("Dataset", available_portfolios, key="portfolio")
        prefix,daily_file,business_file=portfolios[portfolio]
        options={name:short for name,short in models.items() if (out/f"predictions_{prefix}_{short}.csv").exists()}
        if not options:
            st.error("This portfolio has no prediction file."); st.stop()
        with c2: model=st.selectbox("Merchant model",list(options),key="model")
        daily=read_csv(out/daily_file).copy()
        categories=read_csv(out/business_file).copy()
        preds=read_csv(out/f"predictions_{prefix}_{options[model]}.csv").copy()
        for df in (daily,categories):
            df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        flag="High_Confidence_Day"
        if flag not in daily and "High_Confidence_Anomaly_Day" in daily:
            daily=daily.rename(columns={"High_Confidence_Anomaly_Day":flag})
        if flag not in daily: st.error("Missing daily historical flag column"); st.stop()
        daily[flag]=daily[flag].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False)
        if preds["Merchant"].duplicated().any() or not preds["Prediction"].isin([0,1]).all():
            st.error("Invalid prediction file: duplicate names or nonbinary values"); st.stop()
        if set(preds.Merchant)!=set(daily.Merchant):
            st.error("Prediction names do not match the prepared daily dataset"); st.stop()
        preds=preds.sort_values("Merchant").reset_index(drop=True)
        with c3: merchant_filter=st.selectbox("Merchant filter",["All merchants","Flagged only"])
        merchants=preds.loc[preds.Prediction.eq(1),"Merchant"].tolist() if merchant_filter=="Flagged only" else preds.Merchant.tolist()
        if not merchants: st.info("No merchants satisfy this filter"); st.stop()
        with c4: merchant=st.selectbox("Merchant",merchants)
        if len(options)<4:
            st.info(f"{portfolio}: {len(options)} model(s) available. Additional models require their prediction CSVs; predictions are not fabricated.")
        if prefix=="original" and model!="Isolation Forest":
            st.warning("Original supervised predictions are in-sample fits to Isolation Forest pseudo-labels; these are not independently evaluated results.")
        k1,k2,k3,k4=st.columns(4)
        k1.metric("Merchants",f"{len(preds):,}")
        k2.metric("Flagged by selected model",f"{int(preds.Prediction.sum()):,}")
        k3.metric("Historical flagged days",f"{int(daily[flag].sum()):,}")
        k4.metric("Captured sales (USD)",f"{daily.Daily_Sales.sum():,.2f}")
        classification=int(preds.loc[preds.Merchant.eq(merchant),"Prediction"].iloc[0])
        md=daily.loc[daily.Merchant.eq(merchant)].sort_values("Date")
        marked=md.loc[md[flag]]
        st.subheader(merchant)
        st.write(f"**{model} classification:** {'Anomalous' if classification else 'Normal'}  |  **Historical flagged days:** {len(marked)}")
        if prefix=="original" and model=="Isolation Forest" and (out/"merchant_risk.csv").exists():
            risk=read_csv(out/"merchant_risk.csv")
            if "Anomaly_Score" in risk:
                rr=risk.loc[risk.Merchant.eq(merchant),"Anomaly_Score"]
                if not rr.empty: st.caption(f"Isolation Forest anomaly score: {float(rr.iloc[0]):.4f} (higher = more unusual)")
        fig=go.Figure()
        fig.add_scatter(x=md.Date,y=md.Daily_Sales,name="Actual daily sales",mode="lines+markers",line_color="#2266aa")
        fig.add_scatter(x=md.Date,y=md.Expected_Sales,name="Expected sales",mode="lines",line=dict(color="#7ba8d5",dash="dash"))
        fig.add_scatter(x=marked.Date,y=marked.Daily_Sales,name="Flagged day",mode="markers",marker=dict(color="#d5363d",size=11))
        fig.update_layout(height=420,yaxis_title="Sales (USD)",xaxis_title="Date",legend_orientation="h",margin=dict(t=25,b=30))
        fig.update_xaxes(range=[md.Date.min(),md.Date.max()],tickformat="%d %b %Y")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Merchant classification covers the entire period. Red day markers come from a separate past-only deviation rule.")
        st.subheader("Business-category investigation")
        all_dates=sorted(md.Date.dt.date.unique())
        preferred=marked.Date.max().date() if len(marked) else all_dates[-1]
        day=st.selectbox("Investigate date",all_dates,index=all_dates.index(preferred))
        bd=categories.loc[categories.Merchant.eq(merchant)&categories.Date.eq(pd.Timestamp(day))].copy()
        if bd.empty: st.info("No business-category rows for this selection")
        else:
            bd["Difference USD"]=bd.Business_Sales-bd.Expected_Business_Sales
            bd["Abs difference"]=bd["Difference USD"].abs()
            bd=bd.sort_values("Abs difference",ascending=False)
            shown=bd[["Business","Business_Sales","Expected_Business_Sales","Difference USD"]].rename(columns={"Business":"Category","Business_Sales":"Actual sales USD","Expected_Business_Sales":"Expected sales USD"})
            st.dataframe(shown,use_container_width=True,hide_index=True)
            long=shown.melt(id_vars="Category",value_vars=["Actual sales USD","Expected sales USD"],var_name="Series",value_name="USD")
            catfig=px.bar(long,y="Category",x="USD",color="Series",barmode="group",orientation="h")
            catfig.update_layout(height=360,yaxis=dict(categoryorder="array",categoryarray=shown.Category.iloc[::-1].tolist()),margin=dict(t=15,b=10))
            st.plotly_chart(catfig,use_container_width=True)
            st.caption("Category baselines are separately estimated and need not sum to the merchant expected total; deviations are investigatory signals, not proven causes.")
        st.subheader("Merchant classifications and CSV export")
        show=preds.copy(); show["Classification"]=show.Prediction.map({0:"Normal",1:"Anomalous"})
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.download_button("Download model predictions",preds[["Merchant","Prediction"]].to_csv(index=False),file_name=f"predictions_{prefix}_{options[model]}.csv",mime="text/csv")

    with tab_compare:
        st.subheader("Portfolio availability and model comparison")
        summary=[]
        for n,(pf,d,b) in portfolios.items():
            if not prepared_info(pf,d,b):
                summary.append({"Portfolio":n,"Prepared sales":"Missing","Merchants":None,"Models ready":0})
                continue
            dat=read_csv(out/d)
            ready_models=sum((out/f"predictions_{pf}_{short}.csv").exists() for short in models.values())
            summary.append({"Portfolio":n,"Prepared sales":"Available","Merchants":dat.Merchant.nunique(),"Models ready":ready_models})
        st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)
        st.subheader("Recorded external Model Evaluator results")
        st.caption("Historical evaluator observations entered from user-provided screenshots. These results must be matched to the exact uploaded prediction files before being attributed to current outputs.")
        eval_df=pd.DataFrame([
            ["Original","Isolation Forest",94.3,90.9,66.7,76.9,98.9,82.8],
            ["New","Isolation Forest",98.1,100,86.7,92.9,100,93.4],
            ["New","Random Forest",98.1,100,86.7,92.9,100,93.4],
            ["New","Gradient Boosting",96.2,92.3,80.0,85.7,98.9,89.5],
            ["New","Logistic Regression",100,100,100,100,100,100]],columns=["Dataset","Model","Accuracy %","Precision %","Recall %","F1 %","Specificity %","Balanced accuracy %"])
        st.dataframe(eval_df,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(eval_df.loc[eval_df.Dataset.eq("New")],x="Model",y="F1 %",title="Recorded new-portfolio F1 scores",range_y=[0,105]),use_container_width=True)
        st.warning("Evaluator scores are not automatically recalculated from prediction CSVs: the evaluator hides its ground-truth labels. Previously uploaded versions showed conflicting anomalous counts. Reconcile submissions before making accuracy claims about current predictions.")
        st.subheader("Available model prediction counts")
        counts=[]
        for pf in ("original","new"):
            for label,short in models.items():
                path=out/f"predictions_{pf}_{short}.csv"
                if path.exists():
                    f=read_csv(path);counts.append({"Dataset":pf.title(),"Model":label,"Merchants":len(f),"Flagged":int(f.Prediction.sum())})
        if counts: st.dataframe(pd.DataFrame(counts),use_container_width=True,hide_index=True)

    with tab_transactions:
        st.subheader("Transaction exploration")
        st.caption("Transactions are filtered to Captured status. Add compressed files in outputs/ or upload a source CSV/CSV.GZ below. Prepared daily aggregates remain available even without raw transactions.")
        pchoice=st.selectbox("Transaction portfolio",["Original","New"],key="transactions_portfolio")
        path=out/f"transactions_{pchoice.lower()}_captured.csv.gz"
        upload=st.file_uploader("Optional: upload transaction file (.csv or .csv.gz) for this session",type=["csv","gz"])
        transaction_df=None
        if upload:
            try:
                transaction_df=pd.read_csv(upload,compression="gzip" if upload.name.endswith(".gz") else None,low_memory=False)
                if "Status" in transaction_df.columns:
                    transaction_df=transaction_df.loc[transaction_df.Status.astype(str).str.casefold().eq("captured")]
                elif "Status_Code" in transaction_df.columns:
                    st.warning("Status_Code supplied without status lookup. Upload a Captured-only extract or include a resolved Status column; transactions have not been filtered.")
                    transaction_df=None
                else:
                    st.warning("Uploaded file has no Status field. Upload a Captured-only extract generated by batch mode.")
                    transaction_df=None
            except Exception as exc: st.error(f"Could not read uploaded data: {exc}")
        elif path.exists():
            try: transaction_df=read_transactions(path)
            except Exception as exc: st.error(f"Could not read compressed transactions: {exc}")
        if transaction_df is None:
            st.info("Transaction-level file not present. Batch mode can generate compressed Captured transaction files. The dashboard still provides daily aggregation and category analysis.")
            pf="original" if pchoice=="Original" else "new"
            dn="daily_sales_prepared.csv" if pf=="original" else "daily_sales_new_prepared.csv"
            if (out/dn).exists():
                dd=read_csv(out/dn)
                st.metric("Prepared merchant-day rows",f"{len(dd):,}")
                st.dataframe(dd[[x for x in ["Merchant","Date","Daily_Sales","Transaction_Count"] if x in dd]].head(150),use_container_width=True,hide_index=True)
        else:
            st.metric("Captured transaction rows",f"{len(transaction_df):,}")
            tdf=transaction_df
            if "Merchant" in tdf:
                selected_tx=st.selectbox("Filter transaction merchant",["All merchants"]+sorted(tdf.Merchant.dropna().astype(str).unique().tolist()))
                if selected_tx!="All merchants":tdf=tdf.loc[tdf.Merchant.eq(selected_tx)]
            if "Date" in tdf:
                dt=pd.to_datetime(tdf.Date,errors="coerce")
                if dt.notna().any():
                    first,last=dt.min().date(),dt.max().date()
                    period=st.date_input("Transaction date range",(first,last),min_value=first,max_value=last)
                    if isinstance(period,(tuple,list)) and len(period)==2:
                        tdf=tdf.loc[dt.dt.date.between(period[0],period[1])]
            st.caption(f"Showing first 500 of {len(tdf):,} matching Captured transactions")
            st.dataframe(tdf.head(500),use_container_width=True,hide_index=True)
            st.download_button("Download filtered transactions CSV",tdf.to_csv(index=False),file_name=f"{pchoice.lower()}_captured_filtered.csv",mime="text/csv")

    with tab_methods:
        st.subheader("Method and interpretation")
        st.markdown("**Sales:** only transactions with resolved status `Captured`. **Unit of model prediction:** one binary classification per merchant (105 per portfolio). **Daily anomaly markers:** historical baseline and robust deviation rule, distinct from merchant-level model output. **Business-category differences:** diagnostic indicators, not established causes.")
        st.markdown("**Training:** Isolation Forest is fitted to original merchant features; supervised models are trained to reproduce its pseudo-labels. The unchanged fitted models are applied to new merchant features. Original supervised predictions are in-sample and should not be equated with independently validated performance.")
        st.markdown("**Recorded evaluation:** screenshot scores are supplied for reference only. External hidden labels are not available; a match to the exact evaluated CSV version is needed.")
        st.markdown("**Deployment:** raw CSVs are not needed on Streamlit Cloud when prepared outputs are available. Optional compressed Captured transaction extracts can be added to `outputs/`; avoid uploading any confidential production banking data to public GitHub.")

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
