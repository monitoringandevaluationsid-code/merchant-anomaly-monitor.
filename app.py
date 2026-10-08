"""Single-page merchant sales anomaly dashboard for the Big Data Analytics assignment.
Run: streamlit run app.py
Prepared outputs are produced by the separately retained batch pipeline.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Merchant Sales Anomaly Monitor", page_icon="📊", layout="wide")
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

filters = st.columns([1.15, 1.05, 1, 1.65])
with filters[0]:
    model = st.selectbox("Merchant model", list(available))
with filters[1]:
    status = st.selectbox("Merchant filter", ["All merchants", "Flagged only", "Normal only"])
try:
    predictions = read_predictions(str(available[model]))
except Exception as exc:
    st.error(f"Prediction file validation failed: {exc}")
    st.stop()
merchants_in_data = set(daily["Merchant"].dropna().unique())
predictions = predictions[predictions["Merchant"].isin(merchants_in_data)].copy()
if status == "Flagged only":
    merchant_names = predictions.loc[predictions.Prediction.eq(1), "Merchant"].sort_values().tolist()
elif status == "Normal only":
    merchant_names = predictions.loc[predictions.Prediction.eq(0), "Merchant"].sort_values().tolist()
else:
    merchant_names = sorted(predictions["Merchant"].unique().tolist())
if not merchant_names:
    st.warning("No merchants match the selected model and filter.")
    st.stop()
with filters[2]:
    show_dates = st.checkbox("Flagged days only", value=False)
with filters[3]:
    merchant = st.selectbox("Select merchant", merchant_names, index=merchant_names.index("Rachel Sheppard") if "Rachel Sheppard" in merchant_names else 0)

m = daily[daily.Merchant.eq(merchant)].sort_values("Date").copy()
flagged = m[m.Flagged_Day].copy()
model_flag = int(predictions.set_index("Merchant").loc[merchant, "Prediction"])
k1,k2,k3,k4=st.columns(4)
k1.metric("Merchants", f"{len(predictions):,}")
k2.metric("Flagged by selected model", f"{int(predictions.Prediction.sum()):,}")
k3.metric("Historical flagged merchant-days", f"{int(daily.Flagged_Day.sum()):,}")
k4.metric("Captured sales (USD)", f"{daily.Daily_Sales.sum():,.2f}")
st.subheader(f"1. Daily sales pattern — {merchant}")
st.markdown(f"**{model} classification:** {'Anomalous' if model_flag else 'Normal'} · **Historical flagged days:** {len(flagged)}")
fig=go.Figure()
fig.add_trace(go.Scatter(x=m.Date,y=m.Daily_Sales,name="Actual daily sales",mode="lines+markers",line=dict(color="#1769aa",width=2),marker=dict(size=4),hovertemplate="%{x|%d %b %Y}<br>Actual USD %{y:,.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=m.Date,y=m.Expected_Sales,name="Expected sales (past observations)",mode="lines",line=dict(color="#79b1e7",dash="dash"),hovertemplate="%{x|%d %b %Y}<br>Expected USD %{y:,.2f}<extra></extra>"))
if len(flagged):
    fig.add_trace(go.Scatter(x=flagged.Date,y=flagged.Daily_Sales,name="Flagged anomaly day",mode="markers",marker=dict(color="#d83445",size=12,symbol="circle"),hovertemplate="FLAGGED · %{x|%d %b %Y}<br>Sales USD %{y:,.2f}<extra></extra>"))
fig.update_layout(height=390,margin=dict(l=10,r=12,t=10,b=10),xaxis_title="Date",yaxis_title="Captured sales (USD)",legend=dict(orientation="h",y=-0.25),hovermode="x unified")
fig.update_xaxes(range=[m.Date.min(),m.Date.max()])
st.plotly_chart(fig,use_container_width=True)
st.caption("Red markers indicate dates flagged by a past-only deviation rule. These are distinct from full-period model classifications. Early days may lack sufficient history.")

st.subheader("2. Flagged dates and anomaly magnitude")
if len(flagged):
    candidates=flagged["Date"].dt.strftime("%d %b %Y").tolist()
    date_options=candidates if show_dates else m["Date"].dt.strftime("%d %b %Y").tolist()
else:
    date_options=m["Date"].dt.strftime("%d %b %Y").tolist()
if not date_options:
    st.info("No selectable dates")
    st.stop()
default_idx=date_options.index(candidates[-1]) if len(flagged) and candidates[-1] in date_options else len(date_options)-1
selected_label=st.selectbox("Investigate a day (flagged days are prioritised)",date_options,index=default_idx)
selected_date=pd.to_datetime(selected_label,format="%d %b %Y")
obs=m.loc[m.Date.eq(selected_date)].iloc[0]
actual=float(obs.Daily_Sales)
expected=float(obs.Expected_Sales) if pd.notna(obs.Expected_Sales) else np.nan
gap=actual-expected if np.isfinite(expected) else np.nan
pct=100*gap/expected if np.isfinite(gap) and expected>0 else np.nan
c1,c2,c3,c4=st.columns(4)
c1.metric("Actual sales",f"USD {actual:,.2f}")
c2.metric("Expected sales",f"USD {expected:,.2f}" if np.isfinite(expected) else "Insufficient history")
c3.metric("Difference",f"USD {gap:+,.2f}" if np.isfinite(gap) else "N/A")
c4.metric("Relative difference",f"{pct:+.1f}%" if np.isfinite(pct) else "N/A")
st.write("**Day status:**", "🔴 Historical anomaly flag" if bool(obs.Flagged_Day) else "No historical anomaly flag")
if len(flagged):
    view=flagged[["Date","Daily_Sales","Expected_Sales","Robust_Z"]].copy()
    view["Difference_USD"]=view.Daily_Sales-view.Expected_Sales
    view["Date"]=view.Date.dt.strftime("%d %b %Y")
    st.dataframe(view.rename(columns={"Daily_Sales":"Actual USD","Expected_Sales":"Expected USD","Robust_Z":"Robust z","Difference_USD":"Difference USD"}),hide_index=True,use_container_width=True)
else:
    st.info("No individual days crossed the historical-deviation threshold for this merchant.")

st.subheader("3. Business-category drill-down")
cat=business[(business.Merchant.eq(merchant)) & (business.Date.eq(selected_date))].copy()
if cat.empty:
    st.info("Business-category records unavailable for the selected day.")
else:
    cat["Difference_USD"]=cat.Business_Sales-cat.Expected_Business_Sales
    cat["Deviation_%"]=np.where(cat.Expected_Business_Sales>0,100*cat.Difference_USD/cat.Expected_Business_Sales,np.nan)
    cat=cat.sort_values("Difference_USD",key=lambda s:s.abs(),ascending=False)
    fig2=go.Figure()
    fig2.add_trace(go.Bar(y=cat.Business,x=cat.Business_Sales,orientation="h",name="Actual sales",marker_color="#1f6fba"))
    fig2.add_trace(go.Bar(y=cat.Business,x=cat.Expected_Business_Sales,orientation="h",name="Expected sales",marker_color="#8ec1e9"))
    fig2.update_layout(barmode="group",height=max(260,55*len(cat)+120),margin=dict(l=10,r=15,t=15,b=15),xaxis_title="USD",yaxis_title="",yaxis=dict(autorange="reversed"),legend=dict(orientation="h",y=-0.25))
    st.plotly_chart(fig2,use_container_width=True)
    st.dataframe(cat[["Business","Business_Sales","Expected_Business_Sales","Difference_USD","Deviation_%","Business_Transactions"]].rename(columns={"Business":"Category","Business_Sales":"Actual USD","Expected_Business_Sales":"Expected USD","Difference_USD":"Difference USD","Deviation_%":"Deviation %","Business_Transactions":"Captured transactions"}),hide_index=True,use_container_width=True)
    st.caption("Category expectations are estimated separately and may not sum to the merchant-level expected sales. Contributions are investigative signals, not proven causes.")

st.subheader("4. Transaction summary and downloadable results")
total_transactions=int(m.Transaction_Count.sum())
average_ticket=float(m.Daily_Sales.sum()/total_transactions) if total_transactions else 0
z1,z2,z3=st.columns(3)
z1.metric("Merchant captured transactions",f"{total_transactions:,}")
z2.metric("Merchant captured sales",f"USD {m.Daily_Sales.sum():,.2f}")
z3.metric("Average captured transaction",f"USD {average_ticket:,.2f}")
left,right=st.columns(2)
with left:
    st.download_button("Download selected merchant daily data",m.to_csv(index=False).encode(),file_name=f"{prefix}_merchant_daily.csv",mime="text/csv")
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
