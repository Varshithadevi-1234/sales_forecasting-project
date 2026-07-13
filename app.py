"""
Sales Forecasting & Demand Intelligence Dashboard
Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud by pointing it at this file in your GitHub repo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


@st.cache_data
def monthly_series(df, category=None, region=None):
    subset = df.copy()
    if category and category != "All":
        subset = subset[subset["Category"] == category]
    if region and region != "All":
        subset = subset[subset["Region"] == region]
    s = subset.set_index("Order Date").resample("MS")["Sales"].sum()
    return s.asfreq("MS").fillna(0)


@st.cache_data
def weekly_series(df):
    return df.set_index("Order Date").resample("W")["Sales"].sum().reset_index()


@st.cache_resource
def fit_sarima(series):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def sarima_forecast(train, steps):
    fit = fit_sarima(train)
    return fit.get_forecast(steps=steps).predicted_mean


def prophet_forecast(train, steps):
    pdf = train.reset_index()
    pdf.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pdf)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(future).set_index("ds")["yhat"]
    return fc.iloc[-steps:]


def xgboost_forecast(train, steps):
    sdf = train.reset_index()
    sdf.columns = ["Order Date", "Sales"]
    sdf["Month_num"] = sdf["Order Date"].dt.month
    sdf["Quarter"] = sdf["Order Date"].dt.quarter
    sdf["lag_1"] = sdf["Sales"].shift(1)
    sdf["lag_2"] = sdf["Sales"].shift(2)
    sdf["lag_3"] = sdf["Sales"].shift(3)
    sdf["rolling_mean_3"] = sdf["Sales"].shift(1).rolling(3).mean()
    sdf = sdf.dropna().reset_index(drop=True)
    feat_cols = ["Month_num", "Quarter", "lag_1", "lag_2", "lag_3", "rolling_mean_3"]

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(sdf[feat_cols], sdf["Sales"])

    history = sdf[["Order Date", "Sales"]].copy()
    preds = []
    for _ in range(steps):
        last3 = history["Sales"].iloc[-3:].values
        next_date = history["Order Date"].iloc[-1] + pd.DateOffset(months=1)
        row = pd.DataFrame([{
            "Month_num": next_date.month, "Quarter": next_date.quarter,
            "lag_1": last3[-1], "lag_2": last3[-2], "lag_3": last3[-3],
            "rolling_mean_3": last3.mean(),
        }])
        pred = model.predict(row[feat_cols])[0]
        preds.append(pred)
        history = pd.concat([history, pd.DataFrame([{"Order Date": next_date, "Sales": pred}])], ignore_index=True)

    idx = pd.date_range(sdf["Order Date"].iloc[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")
    return pd.Series(preds, index=idx)


MODEL_FNS = {"SARIMA": sarima_forecast, "Prophet": prophet_forecast, "XGBoost": xgboost_forecast}


@st.cache_data
def pick_best_model(series):
    """Fit all 3 models on a train/test split, return (best_name, metrics_dict) - mirrors Task 3's logic."""
    train, test = series.iloc[:-3], series.iloc[-3:]
    metrics = {}
    for name, fn in MODEL_FNS.items():
        try:
            fc = fn(train, 3)
            fc = fc.reindex(test.index)
            mae = float(np.mean(np.abs(test.values - fc.values)))
            rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
            mape = float(np.mean(np.abs((test.values - fc.values) / test.values)) * 100)
            metrics[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        except Exception as e:
            metrics[name] = {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf}
    best = min(metrics, key=lambda k: metrics[k]["MAPE"])
    return best, metrics


df = load_data()

st.title("📈 Sales Forecasting & Demand Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ---------------------------------------------------------------------------
# Page 1 — Sales Overview
# ---------------------------------------------------------------------------
if page == "Sales Overview":
    st.header("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df["Region"].unique().tolist()))
    with col2:
        category_filter = st.selectbox("Filter by Category", ["All"] + sorted(df["Category"].unique().tolist()))

    filtered = df.copy()
    if region_filter != "All":
        filtered = filtered[filtered["Region"] == region_filter]
    if category_filter != "All":
        filtered = filtered[filtered["Category"] == category_filter]

    yearly = filtered.groupby("Year")["Sales"].sum()
    st.subheader("Total Sales by Year")
    st.bar_chart(yearly)

    st.subheader("Monthly Sales Trend")
    monthly = filtered.set_index("Order Date").resample("MS")["Sales"].sum()
    st.line_chart(monthly)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sales by Region")
        st.bar_chart(filtered.groupby("Region")["Sales"].sum())
    with c2:
        st.subheader("Sales by Category")
        st.bar_chart(filtered.groupby("Category")["Sales"].sum())

# ---------------------------------------------------------------------------
# Page 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.header("Forecast Explorer")

    dim_type = st.selectbox("Forecast by", ["Category", "Region"])
    if dim_type == "Category":
        options = sorted(df["Category"].unique().tolist())
        selected = st.selectbox("Select Category", options)
        series = monthly_series(df, category=selected)
    else:
        options = sorted(df["Region"].unique().tolist())
        selected = st.selectbox("Select Region", options)
        series = monthly_series(df, region=selected)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    with st.spinner("Fitting SARIMA, Prophet, and XGBoost, and picking the best one on held-out data..."):
        best_name, metrics = pick_best_model(series)
        train_full = series  # use all available history for the actual forward forecast
        forecast = MODEL_FNS[best_name](train_full, horizon)

    st.info(f"**Best model for this segment: {best_name}** (lowest MAPE on the last 3 known months) — used for the forecast below.")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, label="Actual")
    ax.plot(forecast.index, forecast.values, label=f"{best_name} Forecast", marker="o", color="red")
    ax.set_title(f"{horizon}-Month Forecast: {selected}")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Model comparison for this segment (held-out MAE / RMSE / MAPE)")
    st.dataframe(pd.DataFrame(metrics).T.round(2))

    st.subheader("Forecast values")
    st.dataframe(forecast.rename("Forecasted Sales").reset_index().rename(columns={"index": "Date"}))

# ---------------------------------------------------------------------------
# Page 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.header("Anomaly Report")

    weekly = weekly_series(df)
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["anomaly"] = iso.fit_predict(weekly[["Sales"]])
    anomalies = weekly[weekly["anomaly"] == -1]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(weekly["Order Date"], weekly["Sales"], label="Weekly Sales")
    ax.scatter(anomalies["Order Date"], anomalies["Sales"], color="red", s=50, label="Anomaly", zorder=5)
    ax.legend()
    ax.set_title("Weekly Sales with Detected Anomalies (Isolation Forest)")
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    st.dataframe(anomalies[["Order Date", "Sales"]].reset_index(drop=True))

# ---------------------------------------------------------------------------
# Page 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.header("Product Demand Segments")

    sub_features = df.groupby("Sub-Category").apply(
        lambda g: pd.Series({
            "total_sales": g["Sales"].sum(),
            "avg_order_value": g["Sales"].mean(),
        })
    ).reset_index()

    yearly_sub = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack()
    sub_features["growth_rate"] = sub_features["Sub-Category"].map(
        (yearly_sub[yearly_sub.columns[-1]] - yearly_sub[yearly_sub.columns[0]])
        / yearly_sub[yearly_sub.columns[0]] * 100
    )
    monthly_sub = df.set_index("Order Date").groupby("Sub-Category").resample("MS")["Sales"].sum()
    sub_features["volatility"] = sub_features["Sub-Category"].map(monthly_sub.groupby("Sub-Category").std())
    sub_features = sub_features.dropna()

    feature_cols = ["total_sales", "growth_rate", "volatility", "avg_order_value"]
    X = StandardScaler().fit_transform(sub_features[feature_cols])

    k = st.slider("Number of clusters (k)", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sub_features["cluster"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    sub_features["pca_1"], sub_features["pca_2"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(sub_features["pca_1"], sub_features["pca_2"], c=sub_features["cluster"], cmap="viridis", s=120)
    for _, row in sub_features.iterrows():
        ax.annotate(row["Sub-Category"], (row["pca_1"], row["pca_2"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_title("Product Sub-Category Clusters (PCA-reduced)")
    st.pyplot(fig)

    st.subheader("Sub-categories by cluster")
    st.dataframe(
        sub_features[["Sub-Category", "total_sales", "growth_rate", "volatility", "cluster"]]
        .sort_values("cluster")
        .reset_index(drop=True)
    )
