
import streamlit as st
import pandas as pd
from PIL import Image
import os
from datetime import datetime, timedelta
import plotly.express as px
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

st.set_page_config(layout="wide", page_title="NVIDIA Stock Forecast", page_icon="📈")

@st.cache_data
def load_data():
    try:
        stock = pd.read_csv("stock_data.csv", parse_dates=['Date'], index_col='Date')
        financials = pd.read_csv("stock_financials.csv", index_col='Year')
        financials.index = pd.to_datetime(financials.index, format='%Y')
        financials.index.name = 'Date'
        return stock, financials
    except Exception as e:
        st.error(f"❌ Data loading failed. Error: {str(e)}")
        return None, None

@st.cache_data
def load_results_df():
    try:
        return pd.read_csv("model_results.csv", index_col="Model")
    except:
        return pd.DataFrame()

@st.cache_data
def get_metrics(model_name):
    df = load_results_df()
    model_name = model_name.strip()
    if model_name in df.index:
        return df.loc[model_name].drop("Type", errors="ignore").dropna().to_dict()
    return {}

def prepare_features(stock_df, days_from_now):
    latest = stock_df.iloc[-1:].copy()
    latest['DaysAhead'] = days_from_now
    return latest.drop(columns=['NVDA_Direction'], errors='ignore')

def run_prediction(model_path, pred_date, model_family, stock_df):
    try:
        if not os.path.exists(model_path):
            return None
        days_from_now = (pred_date - datetime.today().date()).days
        if days_from_now <= 0:
            return None
        features = prepare_features(stock_df, days_from_now)

        if model_family == "ARIMA":
            model = joblib.load(model_path)
            forecast = model.forecast(steps=days_from_now)
            predicted_return = forecast[-1]
            direction = 1 if predicted_return > 0 else 0
            price_today = stock_df.iloc[-1]["NVDA_Close"]
            predicted_price = price_today * (1 + predicted_return)
            return direction, predicted_price

        selected_features = [
            'NVDA_Close', 'GSPC_Close', 'NVDA_Volume', 'GSPC_Volume',
            'NVDA_Return', 'GSPC_Return', 'NVDA_RollingVol', 'GSPC_RollingVol', 'NVDA_Return_lag1'
        ]
        features = features[selected_features]

        if model_family == "LSTM":
            scaler = joblib.load("scaler_lstm.pkl")
            features_scaled = scaler.transform(features)
            features_reshaped = features_scaled.reshape((1, 1, features_scaled.shape[1]))
            model = joblib.load(model_path)
            pred = model.predict(features_reshaped)
            predicted_price = pred[0][0] if hasattr(pred[0], '__len__') else pred[0]

        elif model_family == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            pred = model.predict(features)
            predicted_price = pred[0]

        last_price = stock_df.iloc[-1]["NVDA_Close"]
        direction = 1 if predicted_price > last_price else 0
        return direction, predicted_price
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

MODELS = {
    "ARIMA": {
        "base": {"image": "arima_forecast_base.png", "model": "arima_model_base.pkl"},
        "tuned": {"image": "arima_forecast_tuned.png", "model": "arima_model_tuned.pkl"}
    },
    "LSTM": {
        "base": {"image": "lstm_base_history.png", "confusion": "lstm_base_cm.png", "model": "model_lstm_base.pkl"},
        "tuned": {"image": "cm_lstm_tuned.png", "confusion": "cm_lstm_tuned.png", "model": "best_lstm_tuned.pkl"}
    },
    "XGBoost": {
        "base": {"image": "xgb_feature_importance_base.png", "confusion": "xgb_cm_base.png", "model": "xgb_model_base.json"},
        "tuned": {"image": "xgb_feature_importance_tuned.png", "confusion": "xgb_cm_tuned.png", "model": "xgb_model_tuned.json"}
    }
}

if missing := [p for m in MODELS.values() for v in m.values() for k, p in v.items() if k in ['image', 'model', 'confusion'] and not os.path.exists(p)]:
    st.warning("Missing files: " + ", ".join(missing))

df_stock, df_financials = load_data()
st.title("NVIDIA Stock Direction Forecasting")

with st.sidebar:
    section = st.radio("View Section", ["Financial Overview", "Model Forecast"])
    if section == "Model Forecast":
        model_family = st.selectbox("Select Model Type", list(MODELS.keys()))
        model_version = st.radio("Version", ["base", "tuned"], horizontal=True)
        selected_model = MODELS[model_family][model_version]
    else:
        selected_model = None

if section == "Financial Overview":
    st.header("📊 Financial Overview")
    if df_financials is not None:
        yr_range = st.slider("Year Range", min(df_financials.index.year), max(df_financials.index.year),
                             (min(df_financials.index.year), max(df_financials.index.year)))
        filtered = df_financials[df_financials.index.year.between(yr_range[0], yr_range[1])]
        metrics = st.multiselect("Choose Metrics", df_financials.columns.tolist(), default=['Revenue', 'Net_Income'])
        if metrics:
            st.plotly_chart(px.line(filtered.reset_index(), x='Date', y=metrics, title="Financial Metrics"))
        st.dataframe(filtered.reset_index())
else:
    st.header(f"{model_family} {model_version.capitalize()} Forecast")
    img = selected_model.get("image")
    if img and os.path.exists(img):
        st.image(Image.open(img), caption=f"{model_family} Forecast", use_container_width=True)

    st.subheader("📈 Model Performance")
    mkey = f"{model_family}_{model_version}"
    m = get_metrics(mkey)
    if m:
        df_metrics = pd.DataFrame.from_dict(m, orient='index', columns=["Value"])
        st.dataframe(df_metrics.style.format("{:.4f}"))

    st.subheader("📅 Predict Direction")
    start_date = df_stock.index.max().date() + timedelta(days=1)
    pred_date = st.date_input("Pick a prediction date", min_value=start_date, max_value=start_date + timedelta(days=365))

    if st.button("Run Prediction"):
        direction, forecast = run_prediction(selected_model["model"], pred_date, model_family, df_stock)
        if direction is not None:
            label = "📈 Up" if direction else "📉 Down"
            output = {
                "Prediction Date": [pred_date.strftime('%Y-%m-%d')],
                "Predicted Direction": [label]
            }
            if model_family == "ARIMA":
                output["Forecast Price"] = [f"${forecast:,.2f}"]
            st.dataframe(pd.DataFrame(output), use_container_width=True)
        else:
            st.error("Prediction failed. Check logs and input.")

st.caption("Built by Bamise - Omatseye - Gideon • Powered by Streamlit")

