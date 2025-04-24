
# app.py - NVIDIA Stock Direction Forecasting Dashboard
import streamlit as st
import pandas as pd
from PIL import Image
import os
from datetime import datetime, timedelta
import plotly.express as px
import joblib
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", page_title="NVIDIA Stock Forecast", page_icon="📈")

@st.cache_data
def load_data():
    try:
        stock = pd.read_csv("nvidia_stock_data.csv", parse_dates=['Date'], index_col='Date')
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
        df = pd.read_csv("model_results.csv", index_col="Model")
        return df
    except Exception as e:
        st.error(f"❌ Could not load model_results.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def get_metrics(model_name):
    results_df = load_results_df()
    model_name = model_name.strip()
    if model_name in results_df.index:
        metrics = results_df.loc[model_name].drop(labels=["Type"], errors="ignore").dropna()
        return {"metrics": metrics.to_dict()}
    else:
        return None

@st.cache_data
def check_model_files():
    missing = []
    for model_family in MODELS.values():
        for version in model_family.values():
            for file_type, path in version.items():
                if file_type in ['image', 'model', 'confusion'] and not os.path.exists(path):
                    missing.append(path)
    return missing

def prepare_features(stock_df, days_from_now):
    latest = stock_df.iloc[-1:].copy()
    latest['DaysAhead'] = days_from_now
    drop_cols = ['NVDA_Direction']
    return latest.drop(columns=[col for col in drop_cols if col in latest.columns], errors='ignore')

def run_prediction(model_path, pred_date, model_family, stock_df):
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

    elif model_family == "LSTM":
        selected_features = [
            'NVDA_Close', 'GSPC_Close', 'NVDA_Volume', 'GSPC_Volume',
            'NVDA_Return', 'GSPC_Return', 'NVDA_RollingVol', 'GSPC_RollingVol',
            'NVDA_Return_lag1'
        ]
        features = features[selected_features]
        scaler_path = "scaler_lstm.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("❌ Scaler file 'scaler_lstm.pkl' not found. Please train and save it.")
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features)
        features_reshaped = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        model = joblib.load(model_path)
        prediction = model.predict(features_reshaped)
        predicted_price = prediction[0][0] if hasattr(prediction[0], '__len__') else prediction[0]

    elif model_family == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(model_path)  # load the sklearn-style .json
        prediction = model.predict(features.values)
        predicted_price = prediction[0]

    else:
        model = joblib.load(model_path)
        prediction = model.predict(features.values)
        predicted_price = prediction[0] if hasattr(prediction, '__len__') else prediction

    last_price = stock_df.iloc[-1]["NVDA_Close"]
    direction = 1 if predicted_price > last_price else 0
    return direction, predicted_price

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

if missing_files := check_model_files():
    st.warning(f"Missing model files: {', '.join(missing_files)}")

df_stock, df_financials = load_data()

st.title("NVIDIA Stock Direction Forecasting")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Select Section", ["Financial Overview", "Model Forecast"], index=0)
    if section != "Financial Overview":
        model_family = st.selectbox("Model Type", list(MODELS.keys()), index=0)
        model_version = st.radio("Model Version", ["base", "tuned"], horizontal=True)

selected_model = MODELS[model_family][model_version] if section != "Financial Overview" else None

if section == "Financial Overview":
    st.header("Financial Performance")
    if df_financials is not None:
        years = df_financials.index.year.unique()
        min_year, max_year = min(years), max(years)
        year_range = st.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
        filtered_fin = df_financials[(df_financials.index.year >= year_range[0]) & (df_financials.index.year <= year_range[1])]
        filtered_fin = filtered_fin.reset_index()

        metrics = st.multiselect("Select Metrics to Display", options=[col for col in df_financials.columns if col not in ['Date']], default=['Revenue', 'Net_Income', 'Gross_Profit', 'Total_Assets', 'Total_Liabilities'])

        if metrics:
            fig = px.line(filtered_fin, x='Date', y=metrics, title="Financial Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Financial Data")
        st.dataframe(filtered_fin, use_container_width=True)
    else:
        st.warning("Financial data could not be loaded.")

elif section == "Model Forecast":
    if selected_model:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"{model_family} {model_version.capitalize()} Model")
            image_path = selected_model.get("image")
            if image_path and os.path.exists(image_path):
                st.image(Image.open(image_path), caption=f"{model_family} Forecast", use_container_width=True)

            confusion_path = selected_model.get("confusion")
            if confusion_path and os.path.exists(confusion_path):
                st.image(Image.open(confusion_path), caption="Confusion Matrix", use_container_width=True)

        with col2:
            st.subheader("Model Performance")
            model_key = f"{model_family}_{model_version}"
            metrics = get_metrics(model_key)

            if metrics:
                st.metric("Accuracy", f"{metrics['metrics'].get('accuracy', 0):.2%}")
                st.metric("Precision", f"{metrics['metrics'].get('precision', 0):.2%}")
                st.metric("Recall", f"{metrics['metrics'].get('recall', 0):.2%}")
                st.metric("F1 Score", f"{metrics['metrics'].get('f1', 0):.2%}")

                st.subheader("All Metrics")
                metrics_df = pd.DataFrame.from_dict(metrics["metrics"], orient='index', columns=['Value'])
                st.dataframe(metrics_df.style.format("{:.4f}"))

            st.subheader("Prediction")
            pred_date = st.date_input("Select prediction date", min_value=datetime.today(), max_value=datetime.today() + timedelta(days=365))
            if st.button("Predict Direction"):
                model_path = selected_model.get("model")
                if not model_path or not os.path.exists(model_path):
                    st.error(f"❌ Model file not found: {model_path}")
                else:
                    result = run_prediction(model_path, pred_date, model_family, df_stock)
                    if result:
                        direction, price = result
                        direction_label = "📈 Up" if direction == 1 else "📉 Down"
                        st.table(pd.DataFrame({
                            "Prediction Date": [pred_date.strftime('%Y-%m-%d')],
                            "Predicted Direction": [direction_label],
                            "Forecast Price": [f"${price:,.2f}"]
                        }))
                    else:
                        st.error("❌ Prediction failed. Please check your model and input.")
    else:
        st.warning("Please select a model type and version in the sidebar.")

st.markdown("""
<style>
    .stMetric {
        border: 1px solid #e1e4e8;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stMetric label {
        font-size: 1rem !important;
        font-weight: bold !important;
    }
    .stMetric div:first-child {
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.caption("Built by Bamise - Omatseye - Gideon • Powered by Streamlit")

