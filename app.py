
# app.py - NVIDIA Stock Direction Forecasting Dashboard
import streamlit as st
import pandas as pd
from PIL import Image
import os
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMAResults

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
    for key in results_df.index:
        if key.startswith(model_name):
            metrics = results_df.loc[key].drop(labels=["Type"], errors="ignore").dropna()
            return {"metrics": metrics.to_dict(), "used_model": key}
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

def get_actual_market_data(pred_date, ticker="NVDA"):
    try:
        start_date = pred_date - timedelta(days=14)
        end_date = pred_date + timedelta(days=1)
        data = yf.download(ticker, start=start_date, end=end_date)
        if len(data) < 2:
            return None, None
        pred_date_str = pred_date.strftime('%Y-%m-%d')
        if pred_date_str in data.index:
            actual_price = data.loc[pred_date_str]["Close"].item()
            prev_idx = data.index.get_loc(pred_date_str) - 1
            prev_price = data.iloc[prev_idx]["Close"].item()
        else:
            actual_price = data.iloc[-1]["Close"].item()
            prev_price = data.iloc[-2]["Close"].item()
        actual_direction = 1 if actual_price > prev_price else 0
        return actual_price, actual_direction
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None, None

def prepare_features(stock_df, days_from_now):
    latest = stock_df.iloc[-1:].copy()
    latest['DaysAhead'] = days_from_now
    drop_cols = ['NVDA_Direction']
    return latest.drop(columns=[col for col in drop_cols if col in latest.columns], errors='ignore')

def weighted_score(row):
    if 'accuracy' in row and 'precision' in row and 'recall' in row and 'f1' in row:
        return 0.35 * row['accuracy'] + 0.25 * row['precision'] + 0.2 * row['recall'] + 0.2 * row['f1']
    else:
        return 0

def load_model_metrics():
    df = load_results_df()
    df = df[~df.index.str.contains("_nf|_cw")]  # Filter out _nf and _cw
    df = df.dropna(subset=['accuracy', 'precision', 'recall', 'f1'], how='any')
    return df

def run_prediction(model_path, pred_date, model_family, stock_df):
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None

        last_available_date = stock_df.index.max().date()
        days_from_now = (pred_date - last_available_date).days
        if days_from_now <= 0:
            st.warning("⚠️ Prediction date must be after the last available stock date.")
            return None

        if model_family == "ARIMA":
            model = ARIMAResults.load(model_path)
            forecast = model.get_forecast(steps=days_from_now)
            predicted_return = forecast.predicted_mean.iloc[-1]
            price_today = stock_df.iloc[-1]["NVDA_Close"]
            predicted_price = price_today * (1 + predicted_return)
            direction = 1 if predicted_return > 0 else 0
            return direction, predicted_price

        features = prepare_features(stock_df, days_from_now)

        if model_family == "LSTM":
            selected_features = [
                'NVDA_Close', 'GSPC_Close', 'NVDA_Volume', 'GSPC_Volume',
                'NVDA_Return', 'GSPC_Return', 'NVDA_RollingVol', 'GSPC_RollingVol',
                'NVDA_Return_lag1']
            features = features[selected_features]
            scaler = joblib.load("scaler_lstm.pkl")
            features_scaled = scaler.transform(features)
            features_reshaped = features_scaled.reshape((1, 1, features_scaled.shape[1]))
            model = joblib.load(model_path)
            prediction = model.predict(features_reshaped)
            predicted_price = prediction[0][0] if hasattr(prediction[0], '__len__') else prediction[0]

        elif model_family == "XGBoost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            selected_features = [
                'NVDA_Close', 'GSPC_Close', 'NVDA_Volume', 'GSPC_Volume',
                'NVDA_Return', 'GSPC_Return', 'NVDA_RollingVol', 'GSPC_RollingVol',
                'NVDA_Return_lag1']
            features = features[selected_features]
            prediction = model.predict(features)
            predicted_price = prediction[0]

        else:
            st.error("❌ Unknown model type selected.")
            return None

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

if missing_files := check_model_files():
    st.warning(f"Missing model files: {', '.join(missing_files)}")

df_stock, df_financials = load_data()

st.title("NVIDIA Stock Direction Forecasting")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Select Section", ["Financial Overview", "Model Forecast", "Model Comparison"], index=0)
    if section == "Model Forecast":
        model_family = st.selectbox("Model Type", list(MODELS.keys()), index=0)
        model_version = st.radio("Model Version", ["base", "tuned"], horizontal=True)

selected_model = MODELS[model_family][model_version] if section == "Model Forecast" else None

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
        st.header(f"{model_family} {model_version.capitalize()} Model")
        st.subheader("Prediction")

        prediction_start = pd.to_datetime(df_stock.index.max().date()) + timedelta(days=1)
        pred_date = st.date_input("Select prediction date", min_value=prediction_start, max_value=prediction_start + timedelta(days=1095))

        if st.button("Predict Direction"):
            model_path = selected_model.get("model")

            if not model_path or not os.path.exists(model_path):
                st.error(f"❌ Model file not found: {model_path}")
            else:
                result = run_prediction(model_path, pred_date, model_family, df_stock)
                if result:
                    direction, predicted_price = result
                    direction_label = "📈 Up" if direction == 1 else "📉 Down"

                    actual_price, actual_direction = get_actual_market_data(pred_date)

                    result_data = {
                        "Prediction Date": [pred_date.strftime('%Y-%m-%d')],
                        "Predicted Direction": [direction_label],
                        "Actual Direction": ["📈 Up" if actual_direction == 1 else "📉 Down" if actual_direction is not None else "N/A"],
                        "Prediction Accuracy": ["✅ Correct" if direction == actual_direction else "❌ Incorrect" if actual_direction is not None else "N/A"]
                    }

                    if model_family == "ARIMA":
                        result_data["Forecast Price"] = [f"${predicted_price:,.2f}"]
                        result_data["Actual Price"] = [f"${actual_price:,.2f}" if actual_price else "N/A"]

                    if actual_direction is None:
                        st.warning("Market data unavailable for this date (may be future date or market closed)")

                    st.dataframe(pd.DataFrame(result_data), use_container_width=True)

        st.subheader("Model Performance")
        model_key = f"{model_family}_{model_version.capitalize()}"
        metrics_result = get_metrics(model_key)

        if metrics_result and isinstance(metrics_result, dict):
            metrics_data = metrics_result.get("metrics")
            used_model = metrics_result.get("used_model", model_key)

            if metrics_data:
                st.write(f"Showing metrics for: `{used_model}`")
                metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index', columns=['Value'])
                st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
            else:
                st.warning("⚠️ No metric values found for the selected model.")
        else:
            st.warning("⚠️ No model performance data available.")

        st.subheader("Charts")
        image_path = selected_model.get("image")
        if image_path and os.path.exists(image_path):
            st.image(Image.open(image_path), caption=f"{model_family} Forecast", use_container_width=True)

        confusion_path = selected_model.get("confusion")
        if confusion_path and os.path.exists(confusion_path):
            st.image(Image.open(confusion_path), caption="Confusion Matrix", use_container_width=True)

    else:
        st.warning("Please select a model type and version in the sidebar.")

elif section == "Model Comparison":
    st.header("Model Comparison")

    st.markdown("This section compares model performance and exports ARIMA price forecasts alongside actual prices.")

    try:
        model_df = pd.read_csv("model_results.csv")
        model_df = model_df[~model_df["Model"].str.contains("_nf|_cw", na=False)].copy()

        # Convert metric columns to numeric
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_df[metrics] = model_df[metrics].apply(pd.to_numeric, errors='coerce')
        model_df = model_df.fillna(0)

        # Weighted scoring for classification models
        weights = {'accuracy': 0.3, 'precision': 0.2, 'recall': 0.1, 'f1': 0.3}
        model_df["Weighted Score"] = (
            model_df['accuracy'] * weights['accuracy'] +
            model_df['precision'] * weights['precision'] +
            model_df['recall'] * weights['recall'] +
            model_df['f1'] * weights['f1']
        )

        st.subheader("Model Metrics Comparison")
        st.markdown("""
        **How Weighted Scoring Works**

        To determine the best classification model, we compute a **weighted score** based on:
        - **Accuracy (30%)** – overall correct predictions
        - **Precision (20%)** – correctness when predicting positives
        - **Recall (10%)** – ability to find all relevant positives
        - **F1 Score (30%)** – balance between precision and recall

        These weights emphasize **F1 Score** and **Accuracy** as they balance precision and recall, crucial for financial decision-making.
        """)

        st.dataframe(model_df.style.format({
            'accuracy': "{:.4f}", 'precision': "{:.4f}", 'recall': "{:.4f}",
            'f1': "{:.4f}", 'roc_auc': "{:.4f}", 'Weighted Score': "{:.4f}"
        }), use_container_width=True)

        st.subheader("📈 Performance Chart")
        fig = px.bar(model_df.sort_values("Weighted Score", ascending=True),
                     x="Weighted Score", y="Model", orientation='h',
                     title="Model Weighted Scores")
        st.plotly_chart(fig, use_container_width=True)

        best_row = model_df[~model_df['Model'].str.contains("ARIMA")].sort_values(by='Weighted Score', ascending=False).iloc[0]
        best_model_name = best_row["Model"]

        st.success(f"Best performing classification model: **{best_model_name}** with score {best_row['Weighted Score']:.4f}")

    except Exception as e:
        st.error(f"Failed to process batch predictions: {e}")

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

