

# app.py - NVIDIA Stock Direction Forecasting Dashboard
import streamlit as st
import pandas as pd
from PIL import Image
import mlflow
from mlflow.tracking import MlflowClient
import os
from datetime import datetime, timedelta
import plotly.express as px

# Configure page
st.set_page_config(
    layout="wide",
    page_title="NVIDIA Stock Forecast",
    page_icon="📈"
)

# Set MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Load data
@st.cache_data
def load_data():
    stock = pd.read_csv("nvidia_stock_data.csv", parse_dates=['Date'])
    financials = pd.read_csv("stock_financials.csv", parse_dates=['Date'])
    return stock, financials

df_stock, df_financials = load_data()

# Model configuration
MODELS = {
    "ARIMA": {
        "base": {
            "image": "arima_forecast_base.png",
            "model": "arima_model_base.pkl"
        },
        "tuned": {
            "image": "arima_forecast_tuned.png",
            "model": "arima_model_tuned.pkl"
        }
    },
    "LSTM": {
        "base": {
            "image": "lstm_base_history.png",
            "confusion": "lstm_base_cm.png",
            "model": "model_lstm_base.pkl"
        },
        "tuned": {
            "image": "cm_lstm_tuned.png",
            "confusion": "cm_lstm_tuned.png",
            "model": "best_lstm_tuned.pkl"
        }
    },
    "XGBoost": {
        "base": {
            "image": "xgb_feature_importance_base.png",
            "confusion": "xgb_cm_base.png",
            "model": "xgb_model_base.pkl"
        },
        "tuned": {
            "image": "xgb_feature_importance_tuned.png",
            "confusion": "xgb_cm_tuned.png",
            "model": "xgb_tuned_model.pkl"
        }
    }
}

# Get MLflow metrics
@st.cache_data
def get_metrics(model_name):
    experiments = client.search_experiments()
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            if run.data.tags.get("mlflow.runName", "") == model_name:
                return {
                    "params": run.data.params,
                    "metrics": run.data.metrics
                }
    return None

# Main app
st.title("NVIDIA Stock Direction Forecasting")

# Sidebar controls
with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Select Section",
        ["Financial Overview", "Model Forecast", "Model Comparison"],
        index=0
    )
    
    if section != "Financial Overview":
        model_family = st.selectbox(
            "Model Type",
            list(MODELS.keys()),
            index=0
        )
        model_version = st.radio(
            "Model Version",
            ["base", "tuned"],
            horizontal=True
        )

# Selected model config
selected_model = MODELS[model_family][model_version] if section != "Financial Overview" else None

# Section display
if section == "Financial Overview":
    st.header("Financial Performance")
    
    # Date range selector
    min_date = df_financials['Date'].min().date()
    max_date = df_financials['Date'].max().date()
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    
    # Filter data
    filtered_fin = df_financials[
        (df_financials['Date'].dt.date >= date_range[0]) & 
        (df_financials['Date'].dt.date <= date_range[1])
    ]
    
    # Financial metrics selector
    metrics = st.multiselect(
        "Select Metrics to Display",
        options=[col for col in df_financials.columns if col not in ['Date']],
        default=['Revenue', 'Net Income']
    )
    
    # Plot financials
    if metrics:
        fig = px.line(
            filtered_fin,
            x='Date',
            y=metrics,
            title="Financial Metrics Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    st.subheader("Raw Financial Data")
    st.dataframe(filtered_fin, use_container_width=True)

elif section == "Model Forecast":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"{model_family} {model_version.capitalize()} Model")
        
        # Display forecast image
        if os.path.exists(selected_model["image"]):
            st.image(
                Image.open(selected_model["image"]),
                caption=f"{model_family} Forecast",
                use_column_width=True
            )
        
        # Display confusion matrix if available
        if "confusion" in selected_model and os.path.exists(selected_model["confusion"]):
            st.image(
                Image.open(selected_model["confusion"]),
                caption="Confusion Matrix",
                use_column_width=True
            )
    
    with col2:
        # Model metrics
        st.subheader("Model Performance")
        metrics = get_metrics(f"{model_family}_{model_version}")
        
        if metrics:
            # Key metrics cards
            st.metric("Accuracy", f"{metrics['metrics'].get('accuracy', 0):.2%}")
            st.metric("Precision", f"{metrics['metrics'].get('precision', 0):.2%}")
            st.metric("Recall", f"{metrics['metrics'].get('recall', 0):.2%}")
            st.metric("F1 Score", f"{metrics['metrics'].get('f1', 0):.2%}")
            
            # All metrics table
            st.subheader("All Metrics")
            metrics_df = pd.DataFrame.from_dict(metrics["metrics"], orient='index', columns=['Value'])
            st.dataframe(metrics_df.style.format("{:.4f}"))
        
        # Future prediction
        st.subheader("Future Direction Prediction")
        pred_date = st.date_input(
            "Select prediction date",
            min_value=datetime.today(),
            max_value=datetime.today() + timedelta(days=365)
        )
        
        if st.button("Predict Direction"):
            # Here you would add your prediction logic
            # This is a placeholder - replace with actual prediction code
            prediction = 1  # Replace with model prediction
            st.success(f"Predicted direction for {pred_date}: {'↑ Up' if prediction == 1 else '↓ Down'}")
            
            # Export prediction
            if st.button("Export Prediction"):
                # Export logic would go here
                st.success("Prediction exported successfully")

else:  # Model Comparison
    st.header("Model Comparison")
    
    # Select models to compare
    models_to_compare = st.multiselect(
        "Select models to compare",
        options=[f"{family}_{version}" for family in MODELS for version in MODELS[family]],
        default=["ARIMA_base", "LSTM_tuned"]
    )
    
    if models_to_compare:
        # Get metrics for all selected models
        comparison_data = []
        for model in models_to_compare:
            family, version = model.split("_")
            metrics = get_metrics(model)
            if metrics:
                metrics['metrics']['Model'] = model
                comparison_data.append(metrics['metrics'])
        
        if comparison_data:
            # Create comparison dataframe
            compare_df = pd.DataFrame(comparison_data).set_index('Model')
            
            # Display comparison metrics
            st.subheader("Performance Comparison")
            st.dataframe(compare_df.style.format("{:.4f}"), use_container_width=True)
            
            # Visual comparison
            metric_to_plot = st.selectbox(
                "Select metric to visualize",
                options=['accuracy', 'precision', 'recall', 'f1'],
                index=0
            )
            
            if metric_to_plot in compare_df.columns:
                fig = px.bar(
                    compare_df.reset_index(),
                    x='Model',
                    y=metric_to_plot,
                    title=f"{metric_to_plot.capitalize()} Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No metrics available for selected models")

# Add some styling

st.markdown("<style>
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
", unsafe_allow_html=True")

st.caption("Built by Bamise - Omatseye - Gideon • Powered by Streamlit")
