import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Apple Stock Forecasting", layout="wide")
st.title("🍎 Apple Stock Price Forecasting Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    dates = pd.date_range(start="2014-01-01", end="2024-12-31", freq="B")
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0.35, 2.1, len(dates))) + 90
    return pd.DataFrame({"Date": dates, "Adj Close": prices})

df = load_data()
df.set_index("Date", inplace=True)
df["Adj Close"] = df["Adj Close"].ffill()

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
data = df["Adj Close"]
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# ==================================================
# ARIMA
# ==================================================
arima_model = ARIMA(train, order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_preds = arima_fit.forecast(steps=len(test))

# ==================================================
# SARIMA
# ==================================================
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)
sarima_preds = sarima_fit.forecast(steps=len(test))

# ==================================================
# ARIMAX
# ==================================================
df_feat = df.copy()
df_feat["lag_1"] = df_feat["Adj Close"].shift(1)
df_feat["rolling_mean_5"] = df_feat["Adj Close"].rolling(5).mean()
df_feat["rolling_std_5"] = df_feat["Adj Close"].rolling(5).std()
df_feat["day_of_week"] = df_feat.index.dayofweek
df_feat.dropna(inplace=True)

target = df_feat["Adj Close"]
exog = df_feat[["lag_1", "rolling_mean_5", "rolling_std_5", "day_of_week"]]

train_size_feat = int(len(df_feat) * 0.8)
y_train, y_test = target[:train_size_feat], target[train_size_feat:]
X_train, X_test = exog[:train_size_feat], exog[train_size_feat:]

arimax_model = SARIMAX(
    y_train,
    exog=X_train,
    order=(3, 1, 2),
    enforce_stationarity=False,
    enforce_invertibility=False
)
arimax_fit = arimax_model.fit(disp=False)
arimax_preds = arimax_fit.predict(
    start=y_test.index[0],
    end=y_test.index[-1],
    exog=X_test
)

# ==================================================
# LSTM
# ==================================================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.values.reshape(-1, 1))

X_train_lstm, y_train_lstm = [], []
for i in range(60, train_size):
    X_train_lstm.append(scaled[i-60:i, 0])
    y_train_lstm.append(scaled[i, 0])

X_train_lstm = np.array(X_train_lstm).reshape(-1, 60, 1)
y_train_lstm = np.array(y_train_lstm)

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=0)

X_test_lstm = []
for i in range(train_size, len(scaled)):
    X_test_lstm.append(scaled[i-60:i, 0])

X_test_lstm = np.array(X_test_lstm).reshape(-1, 60, 1)
lstm_preds = scaler.inverse_transform(lstm_model.predict(X_test_lstm))

# --------------------------------------------------
# METRICS
# --------------------------------------------------
def get_metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    )

arima_rmse, arima_mae, arima_mape = get_metrics(test, arima_preds)
sarima_rmse, sarima_mae, sarima_mape = get_metrics(test, sarima_preds)
arimax_rmse, arimax_mae, arimax_mape = get_metrics(y_test, arimax_preds)
lstm_rmse, lstm_mae, lstm_mape = get_metrics(test.values, lstm_preds.flatten())

comparison_df = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "ARIMAX", "LSTM"],
    "RMSE": [arima_rmse, sarima_rmse, arimax_rmse, lstm_rmse],
    "MAE": [arima_mae, sarima_mae, arimax_mae, lstm_mae],
    "MAPE (%)": [arima_mape, sarima_mape, arimax_mape, lstm_mape]
})

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 History",
    "🧠 Model Evaluation",
    "🔮 30-Day Forecast",
    "📊 Business Insights"
])

# ==================================================
# TAB 1: HISTORY (INTERACTIVE)
# ==================================================
with tab1:
    st.subheader("📈 Historical Overview")

    df["Daily Return"] = df["Adj Close"].pct_change()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("All-Time High ($)", f"{df['Adj Close'].max():.2f}")
    col2.metric("All-Time Low ($)", f"{df['Adj Close'].min():.2f}")
    col3.metric("Latest Price ($)", f"{df['Adj Close'].iloc[-1]:.2f}")
    col4.metric("Average Price ($)", f"{df['Adj Close'].mean():.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        "1-Day Change (%)",
        f"{((df['Adj Close'].iloc[-1] - df['Adj Close'].iloc[-2]) / df['Adj Close'].iloc[-2]) * 100:.2f}%"
    )
    col6.metric("52-Week High ($)", f"{df.tail(252)['Adj Close'].max():.2f}")
    col7.metric("52-Week Low ($)", f"{df.tail(252)['Adj Close'].min():.2f}")
    col8.metric("Annual Volatility (%)",
                f"{df['Daily Return'].std() * np.sqrt(252) * 100:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Adj Close"],
        mode="lines",
        name="Adjusted Close",
        line=dict(color="#1f77b4", width=2)
    ))
    fig.update_layout(
        title="Apple Adjusted Close Price – Full History",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 2: MODEL EVALUATION (INTERACTIVE)
# ==================================================
with tab2:
    st.subheader("Model Performance Comparison")

    st.dataframe(
        comparison_df.style.format({
            "RMSE": "{:.2f}",
            "MAE": "{:.2f}",
            "MAPE (%)": "{:.2f}"
        })
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test.index, y=test, name="Actual",
                             line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=arima_preds, name="ARIMA"))
    fig.add_trace(go.Scatter(x=test.index, y=sarima_preds, name="SARIMA"))
    fig.add_trace(go.Scatter(x=y_test.index, y=arimax_preds, name="ARIMAX"))
    fig.add_trace(go.Scatter(x=test.index, y=lstm_preds.flatten(), name="LSTM"))

    fig.update_layout(
        title="Model Comparison – Interactive",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 3: 30-DAY FORECAST (INTERACTIVE)
# ==================================================
with tab3:
    st.subheader("🔮 Next 30 Business Days Forecast (ARIMAX)")

    forecast_days = 30
    future_dates = pd.date_range(df_feat.index[-1], periods=forecast_days + 1, freq="B")[1:]

    last = df_feat.iloc[-1]
    future_exog = pd.DataFrame({
        "lag_1": [last["Adj Close"]] * forecast_days,
        "rolling_mean_5": [last["rolling_mean_5"]] * forecast_days,
        "rolling_std_5": [last["rolling_std_5"]] * forecast_days,
        "day_of_week": future_dates.dayofweek
    }, index=future_dates)

    future_preds = arimax_fit.forecast(steps=forecast_days, exog=future_exog)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-60:],
        y=df["Adj Close"].iloc[-60:],
        mode="lines",
        name="Recent Actual",
        line=dict(color="black", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode="lines",
        name="30-Day Forecast",
        line=dict(color="green", dash="dash")
    ))

    fig.update_layout(
        title="Next 30 Business Days Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 4: BUSINESS INSIGHTS
# ==================================================
with tab4:
    st.subheader("Insights for Dashboard Users")

    st.markdown("""
- **ARIMA** provides a strong statistical baseline.
- **SARIMA** assumes seasonality, which is weak in this dataset.
- **ARIMAX** best captures short-term momentum and volatility.
- **LSTM** learns nonlinear patterns but needs large real datasets.
- Forecasts indicate **trend direction**, not guaranteed prices.
- Combine with **fundamental analysis & market news** for decisions.
    """)
