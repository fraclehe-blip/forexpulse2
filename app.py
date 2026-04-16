# ============================================================
# ForexPulse AI Pro - Arquitectura PRO (Cloud-Safe)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import ta
import time
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(page_title="ForexPulse AI Pro", layout="wide")

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "CHF=X"
}

TIMEFRAMES = {
    "M5": ("5m", "5d"),
    "M15": ("15m", "10d")
}

# ================= DATA =================
@st.cache_data(ttl=60)
def get_data(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

# ================= INDICATORS =================
def add_indicators(df):
    df = df.copy()

    df["EMA9"] = ta.trend.ema_indicator(df["Close"], 9)
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["RSI"] = ta.momentum.rsi(df["Close"], 14)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_H"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()

    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"])

    df.dropna(inplace=True)
    return df

# ================= AI ENGINE (SIMULADO PRO) =================
def ai_engine(df):
    """
    Simula un modelo híbrido (LSTM + indicadores).
    Sustituible por API externa real.
    """
    row = df.iloc[-1]

    score = 0.5

    if row["EMA9"] > row["EMA21"]:
        score += 0.12
    else:
        score -= 0.12

    if row["RSI"] < 50:
        score += 0.06
    else:
        score -= 0.06

    if row["MACD_H"] > 0:
        score += 0.08
    else:
        score -= 0.08

    if row["ADX"] > 20:
        score += 0.05

    prob = float(np.clip(score, 0.05, 0.95))
    return prob

# ================= SIGNAL ENGINE =================
def generate_signal(df, prob):
    row = df.iloc[-1]
    confirms = 0

    if row["EMA9"] > row["EMA21"]: confirms += 1
    if row["RSI"] < 70: confirms += 1
    if row["MACD_H"] > 0: confirms += 1
    if row["ADX"] > 20: confirms += 1

    if prob > 0.74 and confirms >= 3:
        return "BUY", confirms
    elif prob < 0.26 and confirms >= 3:
        return "SELL", confirms
    else:
        return "WAIT", confirms

# ================= RISK =================
def risk_management(price, atr, direction, risk_pct=1, capital=10000):
    risk_amount = capital * risk_pct / 100

    if direction == "BUY":
        sl = price - atr * 1.5
        tp = price + atr * 2.5
    elif direction == "SELL":
        sl = price + atr * 1.5
        tp = price - atr * 2.5
    else:
        return 0, 0, 0

    position_size = risk_amount / abs(price - sl)
    return sl, tp, position_size

# ================= UI =================
st.title("📈 ForexPulse AI Pro")

pair = st.selectbox("Par", list(PAIRS.keys()))
tf = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))

symbol = PAIRS[pair]
interval, period = TIMEFRAMES[tf]

df = get_data(symbol, interval, period)

if df.empty:
    st.error("Error cargando datos")
    st.stop()

df = add_indicators(df)

# ================= AI =================
prob = ai_engine(df)
signal, confirms = generate_signal(df, prob)

price = df["Close"].iloc[-1]
atr = df["ATR"].iloc[-1]

sl, tp, size = risk_management(price, atr, signal)

# ================= DASHBOARD =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Precio", f"{price:.5f}")
col2.metric("Probabilidad AI", f"{prob*100:.1f}%")
col3.metric("Confirmaciones", confirms)
col4.metric("ATR", f"{atr:.5f}")

# ================= SIGNAL =================
if signal == "BUY":
    st.success(f"🚨 SEÑAL FUERTE DE COMPRA {pair}")
elif signal == "SELL":
    st.error(f"🚨 SEÑAL FUERTE DE VENTA {pair}")
else:
    st.info("⏸️ SIN SEÑAL")

# ================= RISK =================
if signal != "WAIT":
    st.markdown(f"""
    **SL:** {sl:.5f}  
    **TP:** {tp:.5f}  
    **Tamaño posición:** {size:.2f}
    """)

# ================= CHART =================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Precio"
))

fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21"))

st.plotly_chart(fig, use_container_width=True)

# ================= AUTO REFRESH =================
if st.checkbox("Auto refresh"):
    time.sleep(30)
    st.rerun()

# ================= DISCLAIMER =================
st.warning("""
⚠️ Esta herramienta es exclusivamente educativa y para backtesting.
No constituye consejo financiero ni garantía de resultados.
El trading de forex conlleva alto riesgo de pérdida de capital.
""")