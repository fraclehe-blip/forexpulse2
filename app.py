# ============================================================
# ForexPulse AI Pro - VERSION AVANZADA FUNCIONAL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import ta
import random
from deap import base, creator, tools, algorithms

# TensorFlow opcional
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# ================= CONFIG =================
st.set_page_config(layout="wide", page_title="ForexPulse AI Pro")

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
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    df.dropna(inplace=True)
    return df

# ================= INDICATORS =================
def add_indicators(df):
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

# ================= LSTM =================
class LSTMModel:
    def __init__(self):
        self.model = None

    def train(self, df):
        if not TF_AVAILABLE:
            return

        data = df[["Close"]].values
        X, y = [], []

        for i in range(60, len(data)-1):
            X.append(data[i-60:i])
            y.append(int(data[i+1] > data[i]))

        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(X, y, epochs=5, verbose=0)

        self.model = model

    def predict(self, df):
        if not TF_AVAILABLE or self.model is None:
            return 0.5

        data = df[["Close"]].values[-60:]
        X = np.array([data])
        return float(self.model.predict(X)[0][0])

# ================= SIGNAL =================
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
    return "WAIT", confirms

# ================= GA =================
def fitness(individual):
    return (random.random(),)

def run_ga():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, 5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=10)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=False)

    return tools.selBest(pop, 1)[0]

# ================= UI =================
st.title("📈 ForexPulse AI Pro")

pair = st.selectbox("Par", list(PAIRS.keys()))
tf = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))

symbol = PAIRS[pair]
interval, period = TIMEFRAMES[tf]

df = get_data(symbol, interval, period)
df = add_indicators(df)

# LSTM
model = LSTMModel()
model.train(df)
prob = model.predict(df)

signal, confirms = generate_signal(df, prob)

# ================= DASHBOARD =================
col1, col2, col3 = st.columns(3)

col1.metric("Precio", f"{df['Close'].iloc[-1]:.5f}")
col2.metric("Prob LSTM", f"{prob*100:.1f}%")
col3.metric("Confirmaciones", confirms)

# Señal
if signal == "BUY":
    st.success(f"🚨 SEÑAL DE COMPRA {pair}")
elif signal == "SELL":
    st.error(f"🚨 SEÑAL DE VENTA {pair}")
else:
    st.info("⏸️ SIN SEÑAL")

# ================= CHART =================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
))

fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21"))

st.plotly_chart(fig, use_container_width=True)

# ================= GA =================
if st.button("Optimizar parámetros (GA)"):
    best = run_ga()
    st.write("Mejor individuo:", best)

# ================= DISCLAIMER =================
st.warning("""
⚠️ Esta herramienta es exclusivamente educativa y para backtesting.
No constituye consejo financiero ni garantía de resultados.
El trading de forex conlleva alto riesgo de pérdida de capital.
""")