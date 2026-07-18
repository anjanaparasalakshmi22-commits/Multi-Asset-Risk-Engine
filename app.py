import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Multi-Asset Risk Engine", layout="wide")
st.title("Multi-Asset Risk Engine")
st.caption("VaR & Expected Shortfall under 3 methodologies — historical, parametric, Monte Carlo")

TICKERS = ["BIL", "BNDW", "DBC", "DX-Y.NYB", "REET", "URTH"]
LABELS = ["Cash (BIL)", "Bonds (BNDW)", "Commodities (DBC)",
          "Dollar (DXY)", "Real Estate (REET)", "Equities (URTH)"]

@st.cache_data
def load_returns():
    data = yf.download(TICKERS, start="2023-07-01")["Close"]
    data = data.dropna()
    return data.pct_change().dropna()

returns = load_returns()
st.write("Data shape:", returns.shape)

st.sidebar.header("Your Portfolio")
investment = st.sidebar.number_input("Investment (€)", 1000, 10_000_000, 10_000, step=1000)
alpha = st.sidebar.selectbox("Confidence level", [0.95, 0.99])

weights = []
for label in LABELS:
    w = st.sidebar.slider(label, 0, 100, 16) / 100
    weights.append(w)
weights = np.array(weights)

total = weights.sum()
if total == 0:
    st.warning("Set at least one weight above zero.")
    st.stop()
weights = weights / total
st.sidebar.caption(f"Weights auto-normalized to 100%")

port = returns @ weights
q = 1 - alpha

hist_var = -port.quantile(q)
hist_es = -port[port <= port.quantile(q)].mean()

sigma = np.sqrt(weights @ returns.cov() @ weights)
param_var = sigma * norm.ppf(alpha)

np.random.seed(42)
mu = returns.mean().values
L = np.linalg.cholesky(returns.cov().values)
Z = np.random.standard_normal((10_000, len(weights)))
sim = pd.Series((mu + Z @ L.T) @ weights)
mc_var = -sim.quantile(q)
mc_es = -sim[sim <= sim.quantile(q)].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Portfolio σ (daily)", f"{sigma:.3%}")
col2.metric(f"Historical VaR {alpha:.0%}", f"€{hist_var*investment:,.0f}")
col3.metric(f"Historical ES {alpha:.0%}", f"€{hist_es*investment:,.0f}")

st.subheader("Risk report — all methodologies")
report = pd.DataFrame({
    "Historical": [hist_var, hist_es],
    "Parametric": [param_var, np.nan],
    "Monte Carlo": [mc_var, mc_es],
}, index=["VaR", "ES"]) * investment
st.dataframe(report.style.format("€{:,.0f}"))

st.subheader("Portfolio P&L distribution")
st.bar_chart(np.histogram(port, bins=50)[0])
