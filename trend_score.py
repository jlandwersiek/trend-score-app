import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.rest import APIError
import ta

# === Streamlit Dashboard ===
st.set_page_config(page_title="Trend Follow Dashboard", layout="wide")
st.title("Trend Follow Score Dashboard")

# --- Sidebar Inputs ---
api_key_input = st.sidebar.text_input("Alpaca API Key", type="password")
api_secret_input = st.sidebar.text_input("Alpaca Secret Key", type="password")
# safe secrets fallback
_secrets = st.secrets.get("alpaca") or {}
api_key = api_key_input or _secrets.get("api_key", "")
api_secret = api_secret_input or _secrets.get("api_secret", "")
mode = st.sidebar.selectbox("Trend Mode", ["Bull", "Bear"])
tickers = st.sidebar.text_area("Tickers (comma-separated)", "AAPL,MSFT,GOOG")
def_days = 30
history_days = st.sidebar.number_input("Historical Lookback (days)", def_days, 365, def_days*5)
run_button = st.sidebar.button("Run Analysis")

if not run_button:
    st.info("Enter credentials and click Run Analysis to load data.")

# --- Constants ---
LOOKBACK_BULL = 20
LOOKBACK_BEAR = 25
BEAR_VOL_MULT = 1.10
BEAR_ATR_FRAC = 0.15
ADX_LEN = 14
RMI_LEN = 20
RMI_MOM = 5
# force daily bars
TF = TimeFrame(1, TimeFrameUnit.Day)

# --- Data Fetching ---
def fetch_ohlcv(sym, client, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TF, start=start, end=end)
    try:
        bars = client.get_stock_bars(req).df
    except APIError as e:
        st.error(f"Alpaca API Error for {sym}: {e}")
        return pd.DataFrame()
    df = bars[bars.index.get_level_values('symbol') == sym].reset_index()
    df['dt'] = pd.to_datetime(df['timestamp'])
    df.set_index('dt', inplace=True)
    return df[['open','high','low','close','volume']]

# --- Indicators ---
def compute_indicators(df):
    df2 = df.copy()
    df2['sma20'] = ta.trend.SMAIndicator(df2['close'], 20).sma_indicator()
    df2['sma50'] = ta.trend.SMAIndicator(df2['close'], 50).sma_indicator()
    df2['ema12'] = ta.trend.EMAIndicator(df2['close'], 12).ema_indicator()
    df2['ema26'] = ta.trend.EMAIndicator(df2['close'], 26).ema_indicator()
    df2['macd_line'] = df2['ema12'] - df2['ema26']
    df2['macd_sig'] = ta.trend.EMAIndicator(df2['macd_line'], 9).ema_indicator()
    df2['macd_hist'] = df2['macd_line'] - df2['macd_sig']
    df2['adx'] = ta.trend.ADXIndicator(df2['high'], df2['low'], df2['close'], ADX_LEN).adx()
    df2['vol_avg'] = df2['volume'].rolling(20).mean()
    df2['atr14'] = ta.volatility.AverageTrueRange(df2['high'], df2['low'], df2['close'], 14).average_true_range()
    upR = np.maximum(df2['close'] - df2['close'].shift(RMI_MOM), 0)
    dnR = np.maximum(df2['close'].shift(RMI_MOM) - df2['close'], 0)
    num = ta.trend.EMAIndicator(upR, RMI_LEN).ema_indicator()
    den = num + ta.trend.EMAIndicator(dnR, RMI_LEN).ema_indicator()
    df2['rmi'] = 100 * num / den
    return df2.dropna()

# --- Scoring & Signals ---
def score_signals(df, mode):
    if df.empty:
        return {}
    last = df.iloc[-1]
    lb = LOOKBACK_BEAR if mode == 'Bear' else LOOKBACK_BULL
    rh = df['high'].rolling(lb).max().shift(1).iloc[-1]
    rl = df['low'].rolling(lb).min().shift(1).iloc[-1]
    bh = last['close'] > rh
    br = bh and last['volume'] > last['vol_avg']
    wb = bh and not br
    bl = last['close'] < rl
    bd = last['close'] < rl - BEAR_ATR_FRAC * last['atr14']
    bdv = bd and last['volume'] > last['vol_avg'] * BEAR_VOL_MULT
    wbd = bl and not bdv
    s = 0
    # SMA50
    if mode == 'Bull':
        s += 2 if last['close'] > last['sma50'] else (1 if last['close'] > 0.99 * last['sma50'] else 0)
    else:
        s += 2 if last['close'] < last['sma50'] else (1 if last['close'] < 1.01 * last['sma50'] else 0)
    # MACD
    if mode == 'Bull' and last['macd_line'] > last['macd_sig']:
        s += 2 if last['macd_hist'] > 0.05 else (1.5 if last['macd_hist'] > 0.01 else 1)
    if mode == 'Bear' and last['macd_line'] < last['macd_sig']:
        s += 2 if last['macd_hist'] < -0.05 else (1.5 if last['macd_hist'] < -0.01 else 1)
    # ADX
    s += 2.5 if last['adx'] > 30 else (2 if last['adx'] > 20 else (1 if last['adx'] > 18 else 0))
    # SMA20
    if mode == 'Bull':
        s += 1 if last['close'] > last['sma20'] else (0.5 if last['close'] > 0.99 * last['sma20'] else 0)
    else:
        s += 1 if last['close'] < last['sma20'] else (0.5 if last['close'] < 1.01 * last['sma20'] else 0)
    # Volume
    s += 1 if last['volume'] > last['vol_avg'] else (0.5 if last['volume'] > 0.95 * last['vol_avg'] else 0)
    # RMI
    if mode == 'Bull': s += 1 if last['rmi'] >= 55 else (0.5 if last['rmi'] >= 50 else 0)
    else:         s += 1 if last['rmi'] <= 40 else (0.5 if last['rmi'] <= 50 else 0)
    # Break
    s += 0.5 if (mode == 'Bull' and br) or (mode == 'Bear' and bdv) else 0
    pct = round((s / 10.5) * 100, 2)
    entry = br if mode == 'Bull' else bdv
    exit_ = (last['close'] < last['sma20']) if mode == 'Bull' else (last['close'] > last['sma20'])
    label = (
        'Strong Breakout' if br else
        'Weak Breakout'   if wb else
        'Strong Breakdown' if bdv else
        'Weak Breakdown'   if wbd else
        'None'
    )
    return {
        'Score (%)': pct,
        'Price vs SMA50': 'Close above SMA50' if last['close'] > last['sma50'] else 'Close below SMA50',
        'MACD': 'Bullish' if (mode=='Bull' and last['macd_hist']>0) or (mode=='Bear' and last['macd_hist']<0) else 'Bearish',
        'ADX': round(last['adx'], 2),
        'Price vs SMA20': 'Close above SMA20' if last['close'] > last['sma20'] else 'Close below SMA20',
        'Volume': 'Volume above average' if last['volume'] > last['vol_avg'] else 'Volume below average',
        'RMI': round(last['rmi'], 1),
        'Break': label,
        'Entry': entry,
        'Exit': exit_
    }

# --- Run Analysis ---
if run_button:
    if not api_key or not api_secret:
        st.error("Enter both API key and secret.")
    else:
        client = StockHistoricalDataClient(api_key, api_secret)
        symbols = [s.strip().upper() for s in tickers.split(',')]
        results = []
        for sym in symbols:
            df_ohlcv = fetch_ohlcv(sym, client, history_days)
            rec = score_signals(compute_indicators(df_ohlcv) if not df_ohlcv






