import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import ta

# === Streamlit Dashboard ===
st.set_page_config(page_title="Trend Follow Dashboard", layout="wide")
st.title("Trend Follow Score Dashboard")

# --- Dashboard Description ---
st.markdown(
    """
    This dashboard applies a series of fundamental technical measures to quantify trend strength,
    momentum, and potential reversals:

    • **SMA20 & SMA50 (Simple Moving Averages)**: smooths price over 20 and 50 periods to identify
      short- and medium-term trend direction. Current close relative to these averages highlights
      whether price is in an uptrend or downtrend.

    • **MACD Histogram**: measures momentum by subtracting the 9-period EMA of the MACD line
      (difference between 12- and 26-period EMAs) from the MACD line itself. Positive values
      signal bullish momentum; negative values signal bearish momentum.

    • **ADX (Average Directional Index)**: quantifies trend strength over 14 periods. Readings
      above 20 suggest a developing trend; above 30 indicate a strong trend.

    • **ATR14 (Average True Range)**: gauges volatility over 14 periods. It helps adjust
      breakdown rules and differentiate strong versus weak moves.

    • **Volume vs. 20-period average**: compares today’s volume to its 20-period moving average,
      filtering moves for their conviction.

    • **RMI (Relative Momentum Index)**: a variant of RSI that uses momentum over shifts with a defined
      lookback to highlight overbought and oversold conditions.

    • **Break classification**: categorizes the latest support/resistance breach as
      "Strong Breakout/Breakdown" (high-volume breach) or "Weak Breakout/Breakdown" (low-volume
      breach).

    The combined **Score (%)** normalizes these components into a 0 to 100 scale. **Entry** and
    **Exit** flags indicate high-conviction signals for entering or exiting positions.
    """
)

# --- Sidebar Inputs ---
api_key    = st.sidebar.text_input("Alpaca API Key", type="password")
api_secret = st.sidebar.text_input("Alpaca Secret Key", type="password")
mode       = st.sidebar.selectbox("Trend Mode", ["Bull", "Bear"], index=0)
tickers    = st.sidebar.text_area("Tickers (comma-separated)", "AAPL,MSFT,GOOG")
resolution = st.sidebar.selectbox(
    "Bar Timeframe",
    ["1Min", "5Min", "15Min", "1H", "4H", "1D", "1W"],
    index=5
)
def_days = 30
history_days = st.sidebar.number_input(
    "Historical Lookback (days)", min_value=def_days, max_value=365, value=def_days*5
)
run_button = st.sidebar.button("Run Analysis")

# --- Parameters ---
LOOKBACK_BULL = 20
LOOKBACK_BEAR = 25
BEAR_VOL_MULT = 1.10
BEAR_ATR_FRAC = 0.15
ADX_LEN       = 14
RMI_LEN       = 20
RMI_MOM       = 5

TF_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1H":   TimeFrame(1, TimeFrameUnit.Hour),
    "4H":   TimeFrame(4, TimeFrameUnit.Hour),
    "1D":   TimeFrame(1, TimeFrameUnit.Day),
    "1W":   TimeFrame(1, TimeFrameUnit.Week)
}

# --- Data Fetching ---
def fetch_ohlcv(symbol, client, timeframe, lookback_days):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, start=start, end=end)
    bars = client.get_stock_bars(req).df
    df = bars[bars.index.get_level_values('symbol')==symbol].reset_index()
    df['dt'] = pd.to_datetime(df['timestamp'])
    df.set_index('dt', inplace=True)
    return df[['open','high','low','close','volume']]

# --- Indicators ---
def compute_indicators(df):
    df2 = df.copy()
    df2['sma20'] = ta.trend.SMAIndicator(df2['close'],20).sma_indicator()
    df2['sma50'] = ta.trend.SMAIndicator(df2['close'],50).sma_indicator()
    df2['ema12'] = ta.trend.EMAIndicator(df2['close'],12).ema_indicator()
    df2['ema26'] = ta.trend.EMAIndicator(df2['close'],26).ema_indicator()
    df2['macd_line'] = df2['ema12'] - df2['ema26']
    df2['macd_sig']  = ta.trend.EMAIndicator(df2['macd_line'],9).ema_indicator()
    df2['macd_hist'] = df2['macd_line'] - df2['macd_sig']
    df2['adx']       = ta.trend.ADXIndicator(df2['high'],df2['low'],df2['close'],ADX_LEN).adx()
    df2['vol_avg']   = df2['volume'].rolling(20).mean()
    df2['atr14']     = ta.volatility.AverageTrueRange(df2['high'],df2['low'],df2['close'],14).average_true_range()
    upR = np.maximum(df2['close']-df2['close'].shift(RMI_MOM),0)
    dnR = np.maximum(df2['close'].shift(RMI_MOM)-df2['close'],0)
    num = ta.trend.EMAIndicator(upR,RMI_LEN).ema_indicator()
    den = num + ta.trend.EMAIndicator(dnR,RMI_LEN).ema_indicator()
    df2['rmi']       = 100 * num/den
    return df2.dropna()

# --- Scoring & Signals ---
def score_signals(df, mode):
    last = df.iloc[-1]
    lb = LOOKBACK_BEAR if mode=='Bear' else LOOKBACK_BULL
    rh = df['high'].rolling(lb).max().shift(1).iloc[-1]
    rl = df['low'].rolling(lb).min().shift(1).iloc[-1]
    bh = last['close'] > rh
    br = bh and last['volume']>last['vol_avg']
    wb = bh and not br
    bl = last['close'] < rl
    bd = last['close'] < rl - BEAR_ATR_FRAC*last['atr14']
    bdv= bd and last['volume']>last['vol_avg']*BEAR_VOL_MULT
    wbd= bl and not bdv
    s = 0
    if mode=='Bull':
        s += 2 if last['close']>last['sma50'] else (1 if last['close']>0.99*last['sma50'] else 0)
    else:
        s += 2 if last['close']<last['sma50'] else (1 if last['close']<1.01*last['sma50'] else 0)
    if mode=='Bull' and last['macd_line']>last['macd_sig']:
        s += 2 if last['macd_hist']>0.05 else (1.5 if last['macd_hist']>0.01 else 1)
    if mode=='Bear' and last['macd_line']<last['macd_sig']:
        s += 2 if last['macd_hist']<-0.05 else (1.5 if last['macd_hist']<-0.01 else 1)
    s += 2.5 if last['adx']>30 else (2 if last['adx']>20 else (1 if last['adx']>18 else 0))
    if mode=='Bull':
        s += 1 if last['close']>last['sma20'] else (0.5 if last['close']>0.99*last['sma20'] else 0)
    else:
        s += 1 if last['close']<last['sma20'] else (0.5 if last['close']<1.01*last['sma20'] else 0)
    s += 1 if last['volume']>last['vol_avg'] else (0.5 if last['volume']>0.95*last['vol_avg'] else 0)
    if mode=='Bull': s += 1 if last['rmi']>=55 else (0.5 if last['rmi']>=50 else 0)
    else:         s += 1 if last['rmi']<=40 else (0.5 if last['rmi']<=50 else 0)
    s += 0.5 if (mode=='Bull' and br) or (mode=='Bear' and bdv) else 0
    pct = round((s/10.5)*100,2)
    entry = br if mode=='Bull' else bdv
    exit_ = (last['close']<last['sma20']) if mode=='Bull' else (last['close']>last['sma20'])
    label = (
        'Strong Breakout' if br else
        'Weak Breakout'   if wb else
        'Strong Breakdown' if bdv else
        'Weak Breakdown'   if wbd else
        'None'
    )
    return {
        'Score (%)': pct,
        'Price vs SMA50': 'Close above SMA50' if last['close']>last['sma50'] else 'Close below SMA50',
        'MACD': 'Bullish' if (mode=='Bull' and last['macd_hist']>0) or (mode=='Bear' and last['macd_hist']<0) else 'Bearish',
        'ADX': round(last['adx'],2),
        'Price vs SMA20': 'Close above SMA20' if last['close']>last['sma20'] else 'Close below SMA20',
        'Volume': 'Volume above average' if last['volume']>last['vol_avg'] else 'Volume below average',
        'RMI': round(last['rmi'],1),
        'Break': label,
        'Entry': entry,
        'Exit': exit_
    }

# --- Run Analysis ---
if run_button:
    if not api_key or not api_secret:
        st.error("Please enter both API key and secret.")
    else:
        client = StockHistoricalDataClient(api_key, api_secret)
        tf = TF_MAP[resolution]
        symbols = [s.strip().upper() for s in tickers.split(',') if s.strip()]
        results = []
        for sym in symbols:
            try:
                ohlcv = fetch_ohlcv(sym, client, tf, history_days)
                data = compute_indicators(ohlcv)
                rec = score_signals(data, mode)
                rec['Symbol'] = sym
                results.append(rec)
            except Exception:
                results.append({'Symbol': sym})
        df = pd.DataFrame(results)
        if 'Symbol' in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('Symbol')))
            df = df[cols]
            df = df.set_index('Symbol')

        # Display header
        st.subheader(f"{mode} Trend Scores")

        # Apply styling to Score column only
        def highlight_score(val):
            try:
                pct = float(val)
            except:
                return ''
            # In Bull mode: green ≥90, orange 70–89; In Bear mode: red ≥90, orange 70–89
            if pct >= 90:
                return 'background-color: lightgreen' if mode=='Bull' else 'background-color: lightcoral'
            if 70 <= pct < 90:
                return 'background-color: orange'
            return ''

        try:
            styled = df.style.applymap(highlight_score, subset=['Score (%)'])
            st.dataframe(styled)
        except KeyError:
            st.dataframe(df)






