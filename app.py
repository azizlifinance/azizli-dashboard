# app.py  —  Azizli Software Inc. • Stock Dashboard (web)
# Web version of your desktop app:
# - Up to 10 tickers (with checkboxes for chart overlay)
# - Table: Ticker, Name, Price, $Change, %Change (+ optional Basis, As Of)
# - Colors on Price / $Change / %Change
# - Chart panel (optional): 1D, 5D, 1M, 6M, YTD, 1Y, 5Y, All
# - Intraday (1D/5D/1M): RTH only (9:30–16:00 ET)
# - Overlay multiple tickers, optional Normalize (%)

from datetime import datetime, date, timedelta, time as dtime
import pytz, numpy as np, pandas as pd, yfinance as yf, pandas_market_calendars as mcal
import streamlit as st
import plotly.graph_objects as go

ET = pytz.timezone("America/New_York")

# ---------- Market helpers ----------

def to_et(ts):
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = pytz.UTC.localize(ts)
    return ts.astimezone(ET)

@st.cache_data(ttl=3600)
def nyse_calendar():
    return mcal.get_calendar("XNYS")

def nyse_sched_for(d: date):
    cal = nyse_calendar()
    s = cal.schedule(start_date=d, end_date=d)
    if s.empty:
        return None, None
    return s.iloc[0]["market_open"], s.iloc[0]["market_close"]

def is_market_open_now():
    now_utc = datetime.now(pytz.UTC)
    o, c = nyse_sched_for(now_utc.date())
    if o is None:
        return False, None, None
    return o <= now_utc <= c, o, c

def previous_trading_day(ref_date: date) -> date:
    cal = nyse_calendar()
    start = ref_date - timedelta(days=40)
    sched = cal.schedule(start_date=start, end_date=ref_date)
    if sched.empty:
        return ref_date - timedelta(days=1)
    days = [idx.date() for idx in sched.index]
    if days and days[-1] == ref_date:
        days.pop()
    return days[-1] if days else (ref_date - timedelta(days=1))

@st.cache_data(ttl=300)
def last_close_on_or_before(ticker: str, target_d: date):
    start = (pd.Timestamp(target_d) - pd.tseries.offsets.BDay(20)).date()
    end = target_d + timedelta(days=2)
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No daily data for {ticker}")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.tz_convert(None)
    dates = [ts.date() for ts in df.index]
    idx = None
    for i in range(len(dates)-1, -1, -1):
        if dates[i] <= target_d:
            idx = i; break
    if idx is None:
        raise ValueError(f"No close on/before {target_d} for {ticker}")
    ts = df.index[idx]
    px = float(df["Close"].iloc[idx])
    o_utc, c_utc = nyse_sched_for(ts.date())
    return px, ts, o_utc, c_utc  # session open/close are tz-aware (UTC)

@st.cache_data(ttl=120)
def intraday_last_1m(ticker: str):
    df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=False, prepost=False)
    if df.empty:
        raise ValueError(f"No intraday data for {ticker}")
    return float(df["Close"].iloc[-1]), df.index[-1]

# ---------- Names (cached) ----------
_name_cache = {}

@st.cache_data(ttl=86400)
def get_company_name(ticker: str) -> str:
    t = ticker.upper()
    if t in _name_cache:
        return _name_cache[t]
    name = t
    try:
        info = yf.Ticker(t).get_info()
        name = info.get("longName") or info.get("shortName") or t
    except Exception:
        name = t
    _name_cache[t] = name
    return name

# ---------- Table compute ----------
def fetch_price_and_change(ticker: str, mode: str, picked_date: date):
    t = ticker.strip().upper()
    if not t:
        raise ValueError("Empty ticker")

    if mode == "Current (today logic)":
        today_et = datetime.now(ET).date()
        open_now, _, _ = is_market_open_now()
        if open_now:
            price, asof_ts = intraday_last_1m(t)  # true tick time
            basis = "current (last 1-min)"
            trading_dt_for_prev = to_et(asof_ts).date()
        else:
            price, daily_ts, _, close_utc = last_close_on_or_before(t, today_et)
            asof_ts = close_utc                    # 16:00 ET official close
            basis = "official daily close"
            trading_dt_for_prev = daily_ts.date()
    else:
        price, daily_ts, _, close_utc = last_close_on_or_before(t, picked_date)
        asof_ts = close_utc                        # 16:00 ET official close
        basis = "official daily close"
        trading_dt_for_prev = daily_ts.date()

    prev_day = previous_trading_day(trading_dt_for_prev)
    prev_close, _, _, _ = last_close_on_or_before(t, prev_day)

    change = price - prev_close
    pct = (change / prev_close * 100.0) if prev_close else 0.0

    return {
        "ticker": t,
        "name": get_company_name(t),
        "price": round(price, 2),
        "change": round(change, 2),
        "pct": round(pct, 2),
        "basis": basis,
        "as_of": to_et(asof_ts).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "pos": change > 0,
        "neg": change < 0,
    }

# ---------- Chart helpers ----------
def range_params(range_key: str):
    rk = range_key.upper()
    intraday = rk in {"1D","5D","1M"}
    if rk == "1D":   return {"period":"1d","interval":"1m"}, True
    if rk == "5D":   return {"period":"5d","interval":"5m"}, True
    if rk == "1M":   return {"period":"1mo","interval":"30m"}, True
    if rk == "6M":   return {"period":"6mo","interval":"1d"}, False
    if rk == "YTD":
        now_et = datetime.now(ET)
        start = datetime(now_et.year, 1, 1).date().isoformat()
        return {"start": start, "interval":"1d"}, False
    if rk == "1Y":   return {"period":"1y","interval":"1d"}, False
    if rk == "5Y":   return {"period":"5y","interval":"1wk"}, False
    return {"period":"max","interval":"1mo"}, False

@st.cache_data(ttl=300)
def fetch_series_for_chart(ticker: str, range_key: str) -> pd.Series:
    params, _ = range_params(range_key)
    df = yf.download(ticker, progress=False, auto_adjust=False, **params)
    if df.empty:
        raise ValueError(f"No chart data for {ticker} ({range_key})")
    s = df.get("Close")
    if s is None:
        raise ValueError(f"No Close column for {ticker} ({range_key})")
    if isinstance(s, pd.DataFrame):
        s = s.squeeze("columns")
    return s.dropna()

def filter_to_rth_et(s: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(s.index)
    idx = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(ET)
    s = pd.Series(np.asarray(s.values).reshape(-1), index=idx)
    mask = (idx.time >= dtime(9,30)) & (idx.time <= dtime(16,0))
    return s[mask]

# ---------- UI ----------
st.set_page_config(page_title="Stock Dashboard — Azizli Software Inc.", layout="wide")
st.title("Stock Dashboard — Azizli Software Inc.")

with st.form("controls"):
    top1, top2, top3 = st.columns([2,2,3])

    with top1:
        mode = st.radio("Mode", ["Current (today logic)", "Single Date for all"], horizontal=True)
    with top2:
        date_pick = st.date_input("Date (ET)", value=date.today())
        show_details = st.checkbox("Show more details", value=False)
    with top3:
        show_chart = st.checkbox("Charts", value=False)
        rng = st.radio("Range", ["1D","5D","1M","6M","YTD","1Y","5Y","All"], horizontal=True, index=1,
                       disabled=not show_chart)
        normalize = st.checkbox("Normalize (%)", value=False, disabled=not show_chart)

    st.markdown("**Tickers (up to 10) — check to include in chart**")

    rows = []
    for r in range(2):
        cols = st.columns([1,4,2, 1,4,2, 1,4,2, 1,4,2, 1,4,2])  # label, input, checkbox × 5
        for i in range(5):
            idx = r*5 + i
            cols[i*3+0].markdown(f"{idx+1}.")
            t = cols[i*3+1].text_input("", value=["AAPL","MSFT","GOOGL","AMZN","META",
                                                  "COP","OXY","XOM","BP","APA"][idx], key=f"t{idx}")
            sel = cols[i*3+2].checkbox("Sel", value=(idx<5), key=f"s{idx}", disabled=not show_chart)
            rows.append((t.strip().upper(), sel))

    b1, b2 = st.columns([1,1])
    submit = b1.form_submit_button("Get Prices")
    plot = b2.form_submit_button("Plot", disabled=not show_chart)

# ---------- Table ----------
if submit:
    data = []
    for t, _ in rows:
        if not t:
            continue
        try:
            info = fetch_price_and_change(t, mode, date_pick)
            data.append(info)
        except Exception as e:
            data.append({"ticker": t, "name": f"Error: {e}", "price": None, "change": None,
                         "pct": None, "basis": "", "as_of": "", "pos": False, "neg": False})
    if not data:
        st.info("Please enter at least one ticker.")
    else:
        # Build a colored Plotly table
        show_cols = ["ticker","name","price","change","pct"] + (["basis","as_of"] if show_details else [])
        headers = {
            "ticker": "Ticker", "name": "Name", "price": "Price",
            "change": "Change", "pct": "%Change", "basis": "Basis", "as_of": "As Of"
        }
        # Values per column
        col_vals = {k: [] for k in show_cols}
        col_colors = {k: [] for k in show_cols}  # font colors
        for row in data:
            # format numbers
            price_txt = f"${row['price']:.2f}" if row["price"] is not None else "—"
            chg_txt   = f"${row['change']:.2f}" if row["change"] is not None else "—"
            pct_txt   = f"{row['pct']:.2f}%" if row["pct"] is not None else "—"
            # push values
            mapping = {
                "ticker": row["ticker"],
                "name": row["name"],
                "price": price_txt,
                "change": chg_txt,
                "pct": pct_txt,
                "basis": row["basis"],
                "as_of": row["as_of"],
            }
            # colors
            col = "green" if row["pos"] else ("red" if row["neg"] else "black")
            for k in show_cols:
                col_vals[k].append(mapping[k])
                if k in ("price","change","pct"):
                    col_colors[k].append(col)
                else:
                    col_colors[k].append("black")

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=[headers[k] for k in show_cols],
                        fill_color="#f0f2f6", align="left"),
            cells=dict(values=[col_vals[k] for k in show_cols],
                       align="left",
                       font=dict(color=[col_colors[k] for k in show_cols]))
        )])
        st.plotly_chart(fig_table, use_container_width=True, config={"displayModeBar": False})

# ---------- Chart ----------
if plot and show_chart:
    selected = [t for (t, sel) in rows if sel and t]
    if not selected:
        st.info("Check at least one ticker to plot.")
    else:
        params, is_intraday = range_params(rng)
        fig = go.Figure()
        if is_intraday:
            # Intraday: filter to RTH and (optionally) normalize; keep datetime axis (RTH only)
            for t in selected:
                try:
                    s = fetch_series_for_chart(t, rng)
                    s = filter_to_rth_et(s)
                    if s.empty:
                        continue
                    y = s.values.astype(float)
                    if normalize:
                        y = (y / y[0] - 1.0) * 100.0
                        y_title = "% since start"
                    else:
                        y_title = "Price"
                    fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines", name=t))
                except Exception as e:
                    st.warning(f"{t}: {e}")
            fig.update_layout(
                title=f"{' · '.join(selected)} — {rng}{' (normalized)' if normalize else ''}",
                xaxis_title="ET (RTH only)",
                yaxis_title=("%" if normalize else "Price"),
                hovermode="x unified"
            )
        else:
            # Daily+ ranges
            for t in selected:
                try:
                    s = fetch_series_for_chart(t, rng)
                    if s.empty:
                        continue
                    y = s.values.astype(float)
                    if normalize:
                        y = (y / y[0] - 1.0) * 100.0
                        y_title = "% since start"
                    else:
                        y_title = "Price"
                    # convert to ET for nice display (optional)
                    idx = pd.DatetimeIndex(s.index)
                    idx = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(ET)
                    fig.add_trace(go.Scatter(x=idx, y=y, mode="lines", name=t))
                except Exception as e:
                    st.warning(f"{t}: {e}")
            fig.update_layout(
                title=f"{' · '.join(selected)} — {rng}{' (normalized)' if normalize else ''}",
                xaxis_title="Date/Time",
                yaxis_title=("%" if normalize else "Price"),
                hovermode="x unified"
            )

        st.plotly_chart(fig, use_container_width=True)
