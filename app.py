import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Overview Dashboard", layout="wide")

# =========================
# CSS
# =========================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.section-title {font-weight: 900; font-size: 1.15rem; margin: 0.65rem 0 0.4rem 0;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 10px;
}
.pill {
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.85rem;
}
.pill-green {background: rgba(80, 200, 120, 0.25); color: #B7FFCB; border: 1px solid rgba(80,200,120,0.45);}
.pill-amber {background: rgba(255, 193, 7, 0.18); color: #FFE8A6; border: 1px solid rgba(255,193,7,0.40);}
.pill-red   {background: rgba(255, 99, 132, 0.18); color: #FFB3C2; border: 1px solid rgba(255,99,132,0.40);}

.badge {
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.82rem;
}
.badge-yes {background: rgba(80, 200, 120, 0.22); color: #B7FFCB; border: 1px solid rgba(80,200,120,0.40);}
.badge-neutral {background: rgba(255, 193, 7, 0.16); color: #FFE8A6; border: 1px solid rgba(255,193,7,0.35);}
.badge-no  {background: rgba(255, 99, 132, 0.16); color: #FFB3C2; border: 1px solid rgba(255,99,132,0.35);}

[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BENCHMARK = "SPY"
PRICE_HISTORY_PERIOD = "2y"

def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ============================================================
# YOUR TICKER LIST (1 per line)
# ============================================================
TICKERS_RAW = r"""
SPY
QQQ
DIA
IWM
RSP
QQQE
EDOW
MDY
IWN
IWO
XLC
XLY
XLP
XLE
XLF
XLV
XLI
XLB
XLRE
XLK
XLU
SOXX
SMH
XSD
IGV
XSW
IGM
VGT
XT
CIBR
BOTZ
AIQ
XTL
VOX
FCOM
FDN
SOCL
XRT
IBUY
CARZ
IDRV
ITB
XHB
PEJ
VDC
FSTA
KXI
PBJ
VPU
FUTY
IDU
IYE
VDE
XOP
IEO
OIH
IXC
IBB
XBI
PBE
IDNA
IHI
XHE
XHS
XPH
FHLC
PINK
KBE
KRE
IAT
KIE
IAI
KCE
IYG
VFH
ITA
PPA
XAR
IYT
XTN
VIS
FIDU
XME
GDX
SIL
SLX
PICK
VAW
VNQ
IYR
REET
SRVR
HOMZ
SCHH
NETL
GLD
SLV
UNG
USO
DBA
CORN
DBB
PALL
URA
UGA
CPER
CATL
HOGS
SOYB
WEAT
DBC
IEMG
EUE
C6E
FEZ
E40
DAX
ISF
FXI
EEM
EWJ
EWU
EWZ
EWG
EWT
EWH
EWI
EWW
PIN
IDX
EWY
EWA
EWM
EWS
EWC
EWP
EZA
EWL
UUP
FXE
FXY
FXB
FXA
FXF
FXC
IBIT
ETHA
TLT
BND
SHY
IEF
SGOV
IEI
TLH
AGG
MUB
GOVT
IGSB
USHY
IGIB
""".strip()

def parse_ticker_list(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        t = ln.strip().upper()
        if t:
            out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq

ALL_TICKERS = parse_ticker_list(TICKERS_RAW)
ALL_TICKERS_SET = set(ALL_TICKERS)

MAJOR = ALL_TICKERS[:10]
SECTORS = ALL_TICKERS[10:21]

SUBSECTOR_LEFT = {
    "Semiconductors": ["SOXX","SMH","XSD"],
    "Software / Cloud / Broad Tech": ["IGV","XSW","IGM","VGT","XT"],
    "Cyber Security": ["CIBR"],
    "AI / Robotics / Automation": ["BOTZ","AIQ"],
    "Telecom & Communication": ["XTL","VOX","FCOM"],
    "Internet / Media / Social": ["FDN","SOCL"],
    "Retail": ["XRT","IBUY"],
    "Autos / EV": ["IDRV","CARZ"],
    "Homebuilders / Construction": ["ITB","XHB"],
    "Leisure & Entertainment": ["PEJ"],
    "Consumer Staples": ["VDC","FSTA","KXI","PBJ"],
    "Utilities": ["VPU","FUTY","IDU"],
    "Energy": ["IYE","VDE"],
    "Exploration & Production": ["XOP","IEO"],
    "Oil Services": ["OIH"],
    "Global Energy": ["IXC"],
}

SUBSECTOR_RIGHT = {
    "Biotechnology / Genomics": ["IBB","XBI","PBE","IDNA"],
    "Medical Equipment": ["IHI","XHE"],
    "Health Care Providers / Services": ["XHS"],
    "Pharmaceuticals": ["XPH"],
    "Broad / Alternative Health": ["FHLC","PINK"],
    "Banks": ["KBE","KRE","IAT"],
    "Insurance": ["KIE"],
    "Capital Markets / Brokerage": ["IAI","KCE"],
    "Diversified Financial Services": ["IYG"],
    "Broad Financials": ["VFH"],
    "Aerospace & Defense": ["ITA","PPA","XAR"],
    "Transportation": ["IYT","XTN"],
    "Broad Industrials": ["VIS","FIDU"],
    "Materials": ["XME","GDX","SIL","SLX","PICK","VAW"],
    "Real Estate": ["VNQ","IYR","REET"],
    "Specialty REITs": ["SRVR","HOMZ","SCHH","NETL"],
}

SUBSECTOR_ALL = {}
SUBSECTOR_ALL.update(SUBSECTOR_LEFT)
SUBSECTOR_ALL.update(SUBSECTOR_RIGHT)

# -----------------------------
# Data pulls
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_prices(tickers, period=PRICE_HISTORY_PERIOD):
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from price source.")

    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        close_df = pd.DataFrame(closes)
    else:
        close_df = pd.DataFrame({tickers[0]: df["Close"]})

    return close_df.dropna(how="all").ffill()

@st.cache_data(show_spinner=False, ttl=24*60*60)
def fetch_names(tickers: list[str]) -> dict[str, str]:
    names = {t: t for t in tickers}
    for t in tickers:
        try:
            inf = yf.Ticker(t).info
            n = inf.get("shortName") or inf.get("longName")
            if n:
                names[t] = str(n)
        except Exception:
            pass
    names["SPY"] = "S&P 500"
    names["QQQ"] = "Nasdaq-100"
    names["DIA"] = "Dow"
    names["IWM"] = "Russell 2000"
    names["RSP"] = "S&P 500 EW"
    return names

# -----------------------------
# Sparkline (kept as-is)
# -----------------------------
SPARK_CHARS = "▁▂▃▄▅▆▇█"
SPARK_MAP = {c: i for i, c in enumerate(SPARK_CHARS)}

def sparkline_from_series(s: pd.Series, n=26) -> str:
    s = s.dropna().tail(n)
    if s.empty:
        return ""
    if s.nunique() == 1:
        return SPARK_CHARS[len(SPARK_CHARS)//2] * len(s)

    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return ""
    scaled = (s - lo) / (hi - lo)
    idx = (scaled * (len(SPARK_CHARS)-1)).round().astype(int).clip(0, len(SPARK_CHARS)-1)
    return "".join(SPARK_CHARS[i] for i in idx)

def spark_strength(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return np.nan
    vals = [SPARK_MAP.get(ch, np.nan) for ch in s]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals) / (len(SPARK_CHARS) - 1))

# -----------------------------
# Metrics / Table
# -----------------------------
def _ret(close: pd.Series, periods: int):
    return close.pct_change(periods=periods)

def _ratio_rs(close_t: pd.Series, close_b: pd.Series, periods: int):
    # FIX: use close_b not undefined b
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    return (t / b) - 1

def build_table(p: pd.DataFrame, tickers: list[str], name_map: dict[str, str]) -> pd.DataFrame:
    horizons_ret = {"% 1D": 1, "% 1W": 5, "% 1M": 21, "% 3M": 63, "% 6M": 126, "% 1Y": 252}
    horizons_rs = {"RS 1W": 5, "RS 1M": 21, "RS 3M": 63, "RS 6M": 126, "RS 1Y": 252}

    b = p[BENCHMARK]

    rows = []
    for t in tickers:
        if t not in p.columns:
            continue

        close = p[t]
        last_price = float(close.dropna().iloc[-1]) if close.dropna().shape[0] else np.nan

        rs_ratio_series = (close / close.shift(21)) / (b / b.shift(21))
        spark = sparkline_from_series(rs_ratio_series, n=26)

        rec = {
            "Ticker": t,
            "Name": name_map.get(t, t),
            "Price": last_price,
            "Relative Strength 1M": spark,
        }

        for col, n in horizons_rs.items():
            rr = _ratio_rs(close, b, n)
            rec[col] = float(rr.dropna().iloc[-1]) if rr.dropna().shape[0] else np.nan

        for col, n in horizons_ret.items():
            r = _ret(close, n)
            rec[col] = float(r.dropna().iloc[-1]) if r.dropna().shape[0] else np.nan

        rows.append(rec)

    df = pd.DataFrame(rows)

    for col in horizons_rs.keys():
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = (s.rank(pct=True) * 99).round().clip(1, 99)

    return df

# -----------------------------
# Color helpers
# -----------------------------
def _heat_rs(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    x = (v - 1) / 98.0
    if x < 0.5:
        r = 255
        g = int(80 + (x/0.5) * (180-80))
    else:
        r = int(255 - ((x-0.5)/0.5) * (255-40))
        g = 200
    b = 60
    return f"background-color: rgb({r},{g},{b}); color:#0B0B0B; font-weight:900;"

def _pct_text(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    if v > 0:
        return "color: #7CFC9A; font-weight: 800;"
    if v < 0:
        return "color: #FF6B6B; font-weight: 800;"
    return "opacity:0.9; font-weight:700;"

def _spark_heat(s):
    strength = spark_strength(s)
    if np.isnan(strength):
        return ""
    x = float(strength)
    if x < 0.5:
        r = 255
        g = int(80 + (x/0.5) * (180-80))
    else:
        r = int(255 - ((x-0.5)/0.5) * (255-40))
        g = 200
    b = 60
    return f"background-color: rgba({r},{g},{b},0.0); color: #FFFFFF; font-weight: 900;"

def style_df(df: pd.DataFrame):
    fmt = {
        "Price": "${:,.2f}",
        "% 1D": "{:.2%}", "% 1W": "{:.2%}", "% 1M": "{:.2%}",
        "% 3M": "{:.2%}", "% 6M": "{:.2%}", "% 1Y": "{:.2%}",
        "RS 1W": "{:.0f}", "RS 1M": "{:.0f}", "RS 3M": "{:.0f}", "RS 6M": "{:.0f}", "RS 1Y": "{:.0f}",
    }

    rs_cols = ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"]
    pct_cols = ["% 1D", "% 1W", "% 1M", "% 3M", "% 6M", "% 1Y"]

    sty = df.style.format(fmt, na_rep="")

    for c in rs_cols:
        if c in df.columns:
            sty = sty.applymap(_heat_rs, subset=[c])

    for c in pct_cols:
        if c in df.columns:
            sty = sty.applymap(_pct_text, subset=[c])

    if "Relative Strength 1M" in df.columns:
        # keep monospace for sparkline readability; no background fill
        sty = sty.applymap(_spark_heat, subset=["Relative Strength 1M"])
        sty = sty.set_properties(
            subset=["Relative Strength 1M"],
            **{"font-family": "monospace", "font-weight": "900"}
        )

    return sty

# -----------------------------
# Sub-sector grouped headers
# -----------------------------
def grouped_block(groups: dict[str, list[str]], df_by_ticker: dict[str, dict]) -> pd.DataFrame:
    out_rows = []
    for group_name, ticks in groups.items():
        out_rows.append({
            "Ticker": "",
            "Name": group_name,
            "Price": np.nan,
            "Relative Strength 1M": "",
            "RS 1W": np.nan, "RS 1M": np.nan, "RS 3M": np.nan, "RS 6M": np.nan, "RS 1Y": np.nan,
            "% 1D": np.nan, "% 1W": np.nan, "% 1M": np.nan, "% 3M": np.nan, "% 6M": np.nan, "% 1Y": np.nan,
        })
        for t in ticks:
            if t in df_by_ticker:
                out_rows.append(df_by_ticker[t])
    return pd.DataFrame(out_rows)

def style_grouped(df: pd.DataFrame):
    sty = style_df(df)

    def _header_row_styles(row):
        is_header = (str(row.get("Ticker", "")).strip() == "") and (str(row.get("Name", "")).strip() != "")
        if is_header:
            return ["font-weight:950; background-color: rgba(0,0,0,0.55);" for _ in row.index]
        return ["" for _ in row.index]

    return sty.apply(_header_row_styles, axis=1)

# ===========================
# Manual Inputs (UPDATED)
# ===========================
DEFAULT_RIGHT = {
    "Market Exposure": {"IBD Exposure": "40-60%", "Selected": "X"},
    "Market Type": {"Type": "Bull Quiet"},
    "Trend Condition (QQQ)": {
        "Above 5DMA": "Yes",
        "Above 10DMA": "Yes",
        "Above 20DMA": "Yes",
        "Above 50DMA": "Yes",
        "Above 200DMA": "No",
    },
    "NASDAQ Net 52W New High/Low": {"Daily": 231, "Weekly": 811, "Monthly": -828},
    "Market Indicators": {
        "VIX": 16.34,
        "PCC": 0.67,
        "Up/Down Vol Ratio": 2.36,
        "A/D Ratio": 2.20,
        "U.S. Dollar Trend": "Downtrend",   # dropdown
    },
    "Macro": {
        "Fed Funds": 4.09,
        "M2 Money": 22.2,
        "10yr": 4.02,
        "U.S. Dollar (DXY)": 98.72,         # numeric
    },
    "Breadth & Participation": {
        "% Price Above 10DMA": 56,
        "% Price Above 20DMA": 49,
        "% Price Above 50DMA": 58,
        "% Price Above 200DMA": 68,
    },
    # Composite now uses NUMERIC factors (0.0–2.0 step 0.5) and auto-total
    "Composite Model": {
        "Monetary Policy (Fed Funds)": 1.5,
        "Liquidity Flow (M2 + Credit)": 2.0,
        "Rates & Credit (10yr + HYG/IEI)": 2.0,
        "Tape Strength (Breadth + U/D + Net High/Lows)": 1.5,
        "Sentiment (VIX + Put/Call)": 1.5,
    },
    "Hot Sectors / Industry Groups": {"Notes": "Type here..."},
    "Market Correlations": {"Correlated": "Dow, Nasdaq", "Uncorrelated": "Dollar, Bonds"},
}

YES_NO = ["Yes", "No"]
USD_TREND = ["Uptrend", "Downtrend", "Sideways"]
MARKET_TYPES = ["Bull Quiet", "Bull Volatile", "Bear Quiet", "Bear Volatile", "Sideways Quiet", "Sideways Volatile"]

def init_right_state():
    if "right_panel" not in st.session_state:
        st.session_state.right_panel = DEFAULT_RIGHT

def _score_to_label(score: float):
    """
    Composite component score mapping:
      Bad:     0.0–0.5
      Neutral: 1.0
      Good:    1.5–2.0
    """
    try:
        score = float(score)
    except:
        return ("Neutral", "badge badge-neutral")

    if score >= 1.5:
        return ("Good", "badge badge-yes")
    if score <= 0.5:
        return ("Bad", "badge badge-no")
    return ("Neutral", "badge badge-neutral")

def _total_score_pill_class(total: float) -> str:
    """
    Total score mapping (0–10):
      Green: 7.0–10.0
      Amber: 5.0–6.5
      Red:   0.0–4.5
    """
    try:
        total = float(total)
    except:
        return "pill pill-amber"

    if total >= 7.0:
        return "pill pill-green"
    if total >= 5.0:
        return "pill pill-amber"
    return "pill pill-red"

def right_panel_ui():
    init_right_state()
    rp = st.session_state.right_panel

    st.markdown(
        '<div class="card"><div style="font-weight:900; font-size:1.0rem;">Manual Inputs</div>'
        '<div class="small-muted">You update only these. Everything else pulls & calculates automatically.</div></div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "Download settings.json",
        data=json.dumps(rp, indent=2),
        file_name="dashboard_settings.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("Import settings JSON (optional)", type=["json"])
    if up is not None:
        try:
            st.session_state.right_panel = json.loads(up.read().decode("utf-8"))
            rp = st.session_state.right_panel
            st.success("Imported settings.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # --- Render blocks with per-field widgets (so we can do true dropdowns + numeric inputs) ---
    for block, data in rp.items():
        st.markdown(f'<div class="card"><div style="font-weight:900; margin-bottom:8px;">{block}</div>', unsafe_allow_html=True)

        if block == "Market Exposure":
            rp[block]["IBD Exposure"] = st.selectbox(
                "IBD Exposure",
                options=["80-100%", "60-80%", "40-60%", "20-40%", "0-20%"],
                index=max(0, ["80-100%","60-80%","40-60%","20-40%","0-20%"].index(str(data.get("IBD Exposure","40-60%"))) if str(data.get("IBD Exposure","40-60%")) in ["80-100%","60-80%","40-60%","20-40%","0-20%"] else 2),
                key=f"me_exposure"
            )
            rp[block]["Selected"] = st.text_input("Selected (leave X if you want)", value=str(data.get("Selected", "X")), key="me_selected")

        elif block == "Market Type":
            current = str(data.get("Type", "Bull Quiet"))
            idx = MARKET_TYPES.index(current) if current in MARKET_TYPES else 0
            rp[block]["Type"] = st.selectbox("Type", options=MARKET_TYPES, index=idx, key="mt_type")

        elif block == "Trend Condition (QQQ)":
            for k in ["Above 5DMA","Above 10DMA","Above 20DMA","Above 50DMA","Above 200DMA"]:
                cur = str(data.get(k, "No"))
                idx = YES_NO.index(cur) if cur in YES_NO else 1
                rp[block][k] = st.selectbox(k, options=YES_NO, index=idx, key=f"tc_{k}")

        elif block == "NASDAQ Net 52W New High/Low":
            rp[block]["Daily"] = st.number_input("Daily", value=float(data.get("Daily", 0)), step=1.0, key="nhl_daily")
            rp[block]["Weekly"] = st.number_input("Weekly", value=float(data.get("Weekly", 0)), step=1.0, key="nhl_weekly")
            rp[block]["Monthly"] = st.number_input("Monthly", value=float(data.get("Monthly", 0)), step=1.0, key="nhl_monthly")

        elif block == "Market Indicators":
            rp[block]["VIX"] = st.number_input("VIX", value=float(data.get("VIX", 0.0)), step=0.01, format="%.2f", key="mi_vix")
            rp[block]["PCC"] = st.number_input("PCC", value=float(data.get("PCC", 0.0)), step=0.01, format="%.2f", key="mi_pcc")
            rp[block]["Up/Down Vol Ratio"] = st.number_input("Up/Down Vol Ratio", value=float(data.get("Up/Down Vol Ratio", 0.0)), step=0.01, format="%.2f", key="mi_udvr")
            rp[block]["A/D Ratio"] = st.number_input("A/D Ratio", value=float(data.get("A/D Ratio", 0.0)), step=0.01, format="%.2f", key="mi_adr")

            cur = str(data.get("U.S. Dollar Trend", "Downtrend"))
            idx = USD_TREND.index(cur) if cur in USD_TREND else 1
            rp[block]["U.S. Dollar Trend"] = st.selectbox("U.S. Dollar Trend", options=USD_TREND, index=idx, key="mi_usd_trend")

        elif block == "Macro":
            rp[block]["Fed Funds"] = st.number_input("Fed Funds", value=float(data.get("Fed Funds", 0.0)), step=0.01, format="%.2f", key="mac_ff")
            rp[block]["M2 Money"] = st.number_input("M2 Money", value=float(data.get("M2 Money", 0.0)), step=0.1, format="%.1f", key="mac_m2")
            rp[block]["10yr"] = st.number_input("10yr", value=float(data.get("10yr", 0.0)), step=0.01, format="%.2f", key="mac_10y")
            rp[block]["U.S. Dollar (DXY)"] = st.number_input("U.S. Dollar (DXY)", value=float(data.get("U.S. Dollar (DXY)", 0.0)), step=0.01, format="%.2f", key="mac_dxy")

        elif block == "Breadth & Participation":
            rp[block]["% Price Above 10DMA"] = st.number_input("% Price Above 10DMA", value=float(data.get("% Price Above 10DMA", 0)), step=1.0, key="br_10")
            rp[block]["% Price Above 20DMA"] = st.number_input("% Price Above 20DMA", value=float(data.get("% Price Above 20DMA", 0)), step=1.0, key="br_20")
            rp[block]["% Price Above 50DMA"] = st.number_input("% Price Above 50DMA", value=float(data.get("% Price Above 50DMA", 0)), step=1.0, key="br_50")
            rp[block]["% Price Above 200DMA"] = st.number_input("% Price Above 200DMA", value=float(data.get("% Price Above 200DMA", 0)), step=1.0, key="br_200")

        elif block == "Composite Model":
            st.caption("Score each factor (0.0–2.0). Bad: 0.0–0.5 • Neutral: 1.0 • Good: 1.5–2.0")
            total = 0.0
            for factor in [
                "Monetary Policy (Fed Funds)",
                "Liquidity Flow (M2 + Credit)",
                "Rates & Credit (10yr + HYG/IEI)",
                "Tape Strength (Breadth + U/D + Net High/Lows)",
                "Sentiment (VIX + Put/Call)",
            ]:
                val = float(data.get(factor, 1.0))
                val = st.slider(factor, min_value=0.0, max_value=2.0, step=0.5, value=val, key=f"cm_{factor}")
                rp[block][factor] = val
                total += val
                label, cls = _score_to_label(val)
                st.markdown(f'<div style="margin:4px 0 10px 0;">'
                            f'<span class="{cls}">{label}</span> '
                            f'<span class="small-muted">({val:.1f})</span></div>', unsafe_allow_html=True)

            total_cls = _total_score_pill_class(total)
            st.markdown(
                f'<div style="margin-top:10px;"><b>Total Score:</b> <span class="{total_cls}">{total:.1f}</span> '
                f'<span class="small-muted">/ 10.0</span></div>',
                unsafe_allow_html=True
            )

        elif block == "Hot Sectors / Industry Groups":
            rp[block]["Notes"] = st.text_area("Notes", value=str(data.get("Notes", "")), height=90, key="hot_notes")

        elif block == "Market Correlations":
            rp[block]["Correlated"] = st.text_input("Correlated", value=str(data.get("Correlated","")), key="corr_corr")
            rp[block]["Uncorrelated"] = st.text_input("Uncorrelated", value=str(data.get("Uncorrelated","")), key="corr_uncorr")

        else:
            # fallback
            for k, v in data.items():
                rp[block][k] = st.text_input(k, value=str(v), key=f"misc_{block}_{k}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.right_panel = rp

# =========================
# UI
# =========================
st.title("Market Overview Dashboard")
st.caption(f"As of: {_asof_ts()} • Auto data: Yahoo Finance • RS Benchmark: {BENCHMARK}")

with st.sidebar:
    st.subheader("Controls")
    if st.button("Refresh Data"):
        fetch_prices.clear()
        fetch_names.clear()
        st.rerun()

pull_list = list(dict.fromkeys(ALL_TICKERS + [BENCHMARK]))

try:
    price_df = fetch_prices(pull_list, period=PRICE_HISTORY_PERIOD)
except Exception as e:
    st.error(f"Data pull failed: {e}")
    st.stop()

name_map = fetch_names(pull_list)

df_major = build_table(price_df, MAJOR, name_map)
df_sectors = build_table(price_df, SECTORS, name_map)

all_sub_ticks = []
for v in list(SUBSECTOR_ALL.values()):
    all_sub_ticks.extend(v)
all_sub_ticks = [t for t in all_sub_ticks if t in ALL_TICKERS_SET]

df_sub_master = build_table(price_df, all_sub_ticks, name_map)
df_by_ticker = {r["Ticker"]: r.to_dict() for _, r in df_sub_master.iterrows()}
df_sub_all = grouped_block(SUBSECTOR_ALL, df_by_ticker)

show_cols = [
    "Ticker", "Name", "Price", "Relative Strength 1M",
    "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y",
    "% 1D", "% 1W", "% 1M", "% 3M", "% 6M", "% 1Y"
]

st.markdown('<div class="section-title">Major U.S. Indexes</div>', unsafe_allow_html=True)
st.dataframe(
    style_df(df_major[show_cols]),
    use_container_width=True,
    height=330,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Name": st.column_config.TextColumn(width="medium"),
        "Relative Strength 1M": st.column_config.TextColumn(width="large"),
    },
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">U.S. Sectors</div>', unsafe_allow_html=True)
st.dataframe(
    style_df(df_sectors[show_cols]),
    use_container_width=True,
    height=360,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Name": st.column_config.TextColumn(width="medium"),
        "Relative Strength 1M": st.column_config.TextColumn(width="large"),
    },
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">U.S. Sub-Sectors / Industry Groups</div>', unsafe_allow_html=True)
st.dataframe(
    style_grouped(df_sub_all[show_cols]),
    use_container_width=True,
    height=1100,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Name": st.column_config.TextColumn(width="large"),
        "Relative Strength 1M": st.column_config.TextColumn(width="large"),
    },
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

right_panel_ui()
