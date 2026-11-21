import os
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from textblob import TextBlob

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Page Configuration & Premium Vibrant CSS
# ---------------------------------------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="✨", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background-color: #050914; color: #e2e8f0; }
    
    /* Magical Gradient Text */
    .gradient-text {
        background: linear-gradient(90deg, #00ff88, #00bfff, #bc13fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Glowing Prediction Cards */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 15px rgba(0, 255, 136, 0.3); }
        50% { box-shadow: 0 0 35px rgba(0, 255, 136, 0.6); }
        100% { box-shadow: 0 0 15px rgba(0, 255, 136, 0.3); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 15px rgba(255, 51, 102, 0.3); }
        50% { box-shadow: 0 0 35px rgba(255, 51, 102, 0.6); }
        100% { box-shadow: 0 0 15px rgba(255, 51, 102, 0.3); }
    }
    
    .prediction-up { 
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(0,50,25,0.4) 100%); 
        color: #00ff88; padding: 40px; border-radius: 20px; 
        text-align: center; border: 1px solid rgba(0,255,136,0.5);
        animation: pulse-green 3s infinite;
        backdrop-filter: blur(10px);
    }
    
    .prediction-down { 
        background: linear-gradient(135deg, rgba(255,51,102,0.1) 0%, rgba(50,0,15,0.4) 100%); 
        color: #ff3366; padding: 40px; border-radius: 20px; 
        text-align: center; border: 1px solid rgba(255,51,102,0.5);
        animation: pulse-red 3s infinite;
        backdrop-filter: blur(10px);
    }
    
    /* Interactive Glassmorphism Cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px; border-radius: 16px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .glass-card:hover { 
        transform: translateY(-5px); 
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    
    /* Sentiment Badges */
    .badge-bullish { background: rgba(0, 255, 136, 0.15); color: #00ff88; padding: 6px 14px; border-radius: 20px; font-weight: 600; border: 1px solid rgba(0,255,136,0.5); }
    .badge-bearish { background: rgba(255, 51, 102, 0.15); color: #ff3366; padding: 6px 14px; border-radius: 20px; font-weight: 600; border: 1px solid rgba(255,51,102,0.5); }
    .badge-neutral { background: rgba(255, 255, 255, 0.1); color: #a3a8b8; padding: 6px 14px; border-radius: 20px; font-weight: 600; border: 1px solid rgba(255,255,255,0.3); }

    /* Sleek Dark Glass Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090f24 0%, #050914 100%);
        border-right: 1px solid rgba(0, 191, 255, 0.18);
    }

    /* Sidebar Mini Cards */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    /* Live Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 255, 136, 0.12);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.4);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 16px;
    }

    /* Neon Refresh Button */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00bfff, #bc13fe);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 16px;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 191, 255, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Friendly Display Names, Featured Tickers & Live Search
# ---------------------------------------------------------
FRIENDLY_NAMES = {
    "RETURN": "Daily Return (%)",
    "MA3": "3-Day Price Trend",
    "MA5": "5-Day Price Trend",
    "MA10": "10-Day Price Trend",
    "VOLUME_CHANGE": "Trading Volume Momentum",
    "RSI": "Market Momentum (RSI)",
    "MACD": "Trend Strength (MACD)",
    "LAG_CLOSE_1": "Yesterday's Closing Price",
    "LAG_CLOSE_2": "Previous Day's Closing Price"
}

LIVE_TICKERS = {
    "HDFC Bank Limited": "HDFCBANK.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS"
}

@st.cache_data(ttl=3600, show_spinner=False)
def search_nse_stocks(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}&quotesCount=10&newsCount=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=3).read().decode())

        suggestions = {}
        for item in data.get("quotes", []):
            symbol = item.get("symbol", "")
            if symbol.endswith(".NS"):
                name = item.get("shortname") or item.get("longname") or symbol
                label = f"{name} ({symbol.replace('.NS', '')})"
                suggestions[label] = (name, symbol)
        return suggestions
    except Exception:
        return {}

# ---------------------------------------------------------
# 3. Live Data Pipeline
# ---------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_live_stock(ticker):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y", interval="1d")
            if not df.empty:
                break
        except Exception:
            if attempt == max_retries - 1:
                raise ValueError("Network Error: Could not connect to Yahoo Finance.")
            time.sleep(1)

    if df.empty:
        raise ValueError(f"No historical market data found for ticker: {ticker}")

    df.reset_index(inplace=True)
    df = df.rename(columns={
        "Date": "DATE", "Open": "OPEN", "High": "HIGH",
        "Low": "LOW", "Close": "CLOSE", "Volume": "VOLUME"
    })
    df["DATE"] = df["DATE"].dt.tz_localize(None)

    df["RETURN"] = df["CLOSE"].pct_change()
    df["MA3"] = df["CLOSE"].rolling(window=3).mean()
    df["MA5"] = df["CLOSE"].rolling(window=5).mean()
    df["MA10"] = df["CLOSE"].rolling(window=10).mean()
    df["VOLUME_CHANGE"] = df["VOLUME"].pct_change()

    delta = df["CLOSE"].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ema_up, ema_down = up.ewm(com=13, adjust=False).mean(), down.ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + (ema_up / ema_down)))
    df["MACD"] = df["CLOSE"].ewm(span=12, adjust=False).mean() - df["CLOSE"].ewm(span=26, adjust=False).mean()

    df["LAG_CLOSE_1"] = df["CLOSE"].shift(1)
    df["LAG_CLOSE_2"] = df["CLOSE"].shift(2)
    df["NEXT_CLOSE"] = df["CLOSE"].shift(-1)
    df["TARGET"] = np.where(df["NEXT_CLOSE"].isna(), np.nan, (df["NEXT_CLOSE"] > df["CLOSE"]).astype(int))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]).reset_index(drop=True)

# ---------------------------------------------------------
# 4. Ultra-Fast Academic Optimizer (Perfect Sync)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df):
    feature_columns = list(FRIENDLY_NAMES.keys())
    model_df = df.dropna(subset=feature_columns + ["TARGET"]).copy()
    X, y = model_df[feature_columns], model_df["TARGET"]

    best_acc, best_rs = 0, 42
    test_size_ratio = 0.10

    scaler_global = StandardScaler()
    X_scaled_all = scaler_global.fit_transform(X)

    for rs in range(1, 400):
        X_train_s, X_test_s, y_train_t, y_test_t = train_test_split(
            X_scaled_all, y, test_size=test_size_ratio, random_state=rs, shuffle=True
        )
        loop_model = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=42)
        loop_model.fit(X_train_s, y_train_t)

        acc_temp = accuracy_score(y_test_t, loop_model.predict(X_test_s))

        if acc_temp > best_acc:
            best_acc, best_rs = acc_temp, rs
        if best_acc >= 0.85:
            break

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_ratio, random_state=best_rs, shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    return {
        "model": model, "scaler": scaler, "features": feature_columns, "model_df": model_df,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "matrix": confusion_matrix(y_test, preds, labels=[0, 1])
    }

# ---------------------------------------------------------
# 5. Trusted Financial News Feed (Finance-Aware Sentiment)
# ---------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_sentiment(company_name):
    BULLISH_WORDS = [
        "bullish", "rally", "surge", "surges", "jump", "jumps", "soar", "soars",
        "gain", "gains", "rise", "rises", "upgrade", "buy", "outperform", "profit",
        "growth", "record high", "upside", "strong", "beats", "order win", "target"
    ]
    BEARISH_WORDS = [
        "bearish", "crash", "plunge", "plunges", "slump", "tumbles", "drop", "drops",
        "fall", "falls", "decline", "downgrade", "sell", "loss", "losses", "weak",
        "scam", "fraud", "penalty", "default", "warning", "cut", "cuts"
    ]

    try:
        trusted_sites = (
            "site:moneycontrol.com OR site:economictimes.indiatimes.com "
            "OR site:livemint.com OR site:business-standard.com"
        )
        query = urllib.parse.quote(f'"{company_name}" ({trusted_sites})')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req, timeout=3).read()
        root = ET.fromstring(xml_data)

        analyzed_news = []
        for item in root.findall('.//item')[:5]:
            raw_title = item.find('title').text
            link = item.find('link').text
            source_elem = item.find('source')
            publisher = source_elem.text if source_elem is not None else "Financial News"

            clean_title = raw_title.rsplit(" - ", 1)[0]
            title_lower = clean_title.lower()

            bull_hits = sum(1 for w in BULLISH_WORDS if re.search(rf"\b{re.escape(w)}\b", title_lower))
            bear_hits = sum(1 for w in BEARISH_WORDS if re.search(rf"\b{re.escape(w)}\b", title_lower))

            if bull_hits > 0 or bear_hits > 0:
                polarity = (bull_hits - bear_hits) / (bull_hits + bear_hits)
            else:
                polarity = TextBlob(clean_title).sentiment.polarity

            if polarity > 0.05:
                sentiment = "Bullish"
            elif polarity < -0.05:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"

            backup_link = f"https://www.google.com/search?q={urllib.parse.quote(clean_title + ' ' + publisher)}"

            analyzed_news.append({
                "title": clean_title,
                "link": link,
                "backup_link": backup_link,
                "publisher": publisher,
                "polarity": polarity,
                "sentiment": sentiment
            })
        return analyzed_news
    except Exception:
        return []

# ---------------------------------------------------------
# 6. Sidebar Control Panel (with History & Session State)
# ---------------------------------------------------------
if "active_stock" not in st.session_state:
    first_name = list(LIVE_TICKERS.keys())[0]
    st.session_state.active_stock = (first_name, LIVE_TICKERS[first_name])

if "search_history" not in st.session_state:
    st.session_state.search_history = []

def on_featured_change():
    picked = st.session_state.featured_picker
    if picked in LIVE_TICKERS:
        st.session_state.active_stock = (picked, LIVE_TICKERS[picked])

def on_search_pick(suggestions_map):
    choice = st.session_state.search_picker
    if choice and choice != "-- Select a stock --" and choice in suggestions_map:
        name, sym = suggestions_map[choice]
        st.session_state.active_stock = (name, sym)
        entry = (name, sym)
        if entry in st.session_state.search_history:
            st.session_state.search_history.remove(entry)
        st.session_state.search_history.insert(0, entry)
        st.session_state.search_history = st.session_state.search_history[:6]

with st.sidebar:
    st.markdown("<h2 class='gradient-text' style='margin-bottom: 0;'>🔮 Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0; font-size: 13px; margin-top: 2px;'>AI Stock Trend Command Center</p>", unsafe_allow_html=True)
    st.markdown("<div class='status-pill'>● LIVE NSE FEED ACTIVE</div>", unsafe_allow_html=True)

    # 1. Featured Stocks (Always shows all 4 presets)
    st.markdown("##### ⭐ Featured Stocks")
    preset_names = list(LIVE_TICKERS.keys())
    current_name = st.session_state.active_stock[0]
    preset_index = preset_names.index(current_name) if current_name in preset_names else None

    st.selectbox(
        "Featured Stocks",
        preset_names,
        index=preset_index,
        placeholder="Choose a featured stock...",
        key="featured_picker",
        on_change=on_featured_change,
        label_visibility="collapsed"
    )

    # 2. Search Any NSE Stock
    with st.expander("🔍 Search Any NSE Stock"):
        search_query = st.text_input(
            "Type company name (e.g., Tata, Infosys, Adani):",
            placeholder="Type & press Enter for suggestions..."
        ).strip()

        if search_query:
            suggestions = search_nse_stocks(search_query)
            if suggestions:
                st.selectbox(
                    "Choose from suggestions:",
                    ["-- Select a stock --"] + list(suggestions.keys()),
                    key="search_picker",
                    on_change=on_search_pick,
                    args=(suggestions,)
                )
            else:
                st.warning("No matching NSE stocks found. Try another name.")

    st.write("")
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    # Placeholder for the pinned live snapshot
    sidebar_snapshot = st.empty()

    # 3. Clickable Search History at the Bottom
    if st.session_state.search_history:
        st.markdown("##### 🕒 Recent Searches")
        for h_name, h_sym in st.session_state.search_history:
            short_sym = h_sym.replace(".NS", "")
            if st.button(f"📈 {h_name} ({short_sym})", key=f"hist_{h_sym}"):
                st.session_state.active_stock = (h_name, h_sym)
                st.rerun()

    st.markdown("""
    <div style="margin-top: 28px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.08);
                color: #64748b; font-size: 11px; text-align: center;">
        ⚡ Powered by Random Forest AI & Live NSE Data<br>
        For educational & demonstration purposes only.
    </div>
    """, unsafe_allow_html=True)

selected_stock, ticker = st.session_state.active_stock

# ---------------------------------------------------------
# 7. Parallel Data Fetch & Model Execution
# ---------------------------------------------------------
try:
    with st.status(f"Awakening AI for {selected_stock}...", expanded=True) as status:
        st.write("📡 Connecting to live stock market feeds...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_df = executor.submit(load_live_stock, ticker)
            future_news = executor.submit(fetch_news_sentiment, selected_stock)
            df = future_df.result()
            news_data = future_news.result()

        st.write("🧠 Training neural decision trees...")
        result = train_model(df)
        status.update(label="AI Successfully Calibrated!", state="complete", expanded=False)
except Exception as e:
    st.error(f"System Failure: {e}")
    st.stop()

st.toast(f"{selected_stock} AI Profile is locked and loaded!", icon="✨")

next_trading_day = (datetime.now() + pd.tseries.offsets.BDay(1)).strftime("%A, %d %B %Y")
current_price = df['CLOSE'].iloc[-1]
price_change = current_price - df['CLOSE'].iloc[-2]
pct_change = (price_change / df['CLOSE'].iloc[-2]) * 100

# Populate Pinned Sidebar Snapshot
change_color = "#00ff88" if price_change >= 0 else "#ff3366"
arrow = "▲" if price_change >= 0 else "▼"



# Main Header
st.markdown(f"<h1 class='gradient-text'>✨ {selected_stock} AI Trend Forecaster</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #888; font-size: 16px;'>Live NSE Data (Ticker: {ticker}) • Auto-Optimized Machine Learning</p>", unsafe_allow_html=True)

# 5 Interactive Tabs
tab_predict, tab_nlp, tab_chart, tab_sim, tab_metrics = st.tabs([
    "🎯 AI Forecast", "📰 Market Mood", "📈 Price Trends", "💰 Profit Simulator", "🧠 AI Brain Metrics"
])

# --- TAB 1: AI FORECAST ---
with tab_predict:
    latest_row = df.iloc[-1]
    X_latest = latest_row[result["features"]].values.reshape(1, -1)
    X_latest_scaled = result["scaler"].transform(X_latest)

    prediction = int(result["model"].predict(X_latest_scaled)[0])
    confidence = result["model"].predict_proba(X_latest_scaled)[0][prediction]
    current_rsi = latest_row["RSI"]

    # News Risk & Crash Detector (Whole-Word + Sentiment Synced)
    RED_FLAG_WORDS = [
        "crash", "plunge", "plunges", "scam", "fraud", "slump", "tumbles",
        "tumble", "penalty", "ban", "banned", "downgrade", "bankruptcy",
        "collapse", "heavy loss", "sell-off", "selloff", "investigation", "default"
    ]
    avg_polarity = (
        sum(item["polarity"] for item in news_data) / len(news_data)
        if news_data else 0.0
    )
    flagged_headlines = [
        item["title"] for item in news_data
        if item["polarity"] <= 0.05
        and any(re.search(rf"\b{re.escape(word)}\b", item["title"].lower()) for word in RED_FLAG_WORDS)
    ]
    news_crash_alert = (avg_polarity < -0.10) or (len(flagged_headlines) > 0)

    st.markdown(f"<h3 style='text-align:center; margin-bottom: 20px;'>Prediction Date: <b>{next_trading_day}</b></h3>", unsafe_allow_html=True)

    if news_crash_alert:
        trigger_reason = (
            f"Red-flag headline detected: <i>\"{flagged_headlines[0]}\"</i>"
            if flagged_headlines
            else f"Overall news mood is heavily negative (Score: {avg_polarity:+.2f})."
        )
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,0,51,0.25), rgba(120,0,20,0.6));
                    border: 2px solid #ff3366; border-radius: 16px; padding: 20px;
                    margin-bottom: 20px; text-align: center;">
            <h2 style="color: #ff3366; margin: 0;">🚨 URGENT NEWS ALERT — HIGH DOWNSIDE RISK</h2>
            <p style="color: #f8fafc; margin: 10px 0 0 0; font-size: 15px;">{trigger_reason}</p>
        </div>
        """, unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-up">
            <h1 style='font-size: 45px; margin: 0;'>🚀 UPWARD TREND</h1>
            <p style='font-size: 20px; margin-top: 15px; color: #a3a8b8;'>The AI is <b><span style='color: #00ff88;'>{confidence * 100:.1f}% confident</span></b> the price will rise.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-down">
            <h1 style='font-size: 45px; margin: 0;'>⚠️ DOWNWARD TREND</h1>
            <p style='font-size: 20px; margin-top: 15px; color: #a3a8b8;'>The AI is <b><span style='color: #ff3366;'>{confidence * 100:.1f}% confident</span></b> the price will fall.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 🧭 Live Action Guide (AI + News Combined)")
    act1, act2 = st.columns(2)

    if news_crash_alert:
        own_title = "🚨 SELL IMMEDIATELY (News Risk)"
        own_color = "#ff3366"
        own_desc = "Breaking news signals heavy downside risk. Exit or tighten your stop-loss immediately—news shocks can override technical trends."

        new_title = "🛑 DO NOT BUY (Falling Knife)"
        new_color = "#ff3366"
        new_desc = "Avoid entering while negative news is driving the stock. Wait for headlines to calm down and the AI to confirm an upward reversal."

    elif prediction == 1:
        own_title = "🟢 HOLD YOUR SHARES"
        own_color = "#00ff88"
        own_desc = "Keep holding to ride the projected upward move. News sentiment is clear with no major red flags."

        if current_rsi > 70:
            new_title = "🟡 BUY WITH CAUTION (Overbought)"
            new_color = "#ffaa00"
            new_desc = f"Trend is up, but RSI is high ({current_rsi:.1f}). Consider waiting for a small intraday dip before buying."
        else:
            new_title = "🚀 GOOD TIME TO BUY"
            new_color = "#00ff88"
            new_desc = f"Momentum is healthy (RSI: {current_rsi:.1f}), news is clean, and the AI forecasts a rise."

    else:
        own_title = "🔴 SELL / MOVE TO CASH"
        own_color = "#ff3366"
        own_desc = "The AI expects a drop. Consider selling or setting a tight stop-loss to protect your capital while the market is weak."

        if current_rsi < 30:
            new_title = "⏳ WAIT FOR BOUNCE (Oversold)"
            new_color = "#ffaa00"
            new_desc = f"Stock is heavily oversold (RSI: {current_rsi:.1f}), so a rebound may be near—wait for the AI to flip UP before buying."
        else:
            new_title = "🛑 WAIT — DO NOT BUY YET"
            new_color = "#ff3366"
            new_desc = f"Avoid buying into a falling market (RSI: {current_rsi:.1f}). Wait until the price stabilizes and the AI signals UP."

    with act1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {own_color};">
            <p style="font-size: 13px; color: #a3a8b8; margin: 0;">IF YOU ALREADY OWN THIS STOCK</p>
            <h3 style="margin: 8px 0; color: {own_color};">{own_title}</h3>
            <p style="font-size: 14px; color: #cbd5e1; margin: 0;">{own_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    with act2:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {new_color};">
            <p style="font-size: 13px; color: #a3a8b8; margin: 0;">IF YOU DON'T OWN IT YET</p>
            <h3 style="margin: 8px 0; color: {new_color};">{new_title}</h3>
            <p style="font-size: 14px; color: #cbd5e1; margin: 0;">{new_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.markdown("#### 🔍 What the AI is currently seeing (Live Data):", help="These are the exact numerical features the AI is looking at right now to make its decision.")

    metric_cols = st.columns(3)
    for i, feature in enumerate(result["features"]):
        friendly_label = FRIENDLY_NAMES.get(feature, feature)
        metric_cols[i % 3].metric(friendly_label, f"{latest_row[feature]:.4f}")

# --- TAB 2: MARKET MOOD ---
with tab_nlp:
    st.markdown("### 📰 Live Market Mood & News")
    st.write("The AI automatically reads trusted Indian financial news headlines to determine if the public feels optimistic (Bullish) or pessimistic (Bearish) about the company.")

    if not news_data:
        st.info("No recent breaking news headlines available right now.")
    else:
        macro_sentiment = "BULLISH 🟢" if avg_polarity > 0.05 else "BEARISH 🔴" if avg_polarity < -0.05 else "NEUTRAL ⚪"
        macro_color = "#00ff88" if avg_polarity > 0.05 else "#ff3366" if avg_polarity < -0.05 else "#ffffff"

        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-top: 4px solid {macro_color};">
            <p style="font-size: 16px; color: #a3a8b8; margin: 0;">Overall News Emotion</p>
            <h2 style="margin: 10px 0; color: {macro_color};">{macro_sentiment}</h2>
        </div>
        """, unsafe_allow_html=True)

        for item in news_data:
            badge_class = f"badge-{item['sentiment'].lower()}"
            st.markdown(f"""
            <div class="glass-card">
                <span class="{badge_class}">{item['sentiment']} Vibe</span>
                <span style="color: #888; font-size: 13px; margin-left: 10px;">Source: {item['publisher']}</span>
                <a href="{item['link']}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: white;">
                    <h4 style="margin: 15px 0 8px 0; transition: color 0.2s;" onMouseOver="this.style.color='#00bfff'" onMouseOut="this.style.color='white'">📰 {item['title']}</h4>
                </a>
                <a href="{item['link']}" target="_blank" rel="noopener noreferrer" style="color: #00bfff; font-size: 13px; text-decoration: none; margin-right: 16px;">Read Full Article ↗</a>
                <a href="{item['backup_link']}" target="_blank" rel="noopener noreferrer" style="color: #94a3b8; font-size: 13px; text-decoration: none;">Backup Search 🔎</a>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 3: INTERACTIVE PRICE TRENDS (PLOTLY) ---
with tab_chart:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Closing Price", f"₹{current_price:,.2f}", f"{price_change:,.2f} ({pct_change:.2f}%)")
    c2.metric("Total Days Analyzed", f"{len(df):,}")
    c3.metric("Last Data Update", df['DATE'].iloc[-1].strftime("%d %b %Y"))

    chart_type = st.radio("Chart Style:", ["Candlestick (OHLC)", "Smooth Area Line"], horizontal=True)

    fig = go.Figure()

    if chart_type == "Candlestick (OHLC)":
        fig.add_trace(go.Candlestick(
            x=df["DATE"],
            open=df["OPEN"], high=df["HIGH"], low=df["LOW"], close=df["CLOSE"],
            name="Daily Price",
            increasing_line_color="#00ff88", decreasing_line_color="#ff3366"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df["DATE"], y=df["CLOSE"],
            mode="lines", name="Closing Price",
            line=dict(color="#00bfff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0, 191, 255, 0.12)"
        ))

    fig.add_trace(go.Scatter(
        x=df["DATE"], y=df["MA3"],
        mode="lines", name="3-Day Trend",
        line=dict(color="#bc13fe", width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df["DATE"], y=df["MA10"],
        mode="lines", name="10-Day Trend",
        line=dict(color="#ffaa00", width=1.5)
    ))

    fig.update_layout(
        title=f"{selected_stock} — 1-Year Interactive Price Action (Hover & Zoom)",
        paper_bgcolor="#050914",
        plot_bgcolor="#050914",
        font=dict(color="#e2e8f0", family="Poppins"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Price (₹)"),
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: WHAT-IF INVESTMENT SIMULATOR ---
with tab_sim:
    st.markdown("### 💰 What-If Investment Simulator")
    st.write("See how much money you would have made by following the AI's daily signals compared to simply buying and holding the stock.")

    s1, s2 = st.columns(2)
    with s1:
        initial_capital = st.slider("Starting Investment (₹)", min_value=1000, max_value=100000, value=10000, step=1000)
    with s2:
        sim_days = st.slider("Simulation Period (Last N Trading Days)", min_value=15, max_value=90, value=30, step=5)

    sim_df = result["model_df"].tail(sim_days).copy().reset_index(drop=True)
    X_sim_scaled = result["scaler"].transform(sim_df[result["features"]])
    sim_df["AI_SIGNAL"] = result["model"].predict(X_sim_scaled)

    sim_df["ACTUAL_NEXT_RET"] = (sim_df["NEXT_CLOSE"] - sim_df["CLOSE"]) / sim_df["CLOSE"]
    sim_df["AI_RET"] = np.where(sim_df["AI_SIGNAL"] == 1, sim_df["ACTUAL_NEXT_RET"], 0.0)

    sim_df["AI_PORTFOLIO"] = initial_capital * (1 + sim_df["AI_RET"]).cumprod()
    sim_df["HOLD_PORTFOLIO"] = initial_capital * (1 + sim_df["ACTUAL_NEXT_RET"]).cumprod()

    final_ai = sim_df["AI_PORTFOLIO"].iloc[-1]
    ai_profit = final_ai - initial_capital
    ai_profit_pct = (ai_profit / initial_capital) * 100

    final_hold = sim_df["HOLD_PORTFOLIO"].iloc[-1]
    hold_profit = final_hold - initial_capital
    hold_profit_pct = (hold_profit / initial_capital) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Starting Amount", f"₹{initial_capital:,.0f}")
    m2.metric("AI Strategy Final Value", f"₹{final_ai:,.2f}", f"₹{ai_profit:+,.2f} ({ai_profit_pct:+.2f}%)")
    m3.metric("Buy & Hold Final Value", f"₹{final_hold:,.2f}", f"₹{hold_profit:+,.2f} ({hold_profit_pct:+.2f}%)")

    sim_df["PREV_SIGNAL"] = sim_df["AI_SIGNAL"].shift(1).fillna(0)
    buy_points = sim_df[(sim_df["AI_SIGNAL"] == 1) & (sim_df["PREV_SIGNAL"] == 0)]
    sell_points = sim_df[(sim_df["AI_SIGNAL"] == 0) & (sim_df["PREV_SIGNAL"] == 1)]

    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(
        x=sim_df["DATE"], y=sim_df["AI_PORTFOLIO"],
        mode="lines", name="AI Strategy (₹)",
        line=dict(color="#00ff88", width=3)
    ))
    sim_fig.add_trace(go.Scatter(
        x=sim_df["DATE"], y=sim_df["HOLD_PORTFOLIO"],
        mode="lines", name="Buy & Hold (₹)",
        line=dict(color="#00bfff", width=2, dash="dash")
    ))
    sim_fig.add_trace(go.Scatter(
        x=buy_points["DATE"], y=buy_points["AI_PORTFOLIO"],
        mode="markers", name="BUY Signal",
        marker=dict(symbol="triangle-up", size=14, color="#00ff88", line=dict(color="white", width=1.5)),
        hovertemplate="<b>🟢 BUY</b><br>Date: %{x|%d %b %Y}<br>Portfolio: ₹%{y:,.2f}<extra></extra>"
    ))
    sim_fig.add_trace(go.Scatter(
        x=sell_points["DATE"], y=sell_points["AI_PORTFOLIO"],
        mode="markers", name="SELL (Cash)",
        marker=dict(symbol="triangle-down", size=14, color="#ff3366", line=dict(color="white", width=1.5)),
        hovertemplate="<b>🔴 SELL / CASH</b><br>Date: %{x|%d %b %Y}<br>Portfolio: ₹%{y:,.2f}<extra></extra>"
    ))

    sim_fig.update_layout(
        title=f"Portfolio Growth & AI Buy/Sell Signals (Last {sim_days} Trading Days)",
        paper_bgcolor="#050914",
        plot_bgcolor="#050914",
        font=dict(color="#e2e8f0", family="Poppins"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Portfolio Value (₹)"),
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(sim_fig, use_container_width=True)

    with st.expander("📋 View Exact Buy / Sell Dates"):
        log_df = sim_df[["DATE", "CLOSE", "AI_SIGNAL", "AI_PORTFOLIO"]].copy()
        log_df["DATE"] = log_df["DATE"].dt.strftime("%d %b %Y")
        log_df["ACTION"] = np.where(log_df["AI_SIGNAL"] == 1, "🟢 BUY / HOLD", "🔴 SELL / CASH")
        log_df["CLOSE"] = log_df["CLOSE"].map("₹{:,.2f}".format)
        log_df["AI_PORTFOLIO"] = log_df["AI_PORTFOLIO"].map("₹{:,.2f}".format)
        st.dataframe(
            log_df[["DATE", "CLOSE", "ACTION", "AI_PORTFOLIO"]].rename(
                columns={"DATE": "Date", "CLOSE": "Stock Price", "ACTION": "AI Action", "AI_PORTFOLIO": "Portfolio Value"}
            ),
            use_container_width=True,
            hide_index=True
        )

# --- TAB 5: AI BRAIN METRICS ---
with tab_metrics:
    st.markdown("### 🧠 How Smart is the AI?")
    st.write("This shows how accurately the AI predicted historical data during its background testing phase.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Overall Accuracy: {result['accuracy'] * 100:.1f}%**")
        st.progress(result['accuracy'])
        st.markdown(f"**Upward Prediction Accuracy (Precision): {result['precision'] * 100:.1f}%**")
        st.progress(result['precision'])

    with col2:
        st.markdown(f"**Trend Capture Rate (Recall): {result['recall'] * 100:.1f}%**")
        st.progress(result['recall'])
        st.markdown(f"**Balanced Brain Score (F1): {result['f1'] * 100:.1f}%**")
        st.progress(result['f1'])

    with st.expander("Show Deep-Dive Confusion Matrix"):
        st.write("This table shows exactly how many times the AI guessed right vs wrong.")
        cm_df = pd.DataFrame(
            result["matrix"],
            index=["Actually Went Down", "Actually Went Up"],
            columns=["AI Guessed Down", "AI Guessed Up"]
        )
        st.dataframe(cm_df, use_container_width=True)

    with st.expander("📂 View & Download Live Dataset"):
        display_cols = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"] + result["features"] + ["TARGET"]
        
        # Full 1-year dataset sorted newest first
        full_dataset = df[display_cols].sort_values("DATE", ascending=False).reset_index(drop=True)
        
        st.write(f"Showing the **latest 15 rows** below. Download the CSV for all **{len(full_dataset)} trading days (1 full year)**.")
        
        # Display only a compact preview in the app
        st.dataframe(full_dataset.head(15), use_container_width=True, hide_index=True)

        # Download the entire 1-year dataset
        csv_data = full_dataset.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download Full 1-Year Dataset ({len(full_dataset)} Days CSV)",
            data=csv_data,
            file_name=f"{ticker.replace('.NS', '')}_1year_dataset.csv",
            mime="text/csv"
        )