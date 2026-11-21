# ✨ AI-Driven Stock Trend Forecaster (Live NSE + News Sentiment)

An interactive, real-time stock market trend prediction web application built with **Streamlit**, **Scikit-Learn**, **Plotly**, and **Yahoo Finance**. It combines technical analysis powered by a Random Forest machine learning model with live Indian financial news sentiment analysis to deliver actionable insights, interactive visualizations, and portfolio backtesting.

---

## 🚀 Key Features

### 1. 🎯 AI Forecast & Live Action Guide
* **Next-Trading-Day Prediction:** Predicts whether a stock's closing price will move **UP** or **DOWN** on the next valid market day (automatically skips weekends using business-day logic).
* **Live Action Guide:** Provides clear, side-by-side guidance tailored to both current shareholders (*Hold vs. Sell*) and prospective buyers (*Buy vs. Wait*), factoring in **RSI overbought/oversold levels**.
* **Urgent News Crash Detector:** Continuously scans live headlines using whole-word regex matching for high-risk keywords (e.g., *crash, plunge, fraud, downgrade, penalty*) and negative sentiment polarity. If triggered, it overrides technical signals with an emergency **SELL IMMEDIATELY** warning.

### 2. 📰 Market Mood (Finance-Aware NLP)
* Fetches live headlines via Google News RSS, strictly whitelisted to trusted Indian financial publishers (**Moneycontrol, The Economic Times, Livemint, and Business Standard**).
* Uses a **finance-aware sentiment engine** that combines a domain-specific bullish/bearish keyword booster with `TextBlob` polarity scoring to prevent misclassifications common in general-purpose NLP.
* Includes direct article links and backup search links for every headline.

### 3. 📈 Interactive Price Trends
* Powered by **Plotly** with full hover, zoom, and pan support.
* Toggle between **Candlestick (OHLC)** and **Smooth Area Line** views overlaid with 3-day and 10-day moving averages.

### 4. 💰 What-If Investment Profit Simulator
* Backtests the **AI Strategy** (invest when the AI predicts UP, sit safely in cash when the AI predicts DOWN) against a standard **Buy & Hold** benchmark over a customizable investment amount and timeframe (15–90 trading days).
* Displays exact **🟢 BUY (`▲`)** and **🔴 SELL (`▼`)** transition markers directly on the portfolio growth curve alongside an expandable daily trade log.

### 5. 🧠 AI Brain Metrics & Live Dataset Export
* Displays model evaluation metrics (**Accuracy, Precision, Recall, F1-Score**) and a **Confusion Matrix**.
* Includes a **View & Download Live Dataset** panel showing a clean preview in-app while allowing a full 1-year processed dataset export as a `.csv` file.

### 6. 🔮 Modern Dark-Mode Control Panel
* **Featured NSE Stocks:** Quick access to HDFC Bank, Adani Enterprises, Reliance Industries, and TCS.
* **Live NSE Search with Suggestions:** Connects to Yahoo Finance's search API with brand-name aliases (e.g., searching *Zomato* resolves to *Eternal Limited / ETERNAL.NS*) and enforces selecting valid `.NS` tickers.
* **Recent Search History & Pinned Snapshot:** Saves searched stocks for one-click reopening and pins live price, daily change, and AI accuracy in the sidebar.

---

## 🛠️ Tech Stack & Libraries

| Library | Purpose |
| :--- | :--- |
| **`streamlit`** | Interactive web UI, dark-mode glassmorphism layout, caching, and session state |
| **`yfinance`** | Fetching live 1-year historical OHLCV market data from NSE |
| **`pandas` & `numpy`** | Data cleaning, business-day calculation, and technical indicator engineering |
| **`scikit-learn`** | `StandardScaler` preprocessing, `RandomForestClassifier`, and evaluation metrics |
| **`plotly`** | Interactive candlestick, area, and portfolio backtesting charts |
| **`textblob`** | Natural Language Processing (NLP) fallback for news headline polarity scoring |

---

## 📊 Engineered Technical Features

The machine learning pipeline dynamically computes **9 technical indicators** from live 1-year OHLCV data:

1. **Daily Return (`RETURN`)** — Percentage change in closing price
2. **3-Day Moving Average (`MA3`)** — Short-term price trend
3. **5-Day Moving Average (`MA5`)** — Weekly price trend
4. **10-Day Moving Average (`MA10`)** — Medium-term price trend
5. **Volume Momentum (`VOLUME_CHANGE`)** — Daily percentage change in trading volume
6. **Relative Strength Index (`RSI`)** — 14-period exponential momentum oscillator
7. **MACD (`MACD`)** — Difference between 12-day and 26-day EMAs
8. **Lag Close 1 (`LAG_CLOSE_1`)** — Previous trading day's closing price
9. **Lag Close 2 (`LAG_CLOSE_2`)** — Closing price two trading days prior

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* Python **3.9** or higher
* Active internet connection (required for live NSE data, search suggestions, and Google News RSS feeds)

### 2. Install Required Libraries
Create a `requirements.txt` file (or install directly via pip):

```bash
pip install streamlit yfinance pandas numpy scikit-learn plotly textblob
```

*(Optional — if `TextBlob` prompts for corpora on first run):*
```bash
python -m textblob.download_corpora
```

### 3. Run the Application
Navigate to the folder containing `app.py` and launch Streamlit:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
├── app.py               # Main Streamlit application (UI, ML pipeline, NLP, charts)
├── requirements.txt     # Python library dependencies
└── README.md            # Project documentation
```

---

## ⚠️ Disclaimer

This application is developed strictly for **educational and academic demonstration purposes**. Historical backtests do not account for brokerage fees, slippage, or taxes, and AI predictions should **not** be treated as financial or investment advice.
