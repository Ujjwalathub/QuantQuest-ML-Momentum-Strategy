# QuantQuest: ML-Driven Weekly Momentum Strategy
### Developed for E-Summit '26 (Xpecto), IIT Mandi

This repository contains a quantitative trading strategy that utilizes Machine Learning to predict weekly price momentum for a 10-stock universe. The project was developed as part of the **Quant Hackathon** hosted by the Indian Institute of Technology (IIT), Mandi.

## 🚀 Strategy Overview
The strategy is a **Long-Only Momentum** approach that rebalances weekly. It uses a classification model to predict whether a stock will generate a positive return over the next 5 trading days.

* **Stock Universe:** AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ, BRK.B.
* **Timeframe:** * **Training:** 2017 – 2022 (Historical Market Regimes)
    * [cite_start]**Testing:** 2023 – 2025 (Out-of-Sample Backtest) [cite: 11, 23, 24]
* [cite_start]**Rebalancing:** Weekly (Every Monday) [cite: 17]
* [cite_start]**Transaction Costs:** Strictly applied 0.1% at entry and 0.1% at exit (0.2% total round-trip)[cite: 18].

## 🧠 Machine Learning Architecture
[cite_start]We implemented an **XGBoost Classifier** to handle the non-linear nature of financial time-series data[cite: 20].

### Feature Engineering
11 technical indicators were engineered to provide the model with a holistic view of the market:
* **Momentum:** 5-day, 10-day, and 21-day price returns.
* **Volatility:** 21-day rolling standard deviation.
* **Trend/Strength:** RSI (Relative Strength Index), MACD, and Bollinger Bands.
* **Volume:** Volume moving average ratios to confirm price action.

### Portfolio Logic
1.  **Prediction:** The model generates a probability score for each stock.
2.  [cite_start]**Ranking:** Stocks are ranked by their probability of a positive return[cite: 14].
3.  [cite_start]**Selection:** The top 2 stocks are selected for the weekly portfolio[cite: 15].
4.  [cite_start]**Weighting:** Equal-weight allocation (50% each)[cite: 16].

## 📊 Backtest Results (2023 - 2025)
[cite_start]The following results account for the mandatory 0.1% transaction fee per trade[cite: 49]:

| Metric | Result |
| :--- | :--- |
| **Hit Rate** | 52.41% |
| **Cumulative Return** | -36.45% |
| **Annualized Volatility** | 22.70% |
| **Max Drawdown** | -46.80% |
| **Sharpe Ratio** | -0.81 |

## 📁 Repository Structure
* [cite_start]`Stock_Trading_Strategy_Backtester.ipynb`: Main Jupyter Notebook with code, data cleaning, and visualizations[cite: 27].
* [cite_start]`Predictive Alpha.pdf`: Final presentation and report of findings[cite: 32].
* [cite_start]`backtest_holdings.csv`: Detailed log of weekly predictions and selections[cite: 37].

## 🛠️ Requirements
* Python 3.x
* yfinance
* xgboost
* pandas, numpy
* matplotlib, seaborn (for visualizations)

---
*Disclaimer: This project is for educational purposes as part of a hackathon and does not constitute financial advice.*
