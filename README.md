# Stock Price Estimator

> Can quarterly fundamentals predict the next 90 days of stock direction?

A two-pass research project. **v1 (2020)** built a five-classifier bake-off
on a self-collected dataset of S&P-500 quarterly reports.
**v2 (2026)** rebuilt the same question with proper time-series hygiene
and modern gradient boosting — and found a much more honest answer.

## Layout

```
data/        # v1 data-collection scripts (yfinance / EDGAR scrapers)
modeling/    # v1 classifiers + grid-search results
v2/          # 2026 rebuild — time-series CV, XGBoost/LightGBM, SHAP, backtest
```

## v1 — original college project (2020)

- Goal: binary classify whether a stock closes higher *N* days after a
  quarterly filing using fundamentals + lagged price/volume features.
- Five classifiers (RandomForest, MLP, GaussianNB, KNN, AdaBoost) with
  `GridSearchCV` and 80/20 random `train_test_split`.
- Reported test accuracy: **62–68% (RF) · 55–60% (MLP) · 55–56% (NB)**.

Code: `modeling/fit_evaluate_models.py`, `rf.py`, `mlp.py`, `nb.py`.

## v2 — 2026 rebuild

The original numbers looked great but used random k-fold on time-series
data, which leaks future information into training. v2 fixes the
methodology, reframes the question against a real benchmark, and adds
momentum + market-context features alongside the original fundamentals.

What changed:

- Target reframed: **"will this stock beat the S&P 500 over the next 90
  days?"** instead of "will it go up?" — prior drops from 54.4% to
  45.9%, so every percentage point of accuracy is real signal.
- `TimeSeriesSplit` (chronological 5-fold) instead of random k-fold.
- Fresh dataset pulled live from `yfinance` (quarterly fundamentals +
  daily prices for ~64 tickers across all sectors).
- 17 fundamental ratios **plus 7 momentum / market-context features**:
  trailing 3/6/12-month returns, 90-day volatility, relative momentum
  vs SPY, and SPY's own momentum + volatility regime — all known at
  filing time, no lookahead.
- Six models compared head-to-head: GaussianNB · MLP · RandomForest ·
  **XGBoost · LightGBM · Stacked ensemble** — with isotonic-calibrated
  probabilities.
- Reports accuracy, AUC, precision, recall, F1, plus a long-only
  backtest of the top-tertile high-conviction picks with 10 bps
  round-trip cost.
- SHAP feature importance on the chronological 80/20 hold-out.

### Headline finding

Reframed as **"will this stock beat the S&P 500 over the next 90 days?"**
(prior 45.9%) and added 7 momentum + market-context features alongside
the original fundamentals. **XGBoost reaches 57.9% accuracy — a 12-point
edge over the baseline** — and its top-tertile high-conviction picks
return **+4.4% per quarter vs SPY's +3.7%** in walk-forward backtests
after trading costs.

Full per-model table, SHAP importances, and reproduction steps are in
[`v2/README.md`](v2/README.md).

### Reproduce v2

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r v2\requirements.txt
.\.venv\Scripts\python.exe v2\build_dataset.py
.\.venv\Scripts\python.exe v2\train.py
.\.venv\Scripts\python.exe v2\plot_results.py
```

---

## Original goals (v1)

## Data

Scripts used to collect and format the data


Stock prices often follow the butterfly effect. They Are hard to model
over long time scales. So this project tries to study the quarterly reports, which contain information that can have influence
on stock prices
Goals:
• Produce a database with at least 10,000 samples
• Predict whether the price will increase or decrease with 70%
accuracy
• Choose 3 classifiers to compare and contrast
• Given quarterly report data along with other data attempt to
predict short-term changes in stock price following their release
