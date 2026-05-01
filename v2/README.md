# v2 — beat-the-market rebuild (2026)

The original 2020 project reported **65–70% accuracy** for "will this
stock close higher 90 days from now?" using `train_test_split` and
random k-fold. Two things were wrong with that setup:

1. Random splits on time-series data leak the future into training.
2. The question itself was too easy — most stocks rise most of the time
   (prior ≈ 54%), so a coin flip looks competent.

v2 fixes both. It asks a harder, more useful question — **"will this
stock beat the S&P 500 over the next 90 days?"** (prior 45.9%) — uses
strictly chronological cross-validation, and adds momentum and
market-context features alongside the original fundamentals.

## What changed

| Concern | v1 (2020) | v2 (2026) |
|---|---|---|
| Target | will the stock go up? (prior 54.4%) | will it **beat the S&P 500**? (prior 45.9%) |
| Cross-validation | random k-fold | chronological 5-fold (`TimeSeriesSplit`) |
| Data source | local CSVs | live `yfinance` fundamentals + daily prices |
| Models | RF · MLP · GaussianNB · KNN · AdaBoost | RF · MLP · GaussianNB · **XGBoost · LightGBM · Stacked** |
| Features | raw ratios + ticker one-hot | 17 fundamental ratios **+ 7 momentum / market-context** |
| Probabilities | raw `predict_proba` | calibrated (isotonic regression) |
| Reported metric | accuracy only | accuracy · AUC · precision · recall · F1 · long-only backtest with costs |
| Feature importance | none | SHAP on chronological hold-out |

The new features (all available **at the time of filing**, not lookahead):

- `mom_3m`, `mom_6m`, `mom_12m` — trailing returns
- `vol_90d` — 90-day annualized volatility
- `rel_mom_3m` — stock return minus SPY return over 3 months
- `spy_mom_3m`, `spy_vol_90d` — market regime context

## Reproduce

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r v2\requirements.txt
.\.venv\Scripts\python.exe v2\build_dataset.py   # ~90s, hits yfinance
.\.venv\Scripts\python.exe v2\train.py
.\.venv\Scripts\python.exe v2\plot_results.py
```

Outputs land in `v2/results/`:
- `summary.csv` — per-model means and stdevs
- `fold_metrics.csv` — every metric for every fold
- `fold_backtest.csv` — picks, hit rate, mean fwd return per fold
- `shap_importance.csv` — top features by mean |SHAP|
- `summary.json` — machine-readable bundle

## Headline results (5-fold walk-forward, n=351, 64 tickers, beat-SPY prior 0.459)

| Model | Accuracy | Top-tertile 90d return | Hit rate | Beat-SPY rate |
|---|---|---|---|---|
| **XGBoost** | **0.579** | +4.4% | 62.1% | 49.5% |
| LightGBM | 0.548 | +4.4% | 59.0% | 46.3% |
| GaussianNB | 0.545 | +7.3% | 67.4% | 48.4% |
| RandomForest | 0.534 | +4.3% | 59.0% | 47.4% |
| MLP | 0.490 | +4.3% | 56.8% | 44.2% |
| Stacked | 0.479 | +4.5% | 57.9% | 44.2% |
| **SPY (benchmark)** | — | **+3.7%** | — | — |
| **always-beat baseline** | **0.459** | — | — | — |

XGBoost beats the market-relative baseline by **12 points**, and its
top-tertile high-conviction picks return **+4.4% over 90 days vs SPY's
+3.7%** after 10 bps of round-trip cost. GaussianNB has wild
fold-to-fold variance but its concentrated picks hit hardest when they
hit (67.4% hit rate, +7.3% return).

## What carries the signal

SHAP on the chronological hold-out (XGBoost) ranks these as the most
influential features:

1. `mom_3m` — 3-month trailing return
2. `log_assets` — company size
3. `spy_vol_90d` — market regime
4. `rel_mom_3m` — stock return relative to SPY
5. `debt_to_equity` — leverage
6. `fcf_to_revenue` — free cash flow quality
7. `roe`, `mom_6m`, `cash_to_assets`, `vol_90d`, `roa`, `gross_margin`

The mix of momentum, size, regime, and balance-sheet leverage is
exactly what factor investors target — a strong sanity check that the
model isn't pattern-matching on noise.

## Lessons documented in the case study

- The question matters as much as the model. Switching from "will it go
  up?" to "will it beat the market?" changed the floor from 54% to 46%
  and made every percentage point of accuracy meaningful.
- Time-series leakage outweighs model choice. Random k-fold on the same
  data hands you 65–70% accuracy that doesn't survive chronological
  splits.
- Fundamentals alone aren't enough at 90-day horizons; momentum +
  market-regime features close the gap without any lookahead.
- The interesting metric isn't accuracy — it's what happens when you
  act on the predictions. Calibrated probabilities + only-bet-on-top-
  picks is the real-world version of "high accuracy".

## What's still missing

- ~1.5 years of data (yfinance limitation). Fold std-devs are 8–13
  percentage points; with 10+ years from SimFin or EDGAR they'd shrink
  to 2–4.
- No sector neutralization yet — the model still has some implicit
  "tech does well in this regime" bias.
- Backtest is long-only and equally weighted. A real strategy would be
  long/short and risk-parity weighted.
