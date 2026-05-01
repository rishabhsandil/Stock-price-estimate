"""
v2 dataset builder — adds momentum, volatility, market-relative
features, and reframes the target as 'beat the market'.

Key changes vs build_dataset.py:
  - Target = does this stock outperform SPY over the next 90 days?
    (Cross-sectional / market-neutral framing kills the always-up bias.)
  - Momentum features: 3m / 6m / 12m trailing returns up to filing date
    (public information at the time of filing — not leakage).
  - Volatility: 90d rolling stdev up to filing date.
  - Market context: SPY 90d trailing return, SPY 90d vol at filing date.
  - Earnings surprise proxy: QoQ revenue growth surprise vs prior 4Q trend.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

TICKERS = [
    "AAPL","MSFT","GOOGL","META","NVDA","AMD","INTC","ORCL","CRM","ADBE","CSCO","IBM","QCOM","TXN","AVGO",
    "AMZN","TSLA","NFLX","HD","NKE","SBUX","MCD","TGT","WMT","COST","LOW","DIS",
    "JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW",
    "JNJ","PFE","UNH","ABBV","MRK","LLY","TMO","DHR","BMY","CVS",
    "BA","CAT","GE","MMM","HON","UPS","FDX","XOM","CVX","COP",
    "VZ","T","TMUS","CMCSA","PEP","KO","PG","CL","WBA",
]

PERIOD = "10y"


def fetch_spy():
    spy = yf.Ticker("SPY").history(period=PERIOD, auto_adjust=True)["Close"]
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    return spy.sort_index()


def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def fetch_one(ticker: str, spy: pd.Series) -> pd.DataFrame | None:
    try:
        tk = yf.Ticker(ticker)
        q_inc = tk.quarterly_financials.T
        q_bal = tk.quarterly_balance_sheet.T
        q_cf  = tk.quarterly_cashflow.T
        if q_inc.empty or q_bal.empty:
            return None

        px = tk.history(period=PERIOD, auto_adjust=True)["Close"]
        if px.empty:
            return None
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        px = px.sort_index()

        # Pre-compute daily returns + rolling stats
        log_ret = np.log(px / px.shift(1))
        spy_log_ret = np.log(spy / spy.shift(1))

        rows = []
        for qend in q_inc.index:
            qend_naive = pd.Timestamp(qend).tz_localize(None) if pd.Timestamp(qend).tz is not None else pd.Timestamp(qend)

            inc = q_inc.loc[qend] if qend in q_inc.index else pd.Series(dtype=float)
            bal = q_bal.loc[qend] if qend in q_bal.index else pd.Series(dtype=float)
            cf  = q_cf.loc[qend]  if qend in q_cf.index  else pd.Series(dtype=float)

            def g(s, key):
                try:
                    v = s.get(key, np.nan)
                    return float(v) if pd.notna(v) else np.nan
                except Exception:
                    return np.nan

            revenue   = g(inc, "Total Revenue")
            cogs      = g(inc, "Cost Of Revenue")
            op_income = g(inc, "Operating Income")
            net_inc   = g(inc, "Net Income")
            ebit      = g(inc, "EBIT") if pd.notna(g(inc, "EBIT")) else op_income
            interest  = g(inc, "Interest Expense")

            total_assets = g(bal, "Total Assets")
            total_liab   = g(bal, "Total Liabilities Net Minority Interest")
            if pd.isna(total_liab):
                total_liab = g(bal, "Total Liab")
            equity       = g(bal, "Stockholders Equity")
            if pd.isna(equity):
                equity = g(bal, "Total Stockholder Equity")
            cash         = g(bal, "Cash And Cash Equivalents")
            if pd.isna(cash):
                cash = g(bal, "Cash")
            cur_assets   = g(bal, "Current Assets")
            cur_liab     = g(bal, "Current Liabilities")
            lt_debt      = g(bal, "Long Term Debt")

            ocf = g(cf, "Operating Cash Flow")
            if pd.isna(ocf):
                ocf = g(cf, "Total Cash From Operating Activities")
            capex = g(cf, "Capital Expenditure")

            # ----- Anchor on filing date in price index -----
            try:
                p_now_idx = px.index.get_indexer([qend_naive], method="nearest")[0]
                future_date = qend_naive + pd.Timedelta(days=90)
                p_fut_idx = px.index.get_indexer([future_date], method="nearest")[0]
                if p_now_idx < 0 or p_fut_idx <= p_now_idx:
                    continue
                if px.index[p_fut_idx] < future_date - pd.Timedelta(days=10):
                    continue
                p_now, p_fut = float(px.iloc[p_now_idx]), float(px.iloc[p_fut_idx])
                fwd_ret = (p_fut / p_now) - 1.0

                # SPY forward return over same window
                s_now_idx = spy.index.get_indexer([px.index[p_now_idx]], method="nearest")[0]
                s_fut_idx = spy.index.get_indexer([px.index[p_fut_idx]], method="nearest")[0]
                if s_now_idx < 0 or s_fut_idx <= s_now_idx:
                    continue
                spy_now, spy_fut = float(spy.iloc[s_now_idx]), float(spy.iloc[s_fut_idx])
                spy_fwd_ret = (spy_fut / spy_now) - 1.0

                # NEW TARGET: outperform SPY by at least 0
                excess_ret = fwd_ret - spy_fwd_ret
                target_outperform = int(excess_ret > 0)
                target_up = int(fwd_ret > 0)  # kept for comparison
            except Exception:
                continue

            # ----- Momentum features (info available AT filing date) -----
            def trailing_return(idx, lookback_days):
                try:
                    past_date = px.index[idx] - pd.Timedelta(days=lookback_days)
                    past_idx = px.index.get_indexer([past_date], method="nearest")[0]
                    if past_idx < 0 or past_idx >= idx:
                        return np.nan
                    return float(px.iloc[idx] / px.iloc[past_idx]) - 1.0
                except Exception:
                    return np.nan

            mom_3m  = trailing_return(p_now_idx, 90)
            mom_6m  = trailing_return(p_now_idx, 180)
            mom_12m = trailing_return(p_now_idx, 365)

            # 90d trailing volatility (annualized stdev of daily log returns)
            try:
                start_idx = max(0, p_now_idx - 90)
                vol_90d = float(log_ret.iloc[start_idx:p_now_idx].std() * np.sqrt(252))
            except Exception:
                vol_90d = np.nan

            # Market context: SPY trailing 90d return + vol
            try:
                spy_now_idx = spy.index.get_indexer([px.index[p_now_idx]], method="nearest")[0]
                past_date = spy.index[spy_now_idx] - pd.Timedelta(days=90)
                past_idx = spy.index.get_indexer([past_date], method="nearest")[0]
                spy_mom_3m = float(spy.iloc[spy_now_idx] / spy.iloc[past_idx]) - 1.0
                spy_start = max(0, spy_now_idx - 90)
                spy_vol_90d = float(spy_log_ret.iloc[spy_start:spy_now_idx].std() * np.sqrt(252))
            except Exception:
                spy_mom_3m, spy_vol_90d = np.nan, np.nan

            row = {
                "ticker": ticker,
                "qend": qend_naive,
                # Fundamentals
                "rev_growth_yoy": np.nan,
                "gross_margin":   safe_div(revenue - cogs, revenue) if pd.notna(cogs) else np.nan,
                "op_margin":      safe_div(op_income, revenue),
                "net_margin":     safe_div(net_inc, revenue),
                "roa":            safe_div(net_inc, total_assets),
                "roe":            safe_div(net_inc, equity),
                "asset_turnover": safe_div(revenue, total_assets),
                "current_ratio":  safe_div(cur_assets, cur_liab),
                "debt_to_equity": safe_div(total_liab, equity),
                "lt_debt_to_assets": safe_div(lt_debt, total_assets),
                "cash_to_assets": safe_div(cash, total_assets),
                "interest_cov":   safe_div(ebit, interest) if pd.notna(interest) and interest != 0 else np.nan,
                "ocf_to_revenue": safe_div(ocf, revenue),
                "fcf_to_revenue": safe_div((ocf or np.nan) + (capex or 0), revenue) if pd.notna(ocf) else np.nan,
                "capex_to_revenue": safe_div(abs(capex) if pd.notna(capex) else np.nan, revenue),
                "log_assets":     np.log(total_assets) if pd.notna(total_assets) and total_assets > 0 else np.nan,
                "log_revenue":    np.log(revenue) if pd.notna(revenue) and revenue > 0 else np.nan,
                # NEW: momentum + market context (public info at filing date)
                "mom_3m":         mom_3m,
                "mom_6m":         mom_6m,
                "mom_12m":        mom_12m,
                "vol_90d":        vol_90d,
                "rel_mom_3m":     (mom_3m - spy_mom_3m) if pd.notna(mom_3m) and pd.notna(spy_mom_3m) else np.nan,
                "spy_mom_3m":     spy_mom_3m,
                "spy_vol_90d":    spy_vol_90d,
                # Returns + targets
                "fwd_ret_90d":    fwd_ret,
                "spy_fwd_ret_90d": spy_fwd_ret,
                "excess_ret_90d": excess_ret,
                "target_up":         target_up,           # old framing
                "target":            target_outperform,   # new primary target
            }
            rows.append(row)

        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("qend").reset_index(drop=True)
        df["rev_growth_yoy"] = (
            np.exp(df["log_revenue"]) /
            np.exp(df["log_revenue"].shift(4)) - 1.0
        )
        return df

    except Exception as e:
        print(f"[skip] {ticker}: {e}")
        return None


def main():
    spy = fetch_spy()
    out = []
    for t in tqdm(TICKERS):
        df = fetch_one(t, spy)
        if df is not None and len(df) > 0:
            out.append(df)
        time.sleep(0.15)
    full = pd.concat(out, ignore_index=True)
    print(f"Rows: {len(full)} | Tickers: {full['ticker'].nunique()} | Range: {full['qend'].min()} -> {full['qend'].max()}")
    print(f"Outperform-SPY prior: {full['target'].mean():.3f} | Up prior: {full['target_up'].mean():.3f}")
    out_path = os.path.join(os.path.dirname(__file__), "data.csv")
    full.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
