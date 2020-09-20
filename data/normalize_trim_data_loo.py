import pandas as pd
import numpy as np
import tqdm


KEEP_COLUMNS = ["day_of_week", "eps_diluted", "liabilities", "month", "shareholders_equity", "tsx_idx",
    "sector_Consumer Non-Durables", "sector_Distribution Services", "sector_Finance", "country_AU"] + \
    [s.format(n) for n in range(1,11) for s in ["close_b{}", "range_b{}", "volume_b{}"]] + ["close_1"]

db = pd.read_csv("database.csv", na_values=["", "None"])

# Remove columns we are not interested in
print("Removing columns...")
REMOVE_COLUMNS = ["shares", "ticker_alias", "volume_0", "range_0", "close_0", "stock_type"]
db = db[[c for c in db.columns if c not in REMOVE_COLUMNS]]

# Remove weird reports (For some reason these two reports are crazy outliers)
print("Removing reports...")
REMOVE_REPORTS = [{"ticker": "ETFC", "date": "2004-03-31"}, {"ticker": "ETFC", "date": "2004-12-31"}]
for r in REMOVE_REPORTS:
    r["date"] = pd.to_datetime(r["date"])
    db = db[(db["ticker"] != r["ticker"]) & (db["date"] != r["date"])]

# Order the columns in a better way
print("Re-ordering columns...")
COLUMN_ORDER = ["ticker", "date", "sector", "country", "exchange"]
db = db[COLUMN_ORDER + [c for c in sorted(db.columns) if c not in COLUMN_ORDER]]


# Previous quarterly report as reference for values
print("Re-referencing columns...")
QUARTERLY_REPORT_COLS = ["shares_split_adjusted", "split_factor", "assets", "current_assets",
    "liabilities", "current_liabilities", "shareholders_equity", "noncontrolling_interest",
    "preferred_equity", "goodwill_intangibles", "longterm_debt", "revenue", "earnings",
    "earnings_available_for_common_stockholders", "eps_basic", "eps_diluted", "dividend_per_share",
    "cash_from_operating_activities", "cash_from_investing_activities", "cash_from_financing_activities",
    "cash_change_during_period", "cash_at_end_of_period", "capital_expenditures", "price",
    "price_high", "price_low", "roe", "roa", "book_value_of_equity_per_share", "pb_ratio",
    "pe_ratio", "cumulative_dividends_per_share", "dividend_payout_ratio",
    "longterm_debt_to_equity_ratio", "equity_to_assets_ratio", "net_margin", "asset_turnover",
    "free_cash_flow_per_share", "current_ratio", "dow_idx", "dow_idx_vol", "sp500_idx",
    "sp500_idx_vol", "nasdaq_idx", "nasdaq_idx_vol", "tsx_idx", "tsx_idx_vol"]

db_new = pd.DataFrame()
for name, group in tqdm.tqdm(db.groupby(["ticker"])):
    group = group.sort_values(["ticker", "date"])
    db_new = db_new.append(group[QUARTERLY_REPORT_COLS].pct_change(periods=-1), ignore_index=True)

db[QUARTERLY_REPORT_COLS] = db_new[QUARTERLY_REPORT_COLS]


# Use b1 stock price and volume as reference for values
print("Changing stock price reference...")
APPLY_COLUMNS = ["volume_{}", "close_{}"]
RANGE = [i for i in range(-10, 10) if i != 0]

apply_cols = [[a.format(k if k >= 0 else "b" + str(-k)) for k in RANGE] for a in APPLY_COLUMNS]
refs = [db[a.format("b1")].copy(deep=True) for a in APPLY_COLUMNS]

for ref, cols in zip(refs, apply_cols):
    for col in cols:
        db[col] = (db[col] - ref) / ref


# Day of week and month to 0-1
db["month"] = db["month"]/12
db["day_of_week"] = db["day_of_week"]/7


# Inf to max, -inf to min
print("Replacing inf...")
for col in tqdm.tqdm(db.columns):
    try:
        if db[col].dtype == "float64":
            db.loc[db[col] == float("inf"), col] = db.loc[np.isfinite(db[col])][col].max()
            db.loc[db[col] == float("-inf"), col] = db.loc[np.isfinite(db[col])][col].min()
    except TypeError:
        pass



CLASSIFICATION_FEATURE = "close_1"
IGNORE_FEATURES = [*[s.format(n) for n in range(1,10) for s in ["close_{}", "range_{}", "volume_{}"]], "date", "ticker"]

# Get the input and output
db = db[[c for c in db.columns if c not in IGNORE_FEATURES] + [CLASSIFICATION_FEATURE]]

# Convert input categorical features to one-hot encoded features
category_cols = db.select_dtypes(include='object').columns
for c in category_cols:
    db = pd.concat([db, pd.get_dummies(db[c], prefix=c)], axis=1)
db = db[[c for c in db.columns if c not in category_cols]]


db = db[KEEP_COLUMNS].dropna()

db.to_csv("database_loo.csv", index=False)
