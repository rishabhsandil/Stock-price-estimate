import pandas as pd
import tqdm

FINANCIALS = "data/all_financials.csv"
STOCK = "data/stock_price.csv"
ALIAS = "data/ticker_alias.csv"
INFO = "data/ticker_info.csv"
INDICES = {
    "dow": "data/dow_historical1.csv",
    "sp500": "data/sp500_historical1.csv",
    "nasdaq": "data/nasdaq_historical1.csv",
    "tsx": "data/tsx_historical1.csv"
    }

# Start with the base financials
print("Reading financials...")
database = pd.read_csv(FINANCIALS)
new_colnames = {c: c.lower().replace("& ", "").replace(" ", "_").replace("/", "").replace("-", "") for c in database.columns}
new_colnames["Quarter end"] = "date"
database = database.rename(columns=new_colnames)
database["date"] = pd.to_datetime(database["date"])

# Join alias
print("Reading alias...")
alias = pd.read_csv(ALIAS)
database = database.merge(alias, how="left")

# Join info by ticker or ticker alias
print("Reading info...")
info = pd.read_csv(INFO).rename(columns={"type": "stock_type"})
database = database.merge(info, how="left")

# Join indices
print("Reading indices...")
for index_name, fname in INDICES.items():
    index = pd.read_csv(fname)
    idxn = index_name + "_idx"
    idxm = index_name + "_idx_vol"
    index = index.rename(columns={c: c.lower().replace(" ", "") for c in index.columns})
    index[idxn] = index["close"]
    index[idxm] = index["volume"]
    index = index[["date", idxn, idxm]]
    index["date"] = pd.to_datetime(index["date"])
    # Interpolate to add all the days (max 1 week)
    all_dates = pd.DataFrame({"date": pd.date_range(start=index["date"].min(), end=index["date"].max(), freq="D")})
    all_dates = all_dates.merge(index, how="left")
    all_dates = all_dates.interpolate(method="linear", limit=7)
    database = database.merge(all_dates, how="left")

# Add day of week, month, and distance in month
print("Adding timing information...")
database["day_of_week"] = database["date"].dt.dayofweek
database["month"] = database["date"].apply(lambda x: x.month)
database["days_in_month"] = database["date"].dt.day / database["date"].dt.daysinmonth

# Join stock data 30mins
print("Reading stock...")
stock = pd.read_csv(STOCK)
stock["range"] = (stock["high"] - stock["low"]) / ((stock["high"] + stock["low"]) / 2)
stock = stock[["ticker", "report_date", "date", "relative_day", "volume", "close", "range"]]

df = pd.DataFrame()
for name, group in tqdm.tqdm(stock.groupby(["report_date", "ticker"])):
    d = {}

    # Interpolate at most 1 day (Hollidays etc (Not accounted for by b-days))
    tmp = pd.DataFrame({"relative_day": range(-10, 10)})
    group = tmp.merge(group, how="left")
    group = group.interpolate(method="linear", limit=1)

    for _, row in group.iterrows():
        rd = str(row["relative_day"]).replace("-", "b")
        for k in ["volume", "close", "range"]:
            d[k + "_" + rd] = row[k]
    d["date"] = name[0]
    d["ticker"] = name[1]
    df = df.append(d, ignore_index=True)

df["date"] = pd.to_datetime(df["date"])
database = database.merge(df, how="left")

database.to_csv("output.csv", index=False)
