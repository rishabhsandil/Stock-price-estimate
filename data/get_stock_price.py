"""
This is just a hacked together script to download historical stock price information
from the IEX Finance API.  If run on the all_financials.csv data, this script
will make ~2 million requests for a total cost of around $8 for the IEX API. So
dont if you dont need to.

The "get_bulk_stock_price" function makes requests in parallel because there are
many many requests that had to be made. Unfortunately, there are rate limitations
from IEX. Having timing such that the script presses up against those limitations
would be a great addition. When I ran this, it ran for around 80 hours till completion

The "fix_stock_price" function goes through each set of requests and attempts to
fix failures.  There are 2 types of failures.  The first is a request failure where
the requests returns nothing or a 404 or something, these ones are fixed by running
the request again.  The other type of failure is when the ticker for the stock
has changed. These are fixed by substituting the new ticker into the request and
running it again. These ticker transformations are generated in ticker_alias.py. This
repair took around 12 hours for completion

The final function takes all the requests and puts them into a single, well formatted
database.  This was done separately becuase I wanted to preserve the whole reqest
response because A. they're expensive and B. we may want to include more info
later.
"""

import pandas as pd
from pandas.tseries.offsets import BDay
import requests
import tqdm
import os
import time
import json
import sys
from multiprocessing.pool import ThreadPool

#raise Exception("Read the top of the file before proceeding")


DOWNLOAD_THREADS = 3
IEX_FINANCE_URL="https://cloud.iexapis.com/stable/stock/{ticker}/chart/date/{date}?chartCloseOnly=True&chartByDay=True&token={token}"
IEX_TOKEN=os.getenv("IEX_TOKEN")


# Get a historical stock price, return error if error
def get_historical(args):
    ticker, date = args
    str_date = date.strftime("%Y%m%d")
    result=None
    try:
        url = IEX_FINANCE_URL.format(ticker=ticker, date=str_date, token=IEX_TOKEN)
        result = requests.get(url)
        js = result.json()[0]
    except IndexError as e:
        js = {"error": -1, "date": str_date, "ticker": ticker}
    except Exception as e:
        print(e)
        js = {"error": -2, "date": str_date, "ticker": ticker}

    return js


# Get bulk stock prices in parallel
def get_bulk_stock_price(financial_data, qe_col="report_date", ticker_col="ticker", days_before=10, days_after=10):
    df = pd.read_csv(financial_data)
    stock_price_df = pd.DataFrame()
    df[qe_col] = pd.to_datetime(df[qe_col])
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        ticker = row[ticker_col]
        quarter_end = row[qe_col]
        quarter_end_str = quarter_end.strftime("%Y%m%d")
        cache_file = "cache/{}_{}.json".format(ticker, quarter_end_str)

        # Get the previous and past N business days
        dates = [quarter_end + BDay(i) for i in range(-days_before, days_after)]

        # Only download it if it hasn't
        if not os.path.exists(cache_file):

            # Download stock prices from those days
            with ThreadPool(DOWNLOAD_THREADS) as p:
                results = p.map(get_historical, zip([ticker]*len(dates), dates))

            # Write the requests to the cache
            with open(cache_file, "w") as f:
                f.write(json.dumps(results))


# Fill in missing stock prices due to errors and such
def fix_stock_price(ticker_transformation, cache="cache"):
    tt = pd.read_csv(ticker_transformation)
    tt = {t["ticker"]: t["ticker_alias"] for _, t in tt.iterrows}

    fixed = 0
    prev_fixed = 0
    for cache_file in tqdm.tqdm(os.listdir(cache)):
        with open(cache + "/" + cache_file, "r") as f:
            js = json.load(f)

        # Check to see if whole stock was an error, fix with stock transformation
        if len(js) == sum([1 for j in js if "error" in j]):
            if js[0]["ticker"] in tt:
                fixed += len(js)
                new_ticker = tt[js[0]["ticker"]]
                js = [get_historical((new_ticker, pd.to_datetime(request["date"]))) for request in js]
                print(js)

        # Fix individuals
        else:
            for i, request in enumerate(js):
                if "error" in request.keys():
                    fixed += 1
                    ticker = request["ticker"]
                    date = pd.to_datetime(request["date"])
                    js[i] = get_historical((ticker, date))
                    print(js[i])

        # Write the fixed if applicable
        if fixed > prev_fixed:
            with open(cache + "/" + cache_file, "w") as f:
                json.dump(js, f)
            prev_fixed = fixed


# From the cache, make a stock price table and write
def extract_stock_table_from_cache(cache="cache", output="data/stock_price_data.csv", days_before=10, days_after=10):
    df = pd.DataFrame()
    for cache_file in tqdm.tqdm(os.listdir(cache)):
        with open(cache + "/" + cache_file, "r") as f:
            js = json.load(f)
        if len(js) != sum([1 for j in js if "error" in j]):
            ticker = "_".join(cache_file.split("_")[:-1])
            report_date = cache_file.split("_")[-1].split(".")[0]
            report_date = pd.to_datetime(report_date)

            for r, i in zip(js, list(range(-days_before, days_after))):
                if "close" in r:
                    r["relative"] = i


            df_t = pd.DataFrame({
                "close": [r["close"] for r in js if "close" in r],
                "high": [r["high"] for r in js if "close" in r],
                "low": [r["low"] for r in js if "close" in r],
                "volume": [r["volume"] for r in js if "close" in r],
                "date": [r["date"] for r in js if "close" in r],
                "relative_day": [r["relative"] for r in js if "close" in r]
            })

            df_t["ticker"] = ticker
            df_t["report_date"] = report_date

            df = df.append(df_t, ignore_index=True)
    print(df)
    df.to_csv("test.csv", index=False)


FINANCIAL_DATA = "data/all_financials.csv"
STOCK_PRICE_OUTPUT = "data/stock_prices.csv"
TICKER_TRANSFORMATIONS = "data/ticker_alias.csv"
OUTPUT = "data/appended_stock_price.csv"
DAYS_BEFORE = 10
DAYS_AFTER = 10
VALUE = "closing"
QUARTER_END_COL = "Quarter end"
TICKER_COL = "ticker"


if __name__ == "__main__":
    #get_bulk_stock_price(FINANCIAL_DATA, qe_col=QUARTER_END_COL)
    #fix_stock_price(TICKER_TRANSFORMATION)
    extract_stock_table_from_cache()
