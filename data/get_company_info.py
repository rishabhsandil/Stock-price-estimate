"""
This is just a hacked together script to download data about the stocks in the
analysis
"""

import pandas as pd
import requests
import tqdm
import os
import time
import json
import sys

#raise Exception("Read the top of the file before proceeding")


IEX_FINANCE_URL="https://cloud.iexapis.com/stable/stock/{ticker}/company?&token={token}"
IEX_TOKEN=os.getenv("IEX_TOKEN")


# Get a historical stock price, return error if error
def get_info(ticker):

    result=None
    try:
        url = IEX_FINANCE_URL.format(ticker=ticker, token=IEX_TOKEN)
        result = requests.get(url)
        js = result.json()
    except Exception as e:
        return None

    return js


# Get bulk stock prices in parallel
def get_stock_info(ticker_transformation, tickers, output="output.csv"):
    tt = pd.read_csv(ticker_transformation)
    tt = {t["ticker"]: t["ticker_alias"] for _, t in tt.iterrows()}

    tickers = pd.read_csv(tickers)["ticker"]

    df = pd.DataFrame()
    for t in tqdm.tqdm(tickers):
        info = get_info(t)
        if not info and t in tt: # Try alias
            info = get_info(tt[t])

        if info:
            df = df.append({
                "ticker": t,
                "exchange": info["exchange"],
                "country": info["country"],
                "sector": info["sector"],
                "type": info["issueType"]}, ignore_index=True)

    df.to_csv(output, index=False)


TICKER_TRANSFORMATIONS = "data/ticker_alias.csv"
TICKERS = "data/tickers.csv"
OUTPUT = "data/stock_information.csv"

if __name__ == "__main__":
    get_stock_info(TICKER_TRANSFORMATIONS, TICKERS)
