"""
Some of the tickers change, for example, American Airlines used to be AMR it is
AAL, this script pulls the changes into a dict and writes the json
"""

import pandas as pd
import requests

STOCK_PUP_URL = "http://www.stockpup.com/data/"

# Get a list of all the tickers
page_text = requests.get(STOCK_PUP_URL).text

# Get the area with the list of tickers
download_table = page_text.split(">Download</h4>")[1].split("</div>")[0]

# Split each ticker
tickers_p = download_table.split("</p>")[0:-1]

# Clean each ticker
tickers = [t.split("href=\"/data/")[-1].split("_quarterly_financial")[0] for t in tickers_p]

# Get Updated ticker
tickers_updated = [t.split("<b>")[-1].split("</b>")[0] for t in tickers_p]

update_dict = {}
for t, tu in zip(tickers, tickers_updated):
    if t != tu:
        update_dict[t] = tu

df = pd.DataFrame(list(update_dict.items()), columns=["ticker", "ticker_alias"])
df.to_csv("data/ticker_alias.csv", index=False)
