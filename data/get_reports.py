"""
This data is downloaded from Stock Pup. Automated downloading of data from their site is
against their terms of service so this script downloads the files slower than a human
would with a random interval of time between downloads.  Do not run this script unless
you cannot get the data otherwise and be patient.

I'm literally doing a project on this in my law and ethics class. Although it is
against their terms of service it is almost certainly legal under US law.
"""
import pandas as pd
import requests
import time
import tqdm
import random
import os

raise Exception("Read the top of the file before proceeding")


DOWNLOAD_URL = "{ticker}_quarterly_financial_data.csv"
STOCK_PUP_URL = "http://www.stockpup.com/data/"
DOWNLOAD_LOCATION = "data"

# Get a list of all the tickers
page_text = requests.get(STOCK_PUP_URL).text
time.sleep(2 + random.random()*4)

# Get the area with the list of tickers
download_table = page_text.split(">Download</h4>")[1].split("</div>")[0]

# Split each ticker
tickers_p = download_table.split("</p>")[0:-1]

# Clean each ticker
tickers = [t.split("href=\"/data/")[-1].split("_quarterly_financial")[0] for t in tickers_p]

# Write tickers to a csv file
with open("tickers.csv", "w") as f:
    for t in tickers: f.write(t + "\n")

# Make the output directory
os.makedirs(os.path.join(DOWNLOAD_LOCATION, "raw_data"), exist_ok=True)

tickers_tmp = list(tickers)
for t in tickers_tmp:
    if t == "":
        break
    tickers.remove(t)


# Download each ticker
dfs = []
for t in tqdm.tqdm(tickers):
    url = STOCK_PUP_URL + DOWNLOAD_URL.format(ticker=t)
    output = os.path.join(DOWNLOAD_LOCATION, "raw_data/{ticker}.csv".format(ticker=t))

    # Download each ticker and write to a file
    page_text = requests.get(url).text
    time.sleep(2 + random.random()*8)
    with open(output, "w") as f:
        f.write(page_text)

    # Read the data into a dataframe
    df = pd.read_csv(output)
    df["ticker"] = t
    dfs.append(df)

# Append all the dataframes
df = pd.concat(dfs)

# Write the combined dataframe
df.to_csv(os.path.join(DOWNLOAD_LOCATION, "all_financials.csv"), index=False)
