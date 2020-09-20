import pandas as pd
import plotnine as gg






def plot_stock_price_before_after_range(df, out="before_after_range.png", what="price"):
    close_cols = [c for c in df.columns if "close" in c]
    close_cols.remove("close_b1")
    df = df[close_cols]

    df = df.melt(var_name="time", value_name=what)
    df["relative_time"] = df["time"].apply(lambda x: int(x.split("_")[-1]) if not "b" in x else -int(x.split("b")[-1]))
    df[what] = df[what] * 100 # percent

    plt = (gg.ggplot(gg.aes(y=what, x="factor(relative_time)"), df)
        + gg.geom_violin(color="black", fill="black")
        + gg.geom_hline(yintercept=0, linetype="dashed", color="#FF0000")
        + gg.ylab("% Relative Stock Price Change")
        + gg.xlab("Days Since Report Release")
        + gg.scale_y_continuous(limits=(-10, 10), expand=(0,0))
        + gg.themes.theme_bw())

    plt.save(out, dpi=600, width=10, height=5)




def plot_stock_price_before_after_range_2(df, out="before_after_range_2.png", what="price"):
    close_cols = [c for c in df.columns if "close" in c]
    df = df[close_cols]

    df = df.melt(var_name="time", value_name=what)
    df["relative_time"] = df["time"].apply(lambda x: int(x.split("_")[-1]) if not "b" in x else -int(x.split("b")[-1]))
    df[what] = df[what] * 100 # percent

    df_new = pd.DataFrame()
    for name, group in df.groupby(["relative_time"]):
        df_new = df_new.append({
            "5th Percentile": group[what].quantile(q=0.05),
            "Median": group[what].quantile(q=0.5),
            "95th Percentile": group[what].quantile(q=0.95),
            "relative_time": group["relative_time"].unique()[0]},
            ignore_index = True
            )



    print(df_new)

    plt = (gg.ggplot(gg.aes(y="Median", x="relative_time", ymin="5th Percentile", ymax="95th Percentile"), df_new)
        + gg.geom_line(color="black", size=0.8)
        + gg.geom_ribbon(fill="#00000044")
        + gg.geom_hline(yintercept=0, linetype="dashed", color="#FF0000", size=0.3)
        + gg.geom_vline(xintercept=1, linetype="dashed", color="#0000FF", size=0.3)
        + gg.ylab(f"% Relative Stock {what.title()} Change")
        + gg.xlab("Days Since Report Release")
        + gg.ggtitle("Stock Trade Price")
        + gg.scale_y_continuous(limits=(-11, 11), expand=(0,0))
        + gg.scale_x_continuous(breaks=range(-10,11, 2))
        + gg.themes.theme_bw())

    plt.save(out, dpi=600, width=4, height=3)



def plot_stock_volume_before_after_range(df, out="before_after_volume.png"):
    close_cols = [c for c in df.columns if "volume" in c]
    close_cols.remove("volume_b1")
    df = df[close_cols]

    df = df.melt(var_name="time", value_name="volume")
    df["relative_time"] = df["time"].apply(lambda x: int(x.split("_")[-1]) if not "b" in x else -int(x.split("b")[-1]))
    df["volume"] = df["volume"] * 100 # percent

    plt = (gg.ggplot(gg.aes(y="volume", x="factor(relative_time)"), df)
        + gg.geom_violin(color="black", fill="black")
        + gg.geom_hline(yintercept=0, linetype="dashed", color="#FF0000")
        + gg.ylab("% Relative Trade Volume Change")
        + gg.xlab("Days Since Report Release")
        + gg.scale_y_continuous(limits=(-100, 200), expand=(0,0))
        + gg.themes.theme_bw())

    plt.save(out, dpi=600, width=10, height=5)



def plot_report_correlation(df, out="report_correlation.png"):
    df = df[["shares_split_adjusted", "split_factor", "assets", "current_assets",
        "liabilities", "current_liabilities", "shareholders_equity", "noncontrolling_interest",
        "preferred_equity", "goodwill_intangibles", "longterm_debt", "revenue", "earnings",
        "earnings_available_for_common_stockholders", "eps_basic", "eps_diluted", "dividend_per_share",
        "cash_from_operating_activities", "cash_from_investing_activities", "cash_from_financing_activities",
        "cash_change_during_period", "cash_at_end_of_period", "capital_expenditures", "price",
        "price_high", "price_low", "roe", "roa", "book_value_of_equity_per_share", "pb_ratio",
        "pe_ratio", "cumulative_dividends_per_share", "dividend_payout_ratio",
        "longterm_debt_to_equity_ratio", "equity_to_assets_ratio", "net_margin", "asset_turnover",
        "free_cash_flow_per_share", "current_ratio"]]
    df_corr = df.corr().reset_index()
    df_corr = df_corr.melt(id_vars=["index"], var_name="index_2", value_name="correlation")
    df_corr["Absolute Correlation"] = df_corr["correlation"].abs()

    plt = (gg.ggplot(gg.aes(y="index", x="index_2", fill="Absolute Correlation"), df_corr)
        + gg.geom_tile()
        + gg.themes.theme_bw()
        + gg.ylab("")
        + gg.xlab("")
        + gg.theme(axis_text_x=gg.element_text(rotation=90, hjust=1)))

    plt.save(out, dpi=600, width=10, height=10)
    print(df_corr)



def plot_everything(df, out="histograms/histogram_{}.png"):
    df = df.melt(id_vars=["date"])
    for v in df.variable.unique():
        df_v = df[df["variable"] == v]
        plt = (gg.ggplot(gg.aes(x="value", group="variable"), df_v)
            + gg.geom_histogram()
            + gg.themes.theme_bw())
        print("plotting", v)
        plt.save(out.format(v.lower().replace(" ", "_")), dpi=150, width=5, height=2, limitsize=False)




def plot_index_correlation(df, out="index_correlation.png"):
    df = df[["dow_idx", "dow_idx_vol", "sp500_idx", "sp500_idx_vol", "nasdaq_idx",
            "nasdaq_idx_vol", "tsx_idx", "tsx_idx_vol"]]
    df = df.rename(columns={"dow_idx": "Dow", "dow_idx_vol": "Volume Dow", "sp500_idx": "S&P 500",
            "sp500_idx_vol": "Volume S&P 500", "nasdaq_idx": "NASDAQ", "nasdaq_idx_vol": "Volume NASDAQ",
            "tsx_idx": "TSX", "tsx_idx_vol": "Volume TSX"})


    df_corr_idx = df[[c for c in df.columns if "Volume" not in c]]
    df_corr_idx = df_corr_idx.corr().reset_index()
    df_corr_idx = df_corr_idx.melt(id_vars=["index"], var_name="index_2", value_name="correlation")
    df_corr_idx["value"] = "Closing Price"

    df_corr_vol = df[[c for c in df.columns if "Volume" in c]]
    df_corr_vol = df_corr_vol.rename(columns={c: c.replace("Volume ", "") for c in df_corr_vol.columns})
    df_corr_vol = df_corr_vol.corr().reset_index()
    df_corr_vol = df_corr_vol.melt(id_vars=["index"], var_name="index_2", value_name="correlation")
    df_corr_vol["value"] = "Volume"

    df_corr = pd.concat([df_corr_idx, df_corr_vol])
    df_corr["Correlation"] = df_corr_idx["correlation"].abs()


    plt = (gg.ggplot(gg.aes(y="index", x="index_2", fill="Correlation"), df_corr)
        + gg.geom_tile()
        + gg.themes.theme_bw()
        + gg.ylab("")
        + gg.xlab("")
        + gg.scale_fill_continuous(limits=[0,1])
        + gg.facet_wrap(facets=["value"], ncol=2, scales="free_xy")
        + gg.theme(axis_text_x=gg.element_text(rotation=90, hjust=1), legend_text=gg.element_text(va="bottom")))

    plt.save(out, dpi=600, width=4, height=2, limitsize=False)




def plot_missing_histogram(df, out="report_missing_hist.png"):
    df = df[[c for c in df.columns
        if "vol" not in c and
        "range" not in c and
        "close" not in c and
        "idx" not in c and
        c not in ["sector", "exchange", "country", "ticker", "stock_type"]]]

    df["missing"]  = df.apply(lambda x: x.isnull().sum(), axis=1)

    plt = (gg.ggplot(gg.aes(x="missing"), df)
        + gg.geom_bar(fill="black", width=1)
        + gg.ylab("# of Reports")
        + gg.xlab("Features Missing")
        + gg.scale_x_continuous(minor_breaks=4, breaks=range(0, 26, 5))
        + gg.themes.theme_bw())

    plt.save(out, dpi=600, width=4, height=2, limitsize=False)




def plot_company_info(df, out="company_info_{}.png"):
    df = df[["sector", "exchange", "country", "stock_type"]]
    for c in df.columns:
        df_c = df[[c]].dropna()
        plt = (gg.ggplot(gg.aes(x=c), df_c)
            + gg.geom_bar(fill="black")
            + gg.ggtitle("Company " + c.replace("_", " ").title())
            + gg.xlab("")
            + gg.ylab("Count")
            + gg.themes.theme_bw()
            + gg.theme(axis_text_x=gg.element_text(rotation=90, hjust=1)))

        plt.save(out.format(c), dpi=600, height=2, width=len(df[c].unique())/3)



def plot_day_week_month_hist(df, out="{}_histogram.png"):
    df["day_of_week"] = df["day_of_week"] * 7
    df["month"] = df["month"] * 12
    for d in ["day_of_week", "month"]:
        breaks = df[d].unique()
        plt = (gg.ggplot(gg.aes(x=d), df) +
            gg.geom_bar(fill="black") +
            gg.themes.theme_bw() +
            gg.xlab(d.replace("_", " ").title()) +
            gg.ylab("Count") + 
            gg.scale_x_continuous(minor_breaks=breaks, breaks=breaks))
        plt.save(out.format(d), width=3, height=2, dpi=600)
    



def load_database(path):
    df = pd.read_csv(path, na_values=["None", ",", "#N/A", "NaN", "nan", "null"])
    return df


if __name__ == "__main__":
    df = load_database("databases/database_5.csv")
    #plot_day_week_month_hist(df)
    #plot_stock_price_before_after_range_2(df)
    #plot_stock_volume_before_after_range(df)
    #plot_index_correlation(df)
    #plot_company_info(df)
    #plot_report_correlation(df)
    #plot_missing_histogram(df)
    plot_everything(df)
