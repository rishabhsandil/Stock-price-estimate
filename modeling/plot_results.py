import plotnine as gg
import pandas as pd

def load_results(files):
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs)



def plot_results(df):
    df["data_set"] = df["data_set"].astype('category')
    df["data_set"].cat.reorder_categories(["database_"+str(i) for i in [1,5,10,15,20,25,30,35,40,45]], inplace=True)
    df["data_set"] = df["data_set"].apply(lambda x: x.replace("database_", "").title())

    print(df.columns)

    plt = (gg.ggplot(gg.aes(x="data_set", group="model", y="train_cv_error", fill="model"), df)
        + gg.geom_col(position='dodge')
        + gg.geom_hline(gg.aes(yintercept=0.5), linetype="dotted")
        + gg.scale_y_continuous(expand=(0 , -0.4, 0, 0.05))
        + gg.ylab("Training Accuracy")
        + gg.xlab("Number of Report Features Removed")
        + gg.labs(fill="")
        + gg.themes.theme_bw())

    plt.save("train_results_close_1.png", width=6, height=3)



if __name__ == "__main__":
    FILES = ["results_close_1/{}_results.csv".format(m) for m in ["nb", "rf", "mlp"]]
    df = load_results(FILES)
    plot_results(df)
