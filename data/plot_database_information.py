import plotnine as gg
import pandas as pd
import json


def plot_db_info(db_info, out="database_info.png"):
    db_info = db_info.sort_values("missing", ascending=False).reset_index()[["missing", "database_length", "feature"]].reset_index()
    print(db_info)

    plt = (gg.ggplot(gg.aes(x="index", y="database_length"), db_info)
        + gg.geom_line(size=1)
        + gg.xlab("Number of Features Removed")
        + gg.ylab("Data Set Size")
        + gg.ggtitle("")
        + gg.scale_x_continuous(expand=(0, 0), breaks=range(0, 46, 3), minor_breaks=range(0, 46))
        + gg.scale_y_continuous(breaks=range(0, 35000, 5000), minor_breaks=range(0, 35000, 2500))
        + gg.themes.theme_bw())

    plt.save(out, width=5, height=2, dpi=300)


def load_db_info(db_info_file):
    with open(db_info_file, "r") as f:
        js = json.load(f)
    del js["nothing"]
    db = {k: [] for k in js[list(js)[0]].keys()}
    db["feature"] = []
    for j, jv in js.items():
        for k, v in jv.items():
            db[k].append(v)
        db["feature"].append(j)
    return pd.DataFrame(db)


info = load_db_info("database_information.json")
plot_db_info(info)
