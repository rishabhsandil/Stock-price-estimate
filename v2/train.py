"""
v2 training — momentum + market context features, stacking ensemble,
calibrated probabilities, threshold tuning, and a long-only backtest
of the high-confidence picks.

Target: outperform SPY over the next 90 days (cross-sectional / market-neutral).
"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load():
    df = pd.read_csv(DATA, parse_dates=["qend"]).sort_values("qend").reset_index(drop=True)
    drop_cols = {"ticker", "qend", "fwd_ret_90d", "spy_fwd_ret_90d",
                 "excess_ret_90d", "target", "target_up"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df, feature_cols


def make_pipeline(model, numeric_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), numeric_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def build_models():
    return {
        "GaussianNB":   GaussianNB(),
        "MLP":          MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400,
                                        random_state=42, early_stopping=True),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                                                n_jobs=-1, random_state=42),
        "XGBoost":      XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                        eval_metric="logloss", n_jobs=-1, random_state=42),
        "LightGBM":     LGBMClassifier(n_estimators=400, max_depth=-1, num_leaves=31,
                                        learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, reg_lambda=1.0,
                                        n_jobs=-1, random_state=42, verbose=-1),
        "Stacked":      StackingClassifier(
                            estimators=[
                                ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                       eval_metric="logloss", n_jobs=-1, random_state=42)),
                                ("lgb", LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                        n_jobs=-1, random_state=42, verbose=-1)),
                                ("rf",  RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
                                                                n_jobs=-1, random_state=42)),
                            ],
                            final_estimator=LogisticRegression(max_iter=1000, C=1.0),
                            cv=3, n_jobs=-1, passthrough=False),
    }


def evaluate(df, feature_cols, n_splits=5):
    X = df[feature_cols]
    y = df["target"].astype(int).values
    fwd = df["fwd_ret_90d"].values
    spy_fwd = df["spy_fwd_ret_90d"].values
    excess = df["excess_ret_90d"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows, backtest_rows = [], []
    models = build_models()

    for name, model in models.items():
        pipe = make_pipeline(model, feature_cols)
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            # Calibrate probabilities on training data via inner CV
            cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
            cal.fit(X.iloc[tr], y[tr])

            yprob = cal.predict_proba(X.iloc[te])[:, 1]

            # Default threshold = 0.5
            yhat = (yprob >= 0.5).astype(int)

            acc = accuracy_score(y[te], yhat)
            try:
                auc = roc_auc_score(y[te], yprob)
            except ValueError:
                auc = np.nan
            prec = precision_score(y[te], yhat, zero_division=0)
            rec  = recall_score(y[te], yhat, zero_division=0)
            f1   = f1_score(y[te], yhat, zero_division=0)
            rows.append(dict(model=name, fold=fold, n_train=len(tr), n_test=len(te),
                             accuracy=acc, auc=auc, precision=prec, recall=rec, f1=f1))

            # High-confidence picks: top tertile by probability per fold
            n_picks = max(3, int(len(yprob) * 0.33))
            top_idx_in_fold = np.argsort(yprob)[-n_picks:]
            picks_mask = np.zeros(len(yprob), dtype=bool)
            picks_mask[top_idx_in_fold] = True

            picks_fwd = fwd[te][picks_mask]
            picks_excess = excess[te][picks_mask]
            cost = 0.001
            strat_ret = picks_fwd.mean() - cost if len(picks_fwd) else 0.0
            excess_strat = picks_excess.mean() - cost if len(picks_excess) else 0.0
            base_ret = fwd[te].mean()
            spy_ret = spy_fwd[te].mean()
            hit = (picks_fwd > 0).mean() if len(picks_fwd) else np.nan
            beat_spy_rate = (picks_excess > 0).mean() if len(picks_excess) else np.nan

            backtest_rows.append(dict(model=name, fold=fold, n_picks=int(len(picks_fwd)),
                                       hit_rate=hit, beat_spy_rate=beat_spy_rate,
                                       mean_fwd_ret=strat_ret,
                                       mean_excess_ret=excess_strat,
                                       baseline_ret=base_ret, spy_ret=spy_ret))

    metrics_df = pd.DataFrame(rows)
    backtest_df = pd.DataFrame(backtest_rows)
    return metrics_df, backtest_df


def summarize(metrics_df, backtest_df):
    summary = metrics_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        auc_mean=("auc", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
    ).round(4)
    bt = backtest_df.groupby("model").agg(
        hit_rate=("hit_rate", "mean"),
        beat_spy_rate=("beat_spy_rate", "mean"),
        strat_ret=("mean_fwd_ret", "mean"),
        excess_ret=("mean_excess_ret", "mean"),
        baseline_ret=("baseline_ret", "mean"),
        spy_ret=("spy_ret", "mean"),
        avg_picks=("n_picks", "mean"),
    ).round(4)
    return summary.join(bt)


def shap_importance(df, feature_cols):
    import shap
    n = len(df); cut = int(n * 0.8)
    tr_df, te_df = df.iloc[:cut], df.iloc[cut:]
    pipe = make_pipeline(
        XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      eval_metric="logloss", n_jobs=-1, random_state=42),
        feature_cols,
    )
    pipe.fit(tr_df[feature_cols], tr_df["target"].astype(int).values)
    pre = pipe.named_steps["pre"]; model = pipe.named_steps["model"]
    Xte_t = pre.transform(te_df[feature_cols])
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array(feature_cols)
    sv = shap.TreeExplainer(model).shap_values(Xte_t)
    importance = np.abs(sv).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": importance}) \
              .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"), index=False)
    return imp_df


def main():
    df, feature_cols = load()
    print(f"Loaded {len(df)} rows | features={len(feature_cols)} | beat-SPY prior={df['target'].mean():.3f}")
    metrics, backtest = evaluate(df, feature_cols)
    summary = summarize(metrics, backtest)
    metrics.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)
    backtest.to_csv(os.path.join(RESULTS_DIR, "fold_backtest.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    print("\n=== Summary (TimeSeriesSplit, 5 folds, target = beat SPY 90d) ===")
    print(summary.to_string())

    print("\n=== SHAP top 12 features ===")
    imp = shap_importance(df, feature_cols)
    print(imp.head(12).to_string(index=False))

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({
            "n_rows": int(len(df)),
            "n_tickers": int(df["ticker"].nunique()),
            "date_range": [str(df["qend"].min()), str(df["qend"].max())],
            "class_prior": float(df["target"].mean()),
            "summary": json.loads(summary.reset_index().to_json(orient="records")),
            "top_features": imp.head(12).to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
