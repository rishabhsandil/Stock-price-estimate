import pandas as pd
import numpy as np
import tqdm
import json
from sklearn import model_selection


CLASSIFICATION_FEATURE = "close_1"
IGNORE_FEATURES = [*[s.format(n) for n in range(1,10) for s in ["close_{}", "range_{}", "volume_{}"]], "date", "ticker"]
def load_format_data(data_set):
    df = pd.read_csv(data_set)

    # Get the input and output
    out = df[CLASSIFICATION_FEATURE]
    inp = df[[c for c in df.columns if c not in IGNORE_FEATURES]]

    # Convert input categorical features to one-hot encoded features
    category_cols = inp.select_dtypes(include='object').columns
    for c in category_cols:
        inp = pd.concat([inp, pd.get_dummies(inp[c], prefix=c)], axis=1)
    inp = inp[[c for c in inp.columns if c not in category_cols]]

    # Convert output to + -
    out = out.apply(lambda x: x > 0)
    print("Prior:", sum(out)/len(out))

    # Convert to numpy
    X = inp.to_numpy()
    Y = out.to_numpy()

    return X, Y


def get_feature_names(data_set):
    df = pd.read_csv(data_set)

    # Get the input and output
    out = df[CLASSIFICATION_FEATURE]
    inp = df[[c for c in df.columns if c not in IGNORE_FEATURES]]

    # Convert input categorical features to one-hot encoded features
    category_cols = inp.select_dtypes(include='object').columns
    for c in category_cols:
        inp = pd.concat([inp, pd.get_dummies(inp[c], prefix=c)], axis=1)
    inp = inp[[c for c in inp.columns if c not in category_cols]]

    return inp.columns, CLASSIFICATION_FEATURE



def gather_grid_search_results(grid_search, X_test, Y_test, X_train, Y_train):
    test_error = grid_search.best_estimator_.score(X_test, Y_test)
    train_cv_error = grid_search.best_score_
    model_name = grid_search.best_estimator_.__class__.__name__
    model_parameters = grid_search.best_params_

    results = pd.DataFrame({
        "test_error": [test_error],
        "train_cv_error": [train_cv_error],
        "model": [model_name],
        "parameters": json.dumps(model_parameters)
        })

    all_results = pd.DataFrame(grid_search.cv_results_)

    return results, all_results



TT_RATIO = 0.8
def split_test_train(X, Y):
    return model_selection.train_test_split(X, Y, train_size=TT_RATIO, random_state=42)



def run_grid_search(model, parameters, X, Y):
    # Split into test train
    X_train, X_test, Y_train, Y_test = split_test_train(X, Y)

    # Run the grid search
    grid_search = model_selection.GridSearchCV(model, parameters, n_jobs=5, cv=5, verbose=0, pre_dispatch='n_jobs', return_train_score=True, refit=True)
    grid_search.fit(X_train, Y_train)

    results, all_results = gather_grid_search_results(grid_search, X_test, Y_test, X_train, Y_train)
    return results, all_results, grid_search



def run_all_data_sets(model, parameters, data_sets, fun=run_grid_search):
    results = []
    all_results = []
    models = []
    for d in tqdm.tqdm(data_sets):
        X, Y = load_format_data(d)
        res, a, mod = fun(model, parameters, X, Y)
        res["data_set"] = d.split("/")[-1].split(".")[0]
        a["data_set"] = d.split("/")[-1].split(".")[0]
        models.append(mod)
        results.append(res)
        all_results.append(a)
    results = pd.concat(results)
    all_results = pd.concat(all_results)
    return results, all_results, models



if __name__ == "__main__":
    #from sklearn.neural_network import MLPClassifier
    #from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier

    a, b, c = run_all_data_sets(RandomForestClassifier(n_jobs=12, n_estimators=2000, criterion="entropy", min_samples_split=0.0001), {}, ["results_loo/database_loo.csv"])
    a.to_csv("results.csv")
    b.to_csv("all_results.csv")
