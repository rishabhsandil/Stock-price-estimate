import pickle
from sklearn.ensemble import RandomForestClassifier
import fit_evaluate_models as fev

DATA_SETS = [
    #"../data/databases/database_1.csv",
    #"../data/databases/database_5.csv",
    #"../data/databases/database_10.csv",
    #"../data/databases/database_15.csv",
    #"../data/databases/database_20.csv",
    #"../data/databases/database_25.csv",
    #"../data/databases/database_30.csv",
    "../data/databases/database_35.csv",
    "../data/databases/database_40.csv",
    "../data/databases/database_45.csv"
    ]


if __name__ == "__main__":
    parameters = {
        "n_estimators": [10, 100, 1000],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [0.1, 0.01, 0.001, 0.0001]
    }
    results, all_results, grid_search = fev.run_all_data_sets(RandomForestClassifier(), parameters, DATA_SETS)
    pickle.dump(grid_search, open("rf_fitted_model.pkl", "wb"))
    results.to_csv("rf_results.csv", index=False)
    all_results.to_csv("rf_all_results.csv", index=False)
