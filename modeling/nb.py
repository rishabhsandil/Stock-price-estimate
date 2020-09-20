import pickle
from sklearn.naive_bayes import GaussianNB
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
        "var_smoothing": [10**(-i) for i in range(1, 10)]
    }
    results, all_results, grid_search = fev.run_all_data_sets(GaussianNB(), parameters, DATA_SETS)
    pickle.dump(grid_search, open("nb_fitted_model.pkl", "wb"))
    results.to_csv("nb_results.csv", index=False)
    all_results.to_csv("nb_all_results.csv", index=False)
