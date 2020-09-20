import pickle
from sklearn.neural_network import MLPClassifier
import fit_evaluate_models as fev
import warnings
warnings.filterwarnings("ignore")

DATA_SETS = [
    "../data/databases/database_1.csv",
    "../data/databases/database_5.csv",
    "../data/databases/database_10.csv",
    "../data/databases/database_15.csv",
    "../data/databases/database_20.csv",
    "../data/databases/database_25.csv",
    "../data/databases/database_30.csv",
    "../data/databases/database_35.csv",
    "../data/databases/database_40.csv",
    "../data/databases/database_45.csv"
    ]

INTERVAL_LINEAR = lambda inp, n_layers, out: [round((inp-out) * (k+1) / (n_layers+1)) for k in range(n_layers)]
INTERVAL_EXP = lambda inp, n_layers, out: [round((inp-out) * ((k+1.8) / (n_layers+1))**3) for k in range(n_layers)]
INTERVAL_LOG = lambda inp, n_layers, out: [round((inp-out) * ((k+0.01)*1.5 / (n_layers+1))**(1/3)) for k in range(n_layers)]


def generate_hidden_layer_sizes(input_size, output_size, layers, intervals):
    layer_sizes = []
    for n_layers in layers:
        for i in intervals:
            layer_sizes.append(i(input_size, n_layers, output_size))
    return layer_sizes



def run_data_set(model, parameters, X, Y):
    parameters["hidden_layer_sizes"] = generate_hidden_layer_sizes(X.shape[1], 1, range(1, 4), [INTERVAL_LINEAR])
    return fev.run_grid_search(model, parameters, X, Y)



if __name__ == "__main__":
    parameters = {
        "solver": ["lbfgs", "adam"],
        "alpha": [10**(-i) for i in range(1, 6)],
        "activation": ["logistic", "relu"]
        }
    results, all_results, grid_search = fev.run_all_data_sets(MLPClassifier(), parameters, DATA_SETS, fun=run_data_set)
    pickle.dump(grid_search, open("mlp_fitted_models.pkl", "wb"))
    results.to_csv("mlp_results.csv", index=False)
    all_results.to_csv("mlp_all_results.csv", index=False)
