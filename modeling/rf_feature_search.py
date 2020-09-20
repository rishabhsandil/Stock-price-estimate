import tqdm
import pandas as pd
import numpy as np
from sklearn import model_selection
import fit_evaluate_models as fev




def main_greedy(data_set, clf):
    X, Y = fev.load_format_data(data_set)
    X_train, X_test, Y_train, Y_test = fev.split_test_train(X, Y)
    names_X, names_Y = fev.get_feature_names(data_set)


    # Set the current best to the prior
    included_features = []
    dataframe = pd.DataFrame()
    best_score = sum(Y_train) / len(Y_train)
    prev_features = np.zeros((len(X_train), 0))
    for i, name in tqdm.tqdm(enumerate(names_X), total=len(names_X)):
        curr_features = np.concatenate([prev_features, np.expand_dims(X_train[:, i], 1)], axis=1)
        cv = model_selection.StratifiedKFold()
        score = np.mean(model_selection.cross_val_score(clf, curr_features, Y_train, cv=cv))
        if score > best_score:
            best_score = score
            print("Best Score:", best_score)
            prev_features = curr_features
            included_features.append(i)
            dataframe = dataframe.append({"cv_score": score, "features": name, "feature_i": i}, ignore_index=True)

    print("Included Features:", included_features)
    print("Best CV Score:", best_score)
    clf.fit(prev_features, Y_train)
    print("Best Test Score:", clf.score(X_test[:, included_features], Y_test))
    return dataframe



def main_loo(data_set, clf):
    X, Y = fev.load_format_data(data_set)
    X_train, X_test, Y_train, Y_test = fev.split_test_train(X, Y)
    names_X, names_Y = fev.get_feature_names(data_set)


    # Set the current best to the prior
    removed_features = []
    dataframe = pd.DataFrame()

    # Score to beat is ALL features
    cv = model_selection.StratifiedKFold()
    score_to_beat = np.mean(model_selection.cross_val_score(clf, X_train, Y_train, cv=cv))
    for i, name in tqdm.tqdm(enumerate(names_X), total=len(names_X)):
        curr_features = np.delete(X_train, i, 1)
        cv = model_selection.StratifiedKFold()
        score = np.mean(model_selection.cross_val_score(clf, curr_features, Y_train, cv=cv))
        row = {"cv_score": score, "feature": name, "feature_i": i}
        if score > score_to_beat:
            removed_features.append(i)
            row["removed"] = 1
        else:
            row["removed"] = 0
        dataframe = dataframe.append(row, ignore_index=True)

    return dataframe



if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_jobs=12, n_estimators=2000, criterion="entropy", min_samples_split=0.0001)
    #df = main_greedy("../data/databases/database_1.csv", clf)
    #df.to_csv("rf_feature_search_greedy2.csv", index=False)

    df = main_loo("../data/databases/database_1.csv", clf)
    try:
        df.to_csv("rf_feature_search_loo2.csv", index=False)
    except:
        df.to_csv("C:/Users/Josh Hartmann/Documents/out.csv", index=False)
