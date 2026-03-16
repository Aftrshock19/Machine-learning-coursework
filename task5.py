# task5.py
import os
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

PCA_FILE = "cifar10_pca200.npz"
OUT_MODEL = "best_decision_tree.pkl"
RANDOM_STATE = 42


def load_reduced_data(npz_path=PCA_FILE):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(   f"Missing {npz_path}. Run task4.py first to generate the PCA-reduced dataset."  )
    data = np.load(npz_path)
    X_train = data["X_train_pca"]
    y_train = data["y_train"]
    X_test = data["X_test_pca"]
    y_test = data["y_test"]
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_reduced_data()
    print("Loaded reduced data:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5, 10],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV( estimator=tree, param_grid=param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=2, refit=True,  return_train_score=False, )

    print("Running cross-validation grid search...")
    grid.fit(X_train, y_train)

    print("\nBest hyperparameters:", grid.best_params_)
    print("Best CV accuracy: {:.4f}".format(grid.best_score_))

    best_tree = grid.best_estimator_
    test_acc = best_tree.score(X_test, y_test)
    print("Decision Tree TEST accuracy: {:.4f}".format(test_acc))

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(best_tree, f)
    print(f"Saved best model to {OUT_MODEL}")


if __name__ == "__main__":
    main()
