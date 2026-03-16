# task6.py
import os
import pickle
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


DATA_DIR = "cifar-10-batches-py"
PCA_CACHE = "cifar10_pca200.npz"
N_COMPONENTS = 200


def load_batch(batch_path):
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X = batch[b"data"]
    y = np.array(batch[b"labels"])
    return X, y


def load_cifar10(data_dir):
    X_all, y_all = [], []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)

    X_train = np.concatenate(X_all).astype(np.float32) / 255.0
    y_train = np.concatenate(y_all)

    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def get_reduced_data():
    if os.path.exists(PCA_CACHE):
        data = np.load(PCA_CACHE)
        return data["X_train_pca"], data["y_train"], data["X_test_pca"], data["y_test"]

    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)
    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="randomized",
        random_state=42
    )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    np.savez_compressed(
        PCA_CACHE,
        X_train_pca=X_train_pca,
        y_train=y_train,
        X_test_pca=X_test_pca,
        y_test=y_test
    )
    return X_train_pca, y_train, X_test_pca, y_test


def main():
    print("Loading reduced CIFAR-10 data...")
    X_train_pca, y_train, X_test_pca, y_test = get_reduced_data()
    print("X_train_pca shape:", X_train_pca.shape)
    print("X_test_pca shape :", X_test_pca.shape)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 40],
        "max_features": ["sqrt", 0.3]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("Fitting RandomForest with cross-validation...")
    grid.fit(X_train_pca, y_train)

    print("Best hyperparameters:", grid.best_params_)
    print("Best cross-validation accuracy: {:.4f}".format(grid.best_score_))

    best_rf = grid.best_estimator_
    test_accuracy = best_rf.score(X_test_pca, y_test)
    print("Random Forest test accuracy: {:.4f}".format(test_accuracy))


if __name__ == "__main__":
    main()
