# task4.py
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

DATA_DIR = "cifar-10-batches-py"
OUT_FILE = "cifar10_pca200.npz"
N_COMPONENTS = 200
RANDOM_STATE = 42


def load_batch(batch_path):
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X = batch[b"data"]
    y = np.array(batch[b"labels"])
    return X, y


def load_cifar10(data_dir):
    X_all = []
    y_all = []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)

    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
 
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return X_train / 255.0, y_train, X_test / 255.0, y_test


def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Could not find folder: {DATA_DIR}")

    X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print("Original shapes:", X_train.shape, X_test.shape) 

    print(f"Fitting PCA to {N_COMPONENTS} dims (train only)...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=RANDOM_STATE)

    X_train_200 = pca.fit_transform(X_train)
    X_test_200 = pca.transform(X_test)

    evr_sum = float(np.sum(pca.explained_variance_ratio_))
    print("Reduced shapes :", X_train_200.shape, X_test_200.shape) 
    print("Explained variance ratio sum:", evr_sum)

    print(f"Saving to {OUT_FILE} ...")
    np.savez_compressed(
        OUT_FILE,
        X_train_pca=X_train_200,
        y_train=y_train,
        X_test_pca=X_test_200,
        y_test=y_test,
        explained_variance_ratio_sum=evr_sum,
    )

    print("Done.")


if __name__ == "__main__":
    main()
