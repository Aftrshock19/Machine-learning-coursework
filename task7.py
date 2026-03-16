# task7.py
import os
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt

DATA_DIR = "cifar-10-batches-py"
PCA_CACHE = "cifar10_pca200.npz"
SVM_TRAIN_SIZE = 10000
RANDOM_STATE = 42

CLASS_NAMES = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" ]


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
    if not os.path.exists(PCA_CACHE):
        raise FileNotFoundError(
            f"Missing {PCA_CACHE}. Run task4.py first to generate PCA-reduced data."
        )
    data = np.load(PCA_CACHE)
    return data["X_train_pca"], data["y_train"], data["X_test_pca"], data["y_test"]


def show_and_save_misclassified_images(X_test_raw, y_test, y_pred, filename="misclassified_svm.png"):
    mis_idx = np.where(y_pred != y_test)[0]
    print("Number of misclassified test examples:", len(mis_idx))

    if len(mis_idx) == 0:
        print("No misclassifications found – nothing to plot.")
        return

    chosen = mis_idx[: min(5, len(mis_idx))]

    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(chosen, start=1):
        img = X_test_raw[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        true_label = CLASS_NAMES[int(y_test[idx])]
        pred_label = CLASS_NAMES[int(y_pred[idx])]

        plt.subplot(1, len(chosen), i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"T:{true_label}\nP:{pred_label}")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved misclassified image grid to {filename}")


def main():
    print("Loading reduced CIFAR-10 data...")
    X_train_pca, y_train, X_test_pca, y_test = get_reduced_data()
    print("X_train_pca shape:", X_train_pca.shape)
    print("X_test_pca shape :", X_test_pca.shape)

    X_sub, _, y_sub, _ = train_test_split( X_train_pca, y_train, train_size=SVM_TRAIN_SIZE, stratify=y_train, random_state=RANDOM_STATE )
    print(f"Using subset of size {X_sub.shape[0]} for SVM tuning (CV).")

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(decision_function_shape="ovr"))
    ])

    param_grid = [ {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]}, {"svc__kernel": ["rbf"], "svc__C": [1, 10], "svc__gamma": ["scale", 0.01, 0.001]}, ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("Fitting SVM with cross-validation on subset...")
    grid.fit(X_sub, y_sub)

    print("Best hyperparameters:", grid.best_params_)
    print("Best CV accuracy (subset): {:.4f}".format(grid.best_score_))
    best_svm = grid.best_estimator_
    print("Refitting best SVM on full training set (50k)...")
    best_svm.fit(X_train_pca, y_train)

    test_accuracy = best_svm.score(X_test_pca, y_test)
    print("SVM TEST accuracy (trained on full train): {:.4f}".format(test_accuracy))

    if os.path.isdir(DATA_DIR):
        print("Loading raw CIFAR-10 test images for misclassification visualisation...")
        _, _, X_test_raw, y_test_raw = load_cifar10(DATA_DIR)
        assert np.array_equal(y_test, y_test_raw), "Mismatch between y_test arrays!"

        print("Predicting labels on test set...")
        y_pred = best_svm.predict(X_test_pca)
        show_and_save_misclassified_images(X_test_raw, y_test, y_pred)
    else:
        print(f"Skipping misclassified plot (could not find {DATA_DIR}).")


if __name__ == "__main__":
    main()
