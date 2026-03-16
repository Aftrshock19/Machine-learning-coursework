# task2.py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_preprocessor(X_train: pd.DataFrame):
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    # sklearn changed OneHotEncoder arg name from sparse -> sparse_output
    try:
        ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", ohe, categorical_features),
        ],
        remainder="drop",
    )
    preprocessor.fit(X_train)
    return preprocessor


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def rmse(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    preds = model(X.to(device)).squeeze(1)
    mse = torch.mean((preds - y.to(device)) ** 2).item()
    return float(np.sqrt(mse))


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    file_name = "regression_insurance.csv"
    data = pd.read_csv(file_name)

    X = data.drop(columns=["charges"])
    y = data["charges"].astype(np.float32)

    # Split 80/20 with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess (fit on train only)
    preprocessor = build_preprocessor(X_train)
    X_train_np = preprocessor.transform(X_train).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    # Torch tensors
    X_train_t = torch.from_numpy(X_train_np)
    y_train_t = torch.from_numpy(y_train_np)
    X_test_t = torch.from_numpy(X_test_np)
    y_test_t = torch.from_numpy(y_test_np)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )

    # Model
    model = MLPRegressor(in_dim=X_train_np.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train
    epochs = 500
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        if epoch in {1, 50, 100, 200, 300, 400, 500}:
            train_rmse = rmse(model, X_train_t, y_train_t, device)
            test_rmse = rmse(model, X_test_t, y_test_t, device)
            avg_loss = running_loss / len(train_loader.dataset)
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f} | "
                f"Train MSE loss: {avg_loss:.3f}"
            )

    # Final RMSE prints (explicit)
    train_rmse = rmse(model, X_train_t, y_train_t, device)
    test_rmse = rmse(model, X_test_t, y_test_t, device)
    print(f"\nFinal Training RMSE: {train_rmse:.3f}")
    print(f"Final Test RMSE: {test_rmse:.3f}")

    # Plot predicted vs actual on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t.to(device)).squeeze(1).cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_np, y_pred, alpha=0.7)
    minv = float(min(y_test_np.min(), y_pred.min()))
    maxv = float(max(y_test_np.max(), y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv], linewidth=1)

    plt.title(f"Neural Network: Predicted vs Actual (Test)\nTest RMSE = {test_rmse:.3f}")
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.tight_layout()

    # Save + show (saving is helpful for report inclusion)
    plt.savefig("task2_nn_pred_vs_actual.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
