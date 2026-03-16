# task3.py
import numpy as np
import pandas as pd

import pymc as pm
import arviz as az

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err * err)))


def main():
    
    data = pd.read_csv("regression_insurance.csv")
    X = data.drop(columns=["charges"])
    y = data["charges"].to_numpy(dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    preprocessor = build_preprocessor(X_train)

    X_train_np = preprocessor.transform(X_train).astype(np.float64)
    X_test_np = preprocessor.transform(X_test).astype(np.float64)


    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"x{i}" for i in range(X_train_np.shape[1])], dtype=object)
    feat_names = [str(f).replace("num__", "").replace("cat__", "") for f in feat_names]

    n_features = X_train_np.shape[1]


    with pm.Model() as model:
        X_data = pm.Data("X_data", X_train_np)
        y_obs = pm.Data("y_obs", y_train)


        intercept = pm.Normal("intercept", mu=0.0, sigma=20000.0)
        beta = pm.Normal("beta", mu=0.0, sigma=20000.0, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=20000.0)

        mu = intercept + pm.math.dot(X_data, beta)

        pm.Normal("charges", mu=mu, sigma=sigma, observed=y_obs)

        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=42,
            progressbar=True,
        )


    post = idata.posterior

    intercept_mean = float(post["intercept"].mean().values)
    beta_mean = post["beta"].mean(dim=("chain", "draw")).values
    sigma_mean = float(post["sigma"].mean().values)

    print("\nPosterior means (PyMC Bayesian linear regression)")
    print(f"intercept: {intercept_mean:.3f}")
    for name, b in zip(feat_names, beta_mean):
        print(f"{name}: {float(b):.3f}")
    print(f"sigma (noise std): {sigma_mean:.3f}") 

    yhat_train = intercept_mean + X_train_np @ beta_mean
    yhat_test = intercept_mean + X_test_np @ beta_mean

    train_rmse = rmse(y_train, yhat_train)
    test_rmse = rmse(y_test, yhat_test)

    print(f"\nTraining RMSE (posterior-mean params): {train_rmse:.3f}")
    print(f"Test RMSE (posterior-mean params): {test_rmse:.3f}")


    summ = az.summary(idata, var_names=["intercept", "beta", "sigma"], kind="stats")
    
    cols = [c for c in ["mean", "sd", "r_hat"] if c in summ.columns]
    print("\nArviZ summary (mean/sd/r_hat):")
    print(summ[cols].to_string())


if __name__ == "__main__":
    main()
