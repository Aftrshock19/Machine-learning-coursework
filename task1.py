import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_ohe():
    #to be able to do on diffrent laptops with dif versions od skitlearn
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def main():
    file_name = "regression_insurance.csv"
    data = pd.read_csv(file_name)

    X = data.drop(columns=["charges"])
    y = data["charges"]
 
    X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2, random_state=42 )

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
 
    numeric_transformer = "passthrough"
    categorical_transformer = make_ohe()

    preprocessor = ColumnTransformer( transformers=[ ("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features), ], remainder="drop",  verbose_feature_names_out=False, )

    model = Pipeline(  steps=[ ("preprocess", preprocessor), ("regressor", LinearRegression()), ]  )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = rmse(y_train, y_train_pred)
    test_rmse = rmse(y_test, y_test_pred)

    print("=== Linear Regression RMSE ===")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test  RMSE: {test_rmse:.3f}")
    print()

    linreg = model.named_steps["regressor"]
    fitted_pre = model.named_steps["preprocess"]

    feature_names = list(fitted_pre.get_feature_names_out())
    coefs = linreg.coef_

    print("=== Linear Regression Coefficients ===")
    print(f"Intercept: {linreg.intercept_:.3f}")
    for name, coef in zip(feature_names, coefs):
        print(f"{name}: {coef:.3f}")
    plt.figure()
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title("Linear regression: predicted vs actual (test set)")

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig("task1_pred_vs_actual.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
