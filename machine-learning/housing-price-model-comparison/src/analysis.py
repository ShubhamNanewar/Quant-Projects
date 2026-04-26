from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import jarque_bera


ALPHAS = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
RANDOM_STATE = 42
SUMMARY_FEATURE = "Gymnasium"


def load_data(data_dir: Path) -> pd.DataFrame:
    return pd.read_excel(data_dir / "data_ASS_I_group_17.xlsx", sheet_name="Data")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=random_state
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_regularized_model(
    model_class,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    rows = []
    best_alpha = None
    best_loss = None

    for alpha in ALPHAS:
        model = (
            model_class(alpha=alpha, max_iter=10000, selection="random")
            if model_class is Lasso
            else model_class(alpha=alpha)
        )
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_val_scaled)
        validation_mse = mean_squared_error(y_val, prediction)
        rows.append({"alpha": alpha, "validation_mse": validation_mse})

        if best_loss is None or validation_mse < best_loss:
            best_loss = validation_mse
            best_alpha = alpha

    assert best_alpha is not None
    return float(best_alpha), pd.DataFrame(rows)


def fit_target_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    target_mode: str,
) -> dict[str, object]:
    target_scale = 1_000_000.0 if target_mode == "price" else 1.0
    train_target = np.log(y_train) if target_mode == "log" else y_train / target_scale
    val_target = np.log(y_val) if target_mode == "log" else y_val / target_scale

    best_lasso_alpha, lasso_path = tune_regularized_model(
        Lasso, X_train, train_target, X_val, val_target
    )
    best_ridge_alpha, ridge_path = tune_regularized_model(
        Ridge, X_train, train_target, X_val, val_target
    )

    lasso_scaler = StandardScaler()
    ridge_scaler = StandardScaler()
    X_train_lasso = lasso_scaler.fit_transform(X_train)
    X_test_lasso = lasso_scaler.transform(X_test)
    X_train_ridge = ridge_scaler.fit_transform(X_train)
    X_test_ridge = ridge_scaler.transform(X_test)

    models = {
        "linear": LinearRegression(),
        "lasso": Lasso(alpha=best_lasso_alpha, max_iter=10000, selection="random"),
        "ridge": Ridge(alpha=best_ridge_alpha),
    }

    predictions: dict[str, np.ndarray] = {}
    rows = []
    for model_name, model in models.items():
        if model_name == "linear":
            model.fit(X_train, train_target)
            predicted = model.predict(X_test)
        elif model_name == "lasso":
            model.fit(X_train_lasso, train_target)
            predicted = model.predict(X_test_lasso)
        else:
            model.fit(X_train_ridge, train_target)
            predicted = model.predict(X_test_ridge)
        predicted_price = (
            np.exp(predicted) if target_mode == "log" else predicted * target_scale
        )
        predictions[model_name] = predicted_price
        rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_test, predicted_price),
                "mse": mean_squared_error(y_test, predicted_price),
            }
        )

    return {
        "best_alphas": {"lasso": best_lasso_alpha, "ridge": best_ridge_alpha},
        "lasso_path": lasso_path,
        "ridge_path": ridge_path,
        "metrics": pd.DataFrame(rows).sort_values("mse").reset_index(drop=True),
        "predictions": predictions,
    }


def diebold_mariano_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float]:
    differential = errors_a**2 - errors_b**2
    statistic = differential.mean() / (differential.std(ddof=1) / np.sqrt(len(differential)))
    p_value = 2 * (1 - stats.t.cdf(np.abs(statistic), df=len(differential) - 1))
    return {"dm_stat": float(statistic), "p_value": float(p_value)}


def build_table_1(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Price", "Area", SUMMARY_FEATURE]].describe().T


def fit_log_ols_table(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    model = sm.OLS(np.log(y_train), sm.add_constant(X_train)).fit()
    keep = ["const", "Area", "No. of Bedrooms", "MaintenanceStaff", "Gymnasium", "SwimmingPool"]
    table = pd.DataFrame(
        {
            "coefficient": model.params.loc[keep],
            "std_error": model.bse.loc[keep],
        }
    )
    table.loc["R_squared", "coefficient"] = model.rsquared
    table.loc["R_squared", "std_error"] = np.nan
    return table


def plot_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["Price"], bins=30, color="#2F5D8C", alpha=0.8)
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(np.log(df["Price"]), bins=30, color="#4B8A5F", alpha=0.8)
    axes[1].set_title("Log Price Distribution")
    axes[1].set_xlabel("Log price")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tuning_curves(
    tuning_results: dict[str, object],
    robustness_results: dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_specs = [
        ("price", "lasso", axes[0, 0], tuning_results),
        ("price", "ridge", axes[0, 1], tuning_results),
        ("log", "lasso", axes[1, 0], tuning_results),
        ("log", "ridge", axes[1, 1], tuning_results),
    ]

    for target_mode, model_name, ax, result_source in plot_specs:
        frame = result_source[target_mode][f"{model_name}_path"]
        ax.plot(frame["alpha"], frame["validation_mse"], marker="o")
        ax.set_xscale("log")
        ax.set_title(f"{target_mode.title()} target: {model_name.title()} tuning")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Validation MSE")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_robustness_tuning(
    robustness_results: dict[str, object], output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_specs = [
        ("price", "lasso", axes[0, 0]),
        ("price", "ridge", axes[0, 1]),
        ("log", "lasso", axes[1, 0]),
        ("log", "ridge", axes[1, 1]),
    ]

    for target_mode, model_name, ax in plot_specs:
        frame = robustness_results[target_mode][f"{model_name}_path"]
        ax.plot(frame["alpha"], frame["validation_mse"], marker="o")
        ax.set_xscale("log")
        ax.set_title(f"70/10/20 {target_mode.title()}: {model_name.title()}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Validation MSE")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, object]:
    split = split_data(X, y, train_ratio, val_ratio, test_ratio)
    X_train, X_val, X_test, y_train, y_val, y_test = split

    price_results = fit_target_models(X_train, X_val, X_test, y_train, y_val, y_test, "price")
    log_results = fit_target_models(X_train, X_val, X_test, y_train, y_val, y_test, "log")

    best_price = price_results["metrics"].iloc[0]["model"]
    second_price = price_results["metrics"].iloc[1]["model"]
    best_log = log_results["metrics"].iloc[0]["model"]
    second_log = log_results["metrics"].iloc[1]["model"]

    dm_price = diebold_mariano_test(
        y_test.to_numpy() - price_results["predictions"][best_price],
        y_test.to_numpy() - price_results["predictions"][second_price],
    )
    dm_log = diebold_mariano_test(
        y_test.to_numpy() - log_results["predictions"][best_log],
        y_test.to_numpy() - log_results["predictions"][second_log],
    )
    dm_cross = diebold_mariano_test(
        y_test.to_numpy() - price_results["predictions"][best_price],
        y_test.to_numpy() - log_results["predictions"][best_log],
    )

    return {
        "split": split,
        "price": price_results,
        "log": log_results,
        "dm_price": {
            "model_a": best_price,
            "model_b": second_price,
            **dm_price,
        },
        "dm_log": {
            "model_a": best_log,
            "model_b": second_log,
            **dm_log,
        },
        "dm_cross": {
            "price_model": best_price,
            "log_model": best_log,
            **dm_cross,
        },
        "y_test": y_test,
    }


def write_outputs(project_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=True if frame.index.name else False)


def run_analysis(project_dir: Path) -> dict[str, object]:
    data_dir = project_dir / "data"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    df = load_data(data_dir)
    X = df.drop(columns=["Price"])
    y = df["Price"]

    main_results = evaluate_split(X, y, 0.6, 0.2, 0.2)
    robustness_results = evaluate_split(X, y, 0.7, 0.1, 0.2)

    table_1 = build_table_1(df)
    table_2a = fit_log_ols_table(main_results["split"][0], main_results["split"][3])
    table_3a = main_results["price"]["metrics"].set_index("model")
    table_3b = main_results["log"]["metrics"].set_index("model")
    table_4 = pd.DataFrame(
        {
            "60/20/20": {
                "price_lasso_alpha": main_results["price"]["best_alphas"]["lasso"],
                "price_ridge_alpha": main_results["price"]["best_alphas"]["ridge"],
                "log_lasso_alpha": main_results["log"]["best_alphas"]["lasso"],
                "log_ridge_alpha": main_results["log"]["best_alphas"]["ridge"],
            },
            "70/10/20": {
                "price_lasso_alpha": robustness_results["price"]["best_alphas"]["lasso"],
                "price_ridge_alpha": robustness_results["price"]["best_alphas"]["ridge"],
                "log_lasso_alpha": robustness_results["log"]["best_alphas"]["lasso"],
                "log_ridge_alpha": robustness_results["log"]["best_alphas"]["ridge"],
            },
        }
    )
    table_5a = robustness_results["price"]["metrics"].set_index("model")
    table_5b = robustness_results["log"]["metrics"].set_index("model")

    dm_summary = pd.DataFrame(
        [
            {"split": "60/20/20", "comparison": "price best vs second", **main_results["dm_price"]},
            {"split": "60/20/20", "comparison": "log best vs second", **main_results["dm_log"]},
            {"split": "60/20/20", "comparison": "best price vs best log", **main_results["dm_cross"]},
            {"split": "70/10/20", "comparison": "price best vs second", **robustness_results["dm_price"]},
            {"split": "70/10/20", "comparison": "log best vs second", **robustness_results["dm_log"]},
            {"split": "70/10/20", "comparison": "best price vs best log", **robustness_results["dm_cross"]},
        ]
    )

    normality = pd.DataFrame(
        {
            "series": ["Price", "log(Price)"],
            "jb_stat": [jarque_bera(y)[0], jarque_bera(np.log(y))[0]],
            "p_value": [jarque_bera(y)[1], jarque_bera(np.log(y))[1]],
        }
    )

    plot_distribution(df, figures_dir / "price_vs_logprice_distribution.png")
    plot_tuning_curves(main_results, robustness_results, figures_dir / "tuning_curves_602020.png")
    plot_robustness_tuning(robustness_results, figures_dir / "tuning_curves_701020.png")

    write_outputs(
        project_dir,
        {
            "table_1_summary_statistics": table_1,
            "table_2a_log_ols": table_2a,
            "table_3a_price_metrics": table_3a,
            "table_3b_log_metrics": table_3b,
            "table_4_robustness_hyperparameters": table_4,
            "table_5a_price_metrics_robustness": table_5a,
            "table_5b_log_metrics_robustness": table_5b,
            "diebold_mariano_tests": dm_summary,
            "normality_tests": normality,
        },
    )

    return {
        "df": df,
        "table_1": table_1,
        "table_2a": table_2a,
        "table_3a": table_3a,
        "table_3b": table_3b,
        "table_4": table_4,
        "table_5a": table_5a,
        "table_5b": table_5b,
        "dm_summary": dm_summary,
        "normality": normality,
        "main_results": main_results,
        "robustness_results": robustness_results,
    }


def print_summary(results: dict[str, object]) -> None:
    print("Main split metrics: price target")
    print(results["table_3a"].round(2).to_string())
    print("\nMain split metrics: log target")
    print(results["table_3b"].round(2).to_string())
    print("\nRobustness split metrics: price target")
    print(results["table_5a"].round(2).to_string())
    print("\nRobustness split metrics: log target")
    print(results["table_5b"].round(2).to_string())
    print("\nDiebold-Mariano tests")
    print(results["dm_summary"].round(4).to_string(index=False))


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results = run_analysis(project_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
