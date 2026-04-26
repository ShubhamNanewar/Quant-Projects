from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight


FEATURES = ["wkta", "reta", "ebitta", "mv"]
SMALL_FEATURES = ["wkta", "reta"]
BETA = 2
SEED = 42


def load_data(data_dir: Path) -> pd.DataFrame:
    return pd.read_excel(data_dir / "data.xlsx", sheet_name="Data").dropna().reset_index(drop=True)


def stratified_year_split(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df[features]
    y = df["def"]
    years = df["year"]

    X_train, X_temp, y_train, y_temp, year_train, year_temp = train_test_split(
        X, y, years, test_size=0.4, random_state=SEED, stratify=years
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=year_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def metrics_row(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred),
        "F_beta": fbeta_score(y_true, y_pred, beta=BETA),
    }


def oversample_defaults(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    train_df = X_train.copy()
    train_df["def"] = y_train.values
    majority = train_df[train_df["def"] == 0]
    minority = train_df[train_df["def"] == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=SEED)
    balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=SEED)
    return balanced[X_train.columns], balanced["def"]


def tune_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[int, pd.DataFrame]:
    rows = []
    best_depth = None
    best_score = None

    for depth in range(1, 11):
        model = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)
        model.fit(X_train, y_train)
        score = fbeta_score(y_val, model.predict(X_val), beta=BETA)
        rows.append({"max_depth": depth, "validation_fbeta": score})
        if best_score is None or score > best_score:
            best_score = score
            best_depth = depth

    assert best_depth is not None
    return best_depth, pd.DataFrame(rows)


def tune_mlp(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[tuple[int, ...], pd.DataFrame]:
    X_train_bal, y_train_bal = oversample_defaults(X_train, y_train)
    candidates = [(8,), (16,), (8, 8), (16, 16)]
    rows = []
    best_layers = None
    best_score = None

    for layers in candidates:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation="relu",
                        max_iter=400,
                        random_state=SEED,
                        early_stopping=True,
                        validation_fraction=0.15,
                    ),
                ),
            ]
        )
        model.fit(X_train_bal, y_train_bal)
        score = fbeta_score(y_val, model.predict(X_val), beta=BETA)
        rows.append({"hidden_layers": str(layers), "validation_fbeta": score})
        if best_score is None or score > best_score:
            best_score = score
            best_layers = layers

    assert best_layers is not None
    return best_layers, pd.DataFrame(rows)


def tune_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[int, pd.DataFrame]:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    rows = []
    best_estimators = None
    best_score = None

    for n_estimators in [20, 50, 100, 150, 200]:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.8,
            max_depth=2,
            random_state=0,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        score = fbeta_score(y_val, model.predict(X_val), beta=BETA)
        rows.append({"n_estimators": n_estimators, "validation_fbeta": score})
        if best_score is None or score > best_score:
            best_score = score
            best_estimators = n_estimators

    assert best_estimators is not None
    return best_estimators, pd.DataFrame(rows)


def fit_main_models(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series
) -> dict[str, object]:
    rows = []

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)),
        ]
    )
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    rows.append(metrics_row("Logistic Regression", y_test, logistic_pred))

    best_depth, tree_tuning = tune_decision_tree(X_train, y_train, X_val, y_val)
    decision_tree = DecisionTreeClassifier(max_depth=best_depth, class_weight="balanced", random_state=SEED)
    decision_tree.fit(X_train, y_train)
    tree_pred = decision_tree.predict(X_test)
    rows.append(metrics_row("Decision Tree", y_test, tree_pred))

    best_layers, mlp_tuning = tune_mlp(X_train, y_train, X_val, y_val)
    X_train_bal, y_train_bal = oversample_defaults(X_train, y_train)
    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=best_layers,
                    activation="relu",
                    max_iter=400,
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )
    mlp.fit(X_train_bal, y_train_bal)
    mlp_pred = mlp.predict(X_test)
    rows.append(metrics_row("Neural Network", y_test, mlp_pred))

    best_estimators, gb_tuning = tune_gradient_boosting(X_train, y_train, X_val, y_val)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=best_estimators,
        learning_rate=0.8,
        max_depth=2,
        random_state=0,
    )
    gradient_boosting.fit(X_train, y_train, sample_weight=sample_weight)
    gb_pred = gradient_boosting.predict(X_test)
    rows.append(metrics_row("Gradient Boosting", y_test, gb_pred))

    return {
        "table_2": pd.DataFrame(rows).sort_values("F_beta", ascending=False).reset_index(drop=True),
        "tree_tuning": tree_tuning,
        "mlp_tuning": mlp_tuning,
        "gb_tuning": gb_tuning,
        "best_tree_depth": best_depth,
        "best_mlp_layers": best_layers,
        "best_gb_estimators": best_estimators,
        "decision_tree": decision_tree,
    }


def fit_small_tree_models(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series
) -> dict[str, object]:
    rows = []
    best_depth, tuning = tune_decision_tree(X_train, y_train, X_val, y_val)
    small_tree = DecisionTreeClassifier(max_depth=best_depth, class_weight="balanced", random_state=SEED)
    small_tree.fit(X_train, y_train)
    rows.append(metrics_row("Decision Tree Small", y_test, small_tree.predict(X_test)))

    small_logit = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)),
        ]
    )
    small_logit.fit(X_train, y_train)
    rows.append(metrics_row("Logit (2 features)", y_test, small_logit.predict(X_test)))

    return {
        "table_3": pd.DataFrame(rows).sort_values("F_beta", ascending=False).reset_index(drop=True),
        "small_tree_tuning": tuning,
        "best_small_depth": best_depth,
        "small_tree_model": small_tree,
    }


def plot_feature_distributions(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.flatten(), FEATURES):
        ax.hist(df[feature], bins=40, color="#4472C4", alpha=0.8)
        ax.set_title(feature)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    corr = df[["def"] + FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature and Default Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tuning_curve(frame: pd.DataFrame, x_col: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame[x_col].astype(str), frame["validation_fbeta"], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Validation F-beta")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_small_tree(model: DecisionTreeClassifier, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=SMALL_FEATURES,
        class_names=["No default", "Default"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(project_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def run_analysis(project_dir: Path) -> dict[str, object]:
    data_dir = project_dir / "data"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    df = load_data(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_year_split(df, FEATURES)
    main_results = fit_main_models(X_train, X_val, X_test, y_train, y_val, y_test)

    X2_train, X2_val, X2_test, y2_train, y2_val, y2_test = stratified_year_split(df, SMALL_FEATURES)
    small_results = fit_small_tree_models(X2_train, X2_val, X2_test, y2_train, y2_val, y2_test)

    summary_table = df[["def"] + FEATURES].describe().T.reset_index(names="variable")
    class_year_table = (
        df.groupby("year")["def"]
        .agg(observations="size", defaults="sum")
        .reset_index()
    )

    plot_feature_distributions(df, figures_dir / "feature_distributions.png")
    plot_correlation_matrix(df, figures_dir / "correlation_matrix.png")
    plot_tuning_curve(
        main_results["tree_tuning"],
        "max_depth",
        "Decision Tree Validation F-beta",
        figures_dir / "decision_tree_tuning.png",
    )
    plot_tuning_curve(
        main_results["mlp_tuning"],
        "hidden_layers",
        "Neural Network Validation F-beta",
        figures_dir / "mlp_tuning.png",
    )
    plot_tuning_curve(
        main_results["gb_tuning"],
        "n_estimators",
        "Gradient Boosting Validation F-beta",
        figures_dir / "gradient_boosting_tuning.png",
    )
    plot_small_tree(small_results["small_tree_model"], figures_dir / "small_tree_plot.png")

    write_outputs(
        project_dir,
        {
            "table_1_summary_statistics": summary_table,
            "table_2_model_comparison": main_results["table_2"],
            "table_3_small_tree_comparison": small_results["table_3"],
            "class_year_distribution": class_year_table,
            "decision_tree_tuning": main_results["tree_tuning"],
            "mlp_tuning": main_results["mlp_tuning"],
            "gradient_boosting_tuning": main_results["gb_tuning"],
            "small_tree_tuning": small_results["small_tree_tuning"],
        },
    )

    return {
        "df": df,
        "summary_table": summary_table,
        "class_year_table": class_year_table,
        **main_results,
        **small_results,
    }


def print_summary(results: dict[str, object]) -> None:
    print("Table 2: Main model comparison")
    print(results["table_2"].round(4).to_string(index=False))
    print("\nTable 3: Small tree comparison")
    print(results["table_3"].round(4).to_string(index=False))
    print("\nBest hyperparameters")
    print(
        {
            "decision_tree_depth": results["best_tree_depth"],
            "mlp_hidden_layers": results["best_mlp_layers"],
            "gradient_boosting_estimators": results["best_gb_estimators"],
            "small_tree_depth": results["best_small_depth"],
        }
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results = run_analysis(project_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
