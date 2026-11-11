
"""
SMART REAL ESTATE AI
Optimized Decision Tree and Random Forest Training

Required packages:
python -m pip install pandas numpy matplotlib scikit-learn joblib
"""

from pathlib import Path
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from joblib import dump


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "california_housing.csv"
TARGET = "MedHouseVal"
RANDOM_STATE = 42

# Keep a safety margin below GitHub's 25 MiB limit.
MAX_MODEL_SIZE = 24 * 1024 * 1024

# Candidate Random Forest configurations.
CANDIDATES = [
    {"n_estimators": 20, "max_depth": 12, "min_samples_leaf": 2},
    {"n_estimators": 30, "max_depth": 14, "min_samples_leaf": 2},
    {"n_estimators": 40, "max_depth": 16, "min_samples_leaf": 2},
    {"n_estimators": 50, "max_depth": 18, "min_samples_leaf": 2}
]

# Maximum acceptable R2 reduction compared with the original.
MAX_R2_DROP = 0.02


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def main():

    print("=" * 65)
    print("SMART REAL ESTATE AI - OPTIMIZED MODEL TRAINING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found inside the dataset folder."
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET not in data.columns:
        raise ValueError("MedHouseVal target column is missing.")

    if data[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X = X.apply(pd.to_numeric, errors="raise")
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print("\nDataset:", data.shape)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    for folder in ["models", "results", "plots"]:
        (BASE_DIR / folder).mkdir(exist_ok=True)

    # -------------------------------------------------
    # 1. Train original Random Forest as reference
    # -------------------------------------------------

    print("\nTraining original 200-tree Random Forest...")

    original_model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1
    )

    start = time.perf_counter()
    original_model.fit(X_train, y_train)
    original_training_time = time.perf_counter() - start

    original_predictions = original_model.predict(X_test)
    original_metrics = calculate_metrics(
        y_test, original_predictions
    )

    print("Original R2:", round(original_metrics["R2"], 5))
    print("Original RMSE:", round(original_metrics["RMSE"], 5))

    # -------------------------------------------------
    # 2. Train Decision Tree
    # -------------------------------------------------

    print("\nTraining Decision Tree...")

    decision_model = DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        min_samples_leaf=2
    )

    start = time.perf_counter()
    decision_model.fit(X_train, y_train)
    decision_training_time = time.perf_counter() - start

    decision_predictions = decision_model.predict(X_test)
    decision_metrics = calculate_metrics(
        y_test, decision_predictions
    )

    dump(
        decision_model,
        BASE_DIR / "models" / "decision_tree_model.joblib",
        compress=("xz", 9)
    )

    # -------------------------------------------------
    # 3. Test optimized Random Forest candidates
    # -------------------------------------------------

    print("\nTesting optimized Random Forest models...")

    candidate_results = []

    temp_file = BASE_DIR / "models" / "rf_candidate_temp.joblib"

    for settings in CANDIDATES:

        print("\nTesting:", settings)

        model = RandomForestRegressor(
            **settings,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        # Measure actual compressed model size.
        dump(model, temp_file, compress=("xz", 9))

        size_bytes = os.path.getsize(temp_file)
        size_mb = size_bytes / (1024 * 1024)

        print("R2:", round(metrics["R2"], 5))
        print("RMSE:", round(metrics["RMSE"], 5))
        print("Compressed size:", round(size_mb, 2), "MB")

        candidate_results.append({
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
            "settings": settings,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "training_time": training_time
        })

        if temp_file.exists():
            temp_file.unlink()

    # -------------------------------------------------
    # 4. Select best candidate
    # -------------------------------------------------

    eligible = [
        item for item in candidate_results
        if item["size_bytes"] < MAX_MODEL_SIZE
        and item["metrics"]["R2"] >=
        original_metrics["R2"] - MAX_R2_DROP
    ]

    if not eligible:
        print("\nNo candidate meets both requirements.")

        eligible = [
            item for item in candidate_results
            if item["size_bytes"] < MAX_MODEL_SIZE
        ]

        if not eligible:
            raise RuntimeError(
                "No tested model is below 24 MiB. "
                "More aggressive optimization is required."
            )

        print("Selecting the best model below 24 MiB,")
        print("but its accuracy may be lower than the original.")

    # Lowest RMSE among eligible candidates.
    best = min(
        eligible,
        key=lambda item: item["metrics"]["RMSE"]
    )

    optimized_model = best["model"]
    optimized_predictions = best["predictions"]

    final_model_path = (
        BASE_DIR / "models" / "random_forest_model.joblib"
    )

    dump(
        optimized_model,
        final_model_path,
        compress=("xz", 9)
    )

    final_size = os.path.getsize(final_model_path)
    final_size_mb = final_size / (1024 * 1024)

    # -------------------------------------------------
    # 5. Compare model performance
    # -------------------------------------------------

    metric_rows = []

    comparisons = [
        ("Decision Tree", decision_metrics, decision_training_time),
        ("Original Random Forest", original_metrics,
         original_training_time),
        ("Optimized Random Forest", best["metrics"],
         best["training_time"])
    ]

    for name, metrics, training_time in comparisons:

        metric_rows.append({
            "Model": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "Training_Time_Seconds": training_time
        })

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df.to_csv(
        BASE_DIR / "results" / "model_metrics.csv",
        index=False
    )

    predictions_df = pd.DataFrame({
        "Actual_MedHouseVal": y_test.reset_index(drop=True),
        "Decision_Tree_Prediction": decision_predictions,
        "Original_RF_Prediction": original_predictions,
        "Optimized_RF_Prediction": optimized_predictions
    })

    predictions_df.to_csv(
        BASE_DIR / "results" / "predictions.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 65)

    print(metrics_df.to_string(index=False))

    print("\nSelected configuration:", best["settings"])
    print("Final compressed size:", round(final_size_mb, 2), "MB")

    print(
        "R2 reduction:",
        round(original_metrics["R2"] - best["metrics"]["R2"], 5)
    )

    # -------------------------------------------------
    # 6. Comparison graph
    # -------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, metric, title in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["MAE", "RMSE", "R2"]
    ):

        ax.bar(metrics_df["Model"], metrics_df[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        BASE_DIR / "plots" / "model_comparison.png",
        dpi=160
    )

    plt.close(fig)

    # -------------------------------------------------
    # 7. Actual vs predicted graph
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        y_test,
        optimized_predictions,
        alpha=0.35,
        s=12
    )

    low = min(
        float(y_test.min()),
        float(optimized_predictions.min())
    )

    high = max(
        float(y_test.max()),
        float(optimized_predictions.max())
    )

    ax.plot([low, high], [low, high], linestyle="--")

    ax.set_xlabel("Actual MedHouseVal")
    ax.set_ylabel("Predicted MedHouseVal")
    ax.set_title("Actual vs Predicted - Optimized Random Forest")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        BASE_DIR / "plots" / "actual_vs_predicted.png",
        dpi=160
    )

    plt.close(fig)

    # -------------------------------------------------
    # 8. Residual graph
    # -------------------------------------------------

    residuals = y_test.to_numpy() - optimized_predictions

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        optimized_predictions,
        residuals,
        alpha=0.35,
        s=12
    )

    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted MedHouseVal")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot - Optimized Random Forest")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        BASE_DIR / "plots" / "residual_plot.png",
        dpi=160
    )

    plt.close(fig)

    # -------------------------------------------------
    # 9. Feature importance
    # -------------------------------------------------

    importance = pd.Series(
        optimized_model.feature_importances_,
        index=X.columns
    ).sort_values()

    importance.to_csv(
        BASE_DIR / "results" / "feature_importance.csv",
        header=["Importance"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    importance.plot(kind="barh", ax=ax)

    ax.set_xlabel("Feature Importance")
    ax.set_title("Optimized Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        BASE_DIR / "plots" / "feature_importance.png",
        dpi=160
    )

    plt.close(fig)

    # -------------------------------------------------
    # 10. Training summary
    # -------------------------------------------------

    with open(
        BASE_DIR / "results" / "training_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("SMART REAL ESTATE AI\n")
        file.write("CALIFORNIA HOUSING MODEL TRAINING\n\n")

        file.write("Dataset rows: " + str(len(data)) + "\n")
        file.write("Features: " + ", ".join(X.columns) + "\n")
        file.write("Target: " + TARGET + "\n")
        file.write("Train/test split: 80/20\n")
        file.write("Random state: 42\n\n")

        file.write("MODEL COMPARISON\n")
        file.write(metrics_df.to_string(index=False))

        file.write("\n\nOPTIMIZED RANDOM FOREST SETTINGS\n")
        file.write(str(best["settings"]) + "\n")

        file.write(
            "\nCompressed model size: "
            + str(round(final_size_mb, 2)) + " MB\n"
        )

        file.write(
            "\nMedHouseVal represents census block-group "
            "median house value in units of $100,000, "
            "not an individual-property appraisal.\n"
        )

    print("\n" + "=" * 65)

    if final_size < MAX_MODEL_SIZE:
        print("SUCCESS: Model is below the 24 MiB safety target.")
    else:
        print("WARNING: Model exceeds the size target.")

    print("Optimized model saved at:")
    print(final_model_path)

    print("\nAll training outputs have been saved.")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
