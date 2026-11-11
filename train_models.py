"""
SMART REAL ESTATE AI
Decision Tree and Random Forest comparison using california_housing.csv

Run in Python IDLE: open this file and press F5.
Required packages:
python -m pip install pandas numpy matplotlib scikit-learn joblib
"""
from pathlib import Path
import time
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

def main():
    print("=" * 65)
    print("SMART REAL ESTATE AI - MODEL TRAINING")
    print("=" * 65)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Place california_housing.csv inside the dataset folder."
        )

    data = pd.read_csv(DATA_PATH)
    print("\nDataset loaded successfully!")
    print("Dataset shape:", data.shape)
    print("Target column:", TARGET)

    if TARGET not in data.columns:
        raise ValueError("Target column 'MedHouseVal' is missing from the CSV.")
    if data[TARGET].isna().any():
        raise ValueError("Target column contains missing values. Clean these rows first.")

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    # This supplied dataset contains numerical columns. Median imputation makes
    # the program robust if missing feature values are introduced later.
    X = X.apply(pd.to_numeric, errors="raise")
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = {
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=1
        ),
    }

    output_dirs = ["models", "results", "plots"]
    for name in output_dirs:
        (BASE_DIR / name).mkdir(exist_ok=True)

    metric_rows = []
    prediction_data = pd.DataFrame({
        "Actual_MedHouseVal": y_test.reset_index(drop=True)
    })
    fitted_models = {}

    for name, model in models.items():
        print("\nTraining", name, "...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        predicted = model.predict(X_test)
        prediction_time = time.perf_counter() - start
        fitted_models[name] = model

        mae = mean_absolute_error(y_test, predicted)
        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        r2 = r2_score(y_test, predicted)

        metric_rows.append({
            "Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "Training_Time_Seconds": training_time,
            "Prediction_Time_Seconds": prediction_time
        })
        prediction_data[name.replace(" ", "_") + "_Prediction"] = predicted
        model_file = BASE_DIR / "models" / (name.lower().replace(" ", "_") + "_model.joblib")
        dump(model, model_file)
        print("Training completed.")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(BASE_DIR / "results" / "model_metrics.csv", index=False)
    prediction_data.to_csv(BASE_DIR / "results" / "predictions.csv", index=False)

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(metrics.to_string(index=False))
    best_name = metrics.loc[metrics["RMSE"].idxmin(), "Model"]
    print("\nBest model by lowest test RMSE:", best_name)

    # Comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(
        axes, ["MAE", "RMSE", "R2"],
        ["MAE (lower is better)", "RMSE (lower is better)", "R2 (higher is better)"]
    ):
        ax.bar(metrics["Model"], metrics[metric])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "plots" / "model_comparison.png", dpi=160)
    plt.close(fig)

    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    # Actual versus predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, best_predictions, alpha=0.35, s=12)
    low = min(float(y_test.min()), float(best_predictions.min()))
    high = max(float(y_test.max()), float(best_predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual MedHouseVal")
    ax.set_ylabel("Predicted MedHouseVal")
    ax.set_title("Actual vs Predicted - " + best_name)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "plots" / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Residual plot
    residuals = y_test.to_numpy() - best_predictions
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(best_predictions, residuals, alpha=0.35, s=12)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted MedHouseVal")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot - " + best_name)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "plots" / "residual_plot.png", dpi=160)
    plt.close(fig)

    # Random Forest feature importance
    importance = pd.Series(
        fitted_models["Random Forest"].feature_importances_,
        index=X.columns
    ).sort_values()
    importance.to_csv(
        BASE_DIR / "results" / "feature_importance.csv",
        header=["Importance"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "plots" / "feature_importance.png", dpi=160)
    plt.close(fig)

    with open(BASE_DIR / "results" / "training_summary.txt", "w", encoding="utf-8") as file:
        file.write("SMART REAL ESTATE AI - CALIFORNIA HOUSING\n")
        file.write("Dataset rows: " + str(len(data)) + "\n")
        file.write("Feature columns: " + ", ".join(X.columns) + "\n")
        file.write("Target: " + TARGET + "\n")
        file.write("Train/test split: 80/20; random_state=42\n\n")
        file.write(metrics.to_string(index=False))
        file.write("\n\nBest model by lowest test RMSE: " + best_name + "\n")
        file.write(
            "\nMedHouseVal represents census block-group median house value "
            "in units of $100,000, not an individual-property appraisal.\n"
        )

    print("\nAll outputs saved in the models, results, and plots folders.")
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
