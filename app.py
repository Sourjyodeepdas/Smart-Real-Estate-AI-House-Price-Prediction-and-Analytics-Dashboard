
"""
Smart Real Estate AI — Streamlit dashboard for California Housing.

Run from this directory:
python -m streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "dataset" / "california_housing.csv"
METRICS_FILE = ROOT / "results" / "model_metrics.csv"
PREDICTIONS_FILE = ROOT / "results" / "predictions.csv"
IMPORTANCE_FILE = ROOT / "results" / "feature_importance.csv"

MODELS = {
    "Random Forest": ROOT / "models" / "random_forest_model.joblib",
    "Decision Tree": ROOT / "models" / "decision_tree_model.joblib",
}

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

DESCRIPTIONS = {
    "MedInc": "Median income (units of $10,000)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block-group population",
    "AveOccup": "Average household occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

# Correct mapping between model selection and CSV columns
PREDICTION_COLUMNS = {
    "Random Forest": "Optimized_RF_Prediction",
    "Decision Tree": "Decision_Tree_Prediction",
}


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Real Estate AI",
    page_icon="🏘️",
    layout="wide"
)


# ============================================================
# 3. DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_data():
    return (
        pd.read_csv(DATA_FILE),
        pd.read_csv(METRICS_FILE),
        pd.read_csv(PREDICTIONS_FILE),
        pd.read_csv(IMPORTANCE_FILE, index_col=0)
    )


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_name):
    return joblib.load(MODELS[model_name])


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = [
    DATA_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    IMPORTANCE_FILE,
]

for path in required_files:
    if not path.exists():
        st.error(
            f"Missing required file: "
            f"{path.relative_to(ROOT)}. "
            f"Check your GitHub repository."
        )
        st.stop()

for model_name, model_path in MODELS.items():
    if not model_path.exists():
        st.error(f"Missing model file: {model_path.relative_to(ROOT)}")
        st.stop()


# ============================================================
# 5. LOAD DATA
# ============================================================

data, metrics, predictions, importance = load_data()


# ============================================================
# 6. VALIDATE REQUIRED COLUMNS
# ============================================================

required_data_columns = FEATURES + ["MedHouseVal"]

required_prediction_columns = [
    "Actual_MedHouseVal",
    *PREDICTION_COLUMNS.values(),
]

required_metric_columns = [
    "Model",
    "MAE",
    "RMSE",
    "R2",
]

missing_data_columns = [
    col for col in required_data_columns
    if col not in data.columns
]

missing_prediction_columns = [
    col for col in required_prediction_columns
    if col not in predictions.columns
]

missing_metric_columns = [
    col for col in required_metric_columns
    if col not in metrics.columns
]

if missing_data_columns:
    st.error(f"Missing dataset columns: {missing_data_columns}")
    st.stop()

if missing_prediction_columns:
    st.error(
        f"Missing prediction CSV columns: "
        f"{missing_prediction_columns}"
    )
    st.stop()

if missing_metric_columns:
    st.error(f"Missing metrics columns: {missing_metric_columns}")
    st.stop()


# ============================================================
# 7. DASHBOARD HEADER
# ============================================================

st.title("🏘️ Smart Real Estate AI")

st.caption(
    "California Housing | Prediction • Model comparison • "
    "Analytics dashboard"
)

st.info(
    "**Interpretation:** This dataset describes California census "
    "block groups, not individual houses. Predictions estimate a "
    "block group's median house value and should not be used as "
    "individual-property appraisals."
)


# ============================================================
# 8. SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("Dashboard settings")

    model_name = st.selectbox(
        "Prediction model",
        list(MODELS.keys())
    )

    st.caption(
        "Both models were trained on 80% of the dataset and "
        "evaluated on the same 20% held-out test set "
        "(random_state=42)."
    )

    st.markdown("**Target units:** MedHouseVal × $100,000")

    st.markdown("**Dataset:** 20,640 census block groups")

    st.markdown("**Input data:** 8 numerical features")


# ============================================================
# 9. DASHBOARD TABS
# ============================================================

prediction_tab, comparison_tab, diagnostics_tab, geography_tab, data_tab = st.tabs(
    [
        "🏠 Predict",
        "📊 Model comparison",
        "📈 Prediction diagnostics",
        "🗺️ Geographic analysis",
        "📁 Dataset",
    ]
)


# ============================================================
# 10. HOUSE PRICE PREDICTION
# ============================================================

with prediction_tab:

    st.subheader("Predict median house value")

    st.write(
        "Enter the characteristics of a California census "
        "block group. The starting values are the dataset medians."
    )

    with st.form("predict_form"):

        left, right = st.columns(2)

        entered = {}

        for idx, feature in enumerate(FEATURES):

            with (left if idx < 4 else right):

                val = float(data[feature].median())

                entered[feature] = st.number_input(
                    DESCRIPTIONS[feature],
                    value=round(val, 4),
                    step=(
                        0.1
                        if feature not in ("Latitude", "Longitude")
                        else 0.01
                    ),
                    format="%.4f",
                    key="field_" + feature
                )

        submitted = st.form_submit_button(
            "Predict median house value",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if entered["AveBedrms"] > entered["AveRooms"]:

            st.warning(
                "Average bedrooms exceeds average rooms. "
                "Check the entered values."
            )

        if (
            entered["Population"] < 0
            or entered["AveOccup"] <= 0
            or entered["AveRooms"] <= 0
        ):

            st.error(
                "Population must be nonnegative, and average "
                "occupancy and rooms must be positive."
            )

        elif not (
            32 <= entered["Latitude"] <= 43
            and -125 <= entered["Longitude"] <= -113
        ):

            st.error(
                "Enter latitude and longitude within "
                "the approximate California range."
            )

        else:

            model = load_model(model_name)

            row = pd.DataFrame(
                [entered],
                columns=FEATURES
            )

            predicted_units = float(
                model.predict(row)[0]
            )

            st.metric(
                "Estimated block-group median house value",
                f"${predicted_units * 100_000:,.0f}"
            )

            st.caption(
                f"Model: {model_name} | "
                f"Raw prediction: {predicted_units:.4f} × $100,000"
            )

            if predicted_units > 5.0 or predicted_units < 0:

                st.warning(
                    "This estimate falls outside the usual "
                    "capped target range; treat it with caution."
                )

            st.caption(
                "These are model estimates, not guarantees. "
                "The original target is capped near $500,000, "
                "which limits accuracy for expensive areas."
            )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

with comparison_tab:

    st.subheader("Decision Tree vs Random Forest")

    metric_labels = {
        "MAE": "Mean absolute error",
        "RMSE": "Root mean squared error",
        "R2": "R² score",
    }

    columns = st.columns(3)

    for col, metric in zip(
        columns,
        ["MAE", "RMSE", "R2"]
    ):

        with col:

            fig = px.bar(
                metrics,
                x="Model",
                y=metric,
                color="Model",
                text_auto=".3f",
                title=metric_labels[metric]
            )

            fig.update_layout(
                showlegend=False,
                height=340,
                margin=dict(
                    l=15,
                    r=15,
                    t=55,
                    b=15
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.dataframe(
        metrics[["Model", "MAE", "RMSE", "R2"]].round(6),
        use_container_width=True,
        hide_index=True
    )

    best = metrics.loc[
        metrics["RMSE"].idxmin(),
        "Model"
    ]

    st.success(
        f"Best held-out test RMSE: {best}"
    )

    st.caption(
        "MAE and RMSE are in units of $100,000; lower is better. "
        "Higher R² is better. These scores are from one test split, "
        "not cross-validation or a tuned model search."
    )

    st.download_button(
        "Download metrics CSV",
        METRICS_FILE.read_bytes(),
        "model_metrics.csv",
        "text/csv"
    )


# ============================================================
# 12. PREDICTION DIAGNOSTICS
# ============================================================

with diagnostics_tab:

    st.subheader("Actual vs predicted values")

    chosen = st.radio(
        "View model",
        list(MODELS.keys()),
        horizontal=True,
        key="diagnostics_model"
    )

    # FIXED: Correct prediction column mapping

    pred_col = PREDICTION_COLUMNS[chosen]

    actual = predictions["Actual_MedHouseVal"]

    predicted = predictions[pred_col]

    diagnostics = pd.DataFrame({
        "Actual": actual,
        "Predicted": predicted,
        "Residual": actual - predicted
    })

    # Actual vs predicted scatter plot

    fig = px.scatter(
        diagnostics,
        x="Actual",
        y="Predicted",
        opacity=0.35,
        title=f"Actual vs predicted — {chosen}",
        labels={
            "Actual": "Actual (× $100,000)",
            "Predicted": "Predicted (× $100,000)"
        }
    )

    low = float(
        min(actual.min(), predicted.min())
    )

    high = float(
        max(actual.max(), predicted.max())
    )

    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect prediction",
            line=dict(dash="dash")
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RESIDUAL ANALYSIS
    # --------------------------------------------------------

    st.subheader("Residual / error analysis")

    fig = px.scatter(
        diagnostics,
        x="Predicted",
        y="Residual",
        opacity=0.35,
        title=f"Residuals — {chosen}",
        labels={
            "Predicted": "Predicted (× $100,000)",
            "Residual": "Actual − predicted (× $100,000)"
        }
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("Random Forest feature importance")

    fi = importance.reset_index()

    fi.columns = [
        "Feature",
        "Importance"
    ]

    fig = px.bar(
        fi.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Impurity-based feature importance (Random Forest)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Impurity-based importance is not causal. "
        "Correlated features may share or distort importance; "
        "permutation importance is an alternative."
    )

    st.download_button(
        "Download test predictions CSV",
        PREDICTIONS_FILE.read_bytes(),
        "predictions.csv",
        "text/csv"
    )


# ============================================================
# 13. GEOGRAPHIC ANALYSIS
# ============================================================

with geography_tab:

    st.subheader(
        "Geographic distribution of median house values"
    )

    st.caption(
        "Each point represents a census block group, not a single "
        "property. Map uses a reproducible sample for responsiveness."
    )

    sample_size = st.slider(
        "Number of locations",
        500,
        min(5000, len(data)),
        min(2000, len(data)),
        step=500
    )

    geo = data.sample(
        n=sample_size,
        random_state=42
    ).copy()

    geo["Median value ($)"] = (
        geo["MedHouseVal"] * 100_000
    )

    fig = px.scatter_map(
        geo,
        lat="Latitude",
        lon="Longitude",
        color="Median value ($)",
        hover_data={
            "MedInc": ":.2f",
            "HouseAge": ":.1f",
            "Median value ($)": ":,.0f"
        },
        zoom=5,
        height=580,
        map_style="open-street-map",
        title="California Housing — sample of block groups"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 14. DATASET OVERVIEW
# ============================================================

with data_tab:

    st.subheader("Dataset overview")

    a, b, c = st.columns(3)

    a.metric(
        "Records",
        f"{len(data):,}"
    )

    b.metric(
        "Input features",
        str(len(FEATURES))
    )

    c.metric(
        "Missing cells",
        str(int(data.isna().sum().sum()))
    )

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Descriptive statistics")

    st.dataframe(
        data.describe().T.round(3),
        use_container_width=True
    )

    st.caption(
        "MedInc is in units of $10,000. "
        "MedHouseVal is in units of $100,000. "
        "Some target values are capped near 5.0."
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "University AI/ML project • California Housing • "
    "Decision Tree & Random Forest • Reproducible seed: 42"
)
