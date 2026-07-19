"""app_batch_abdel.py - custom classification project.

Applies the classification workflow techniques from this module
(train/test split, model training, confusion matrix, precision/recall/F1)
to a new problem: predicting whether a manufacturing batch will pass or
fail final spec compliance testing, based on in-process measurements.

Author: Abdelhafidh Mahouel
Date: 2026-07

Process:
    - Load a CSV dataset.
    - Inspect and check data quality.
    - Create a clean modeling view.
    - Train a supervised classification model (Decision Tree).
    - Evaluate model performance (accuracy, confusion matrix, classification report).
    - Predict one new case.
    - Save a predictions report.
    - Create useful charts.

Data Source:
- data/raw/batch_quality_abdel.csv

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_batch_abdel
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "batch_quality_abdel"

# STEP 1. Pick the target variable we want to predict.
# batch_pass: 1 = batch passed final spec compliance, 0 = batch failed.

TARGET_COL: Final[str] = "batch_pass"

# STEP 2. Define the column names (features) that may help predict the target.
# batch_id is an identifier, not a predictive feature, so it is excluded.

FEATURE_COLS: Final[list[str]] = [
    "mixing_time_min",
    "process_temp_c",
    "ph_level",
    "viscosity_cp",
    "raw_material_age_days",
    "operator_experience_yrs",
    "humidity_pct",
    "line_speed_units_min",
]

# STEP 3. Define the test size and random state for reproducibility.

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42

# STEP 4. Define the decision tree depth (controls overfitting risk).

MAX_DEPTH: Final[int] = 4

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the batch quality dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info("DataFrame info")
    df.info()

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    LOG.info("Target class balance")
    LOG.debug(f"\n{df[TARGET_COL].value_counts(normalize=True)}")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling."""
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]
    df_no_missing: pd.DataFrame = df_selected.dropna()
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean


# === Section 6. Train Supervised Classification Model ===


def train_model(
    df_clean: pd.DataFrame,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series, pd.Series[int]]:
    """Train a DecisionTreeClassifier and evaluate on a held-out test set."""
    LOG.info("Training DecisionTreeClassifier model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy: float = accuracy_score(y_test, y_pred)
    LOG.info(f"Test accuracy: {accuracy:.3f}")

    LOG.info("Confusion matrix (rows=actual, cols=predicted)")
    cm = confusion_matrix(y_test, y_pred)
    LOG.debug(f"\n{cm}")

    LOG.info("Classification report (precision, recall, F1)")
    report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])
    LOG.debug(f"\n{report}")

    return model, x_test, y_test, y_pred


# === Section 7. Predict One New Case ===


def predict_example(model: DecisionTreeClassifier) -> None:
    """Use the trained model to predict one new batch outcome."""
    LOG.info("Predicting one new case")

    new_case = pd.DataFrame(
        [
            {
                "mixing_time_min": 44.5,
                "process_temp_c": 69.8,
                "ph_level": 7.02,
                "viscosity_cp": 1180,
                "raw_material_age_days": 15,
                "operator_experience_yrs": 8,
                "humidity_pct": 46.0,
                "line_speed_units_min": 79.5,
            }
        ]
    )

    prediction: int = model.predict(new_case)[0]
    proba = model.predict_proba(new_case)[0]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted outcome: {'Pass' if prediction == 1 else 'Fail'}")
    LOG.info(f"Predicted probabilities: Fail={proba[0]:.2f}, Pass={proba[1]:.2f}")


# === Section 7b. Save Predictions Report ===


def save_predictions_report(
    x_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series[int]
) -> None:
    """Save actual vs. predicted results to a CSV report."""
    LOG.info("Saving predictions report.......")

    report = x_test.copy()
    report["actual_batch_pass"] = y_test.to_numpy()
    report["predicted_batch_pass"] = y_pred
    report["correct_prediction"] = (
        report["actual_batch_pass"] == report["predicted_batch_pass"]
    )

    output_path = "data/processed/batch_predictions_abdel.csv"
    report.to_csv(output_path, index=False)

    LOG.info(f"Saved predictions report to: {output_path}")
    LOG.debug(f"\n{report}")


# === Section 8. Create Visualizations ===


def make_plots(
    model: DecisionTreeClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series[int],
) -> None:
    """Create charts for the classification case."""
    LOG.info("Creating chart: confusion matrix")

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Confusion Matrix (CLOSE chart to continue)")

    LOG.info("Creating chart: feature importance")

    fig, ax = plt.subplots(figsize=(9, 5))

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    bar_plt: Axes = sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        ax=ax,
    )

    bar_plt.set_title("Feature Importance (CLOSE chart to continue)")
    bar_plt.set_xlabel("Importance")
    bar_plt.set_ylabel("Feature")


# === Section 9. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL}")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the custom classification workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model, x_test, y_test, y_pred = train_model(df_clean)

    LOG.info("Predict one case..........")
    predict_example(model)

    LOG.info("Save predictions report...")
    save_predictions_report(x_test, y_test, y_pred)

    LOG.info("Create charts.............")
    make_plots(model, x_test, y_test, y_pred)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, CLOSE the chart windows with the close button to CONTINUE -----"
    )

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info("Close chart windows and terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
