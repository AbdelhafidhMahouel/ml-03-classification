"""app_abdel.py - modified version of app_case.py.

A modified example of a supervised regression case.
Based on the original app_case.py example provided in this project.

Author: Abdelhafidh Mahouel
Date: 2026-07

Modification:
    - Added an engineered feature, engagement_score, combining hours
      studied and attendance percentage into one composite measure.
    - Added a new chart: actual vs. predicted score (residual plot).
    - Added a new saved output: a CSV report of test set predictions
      and residuals, saved to data/processed/model_predictions.csv.

Process:
    - Load a CSV dataset.
    - Add a derived feature.
    - Train a supervised regression model.
    - Evaluate model performance.
    - Predict one new case.
    - Save a predictions report.
    - Create useful charts.

Data Source:
- data/raw/hours_scores_case.csv

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_abdel
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "hours_scores_case"

# STEP 1. Pick the target variable we want to predict.

TARGET_COL: Final[str] = "score"

# STEP 2. Define the raw column names (features) available in the source data.

RAW_FEATURE_COLS: Final[list[str]] = [
    "hours_studied",
    "practice_quizzes",
    "attendance_pct",
    "sleep_hours",
    "prior_score",
]

# MODIFICATION: engagement_score is a derived feature combining hours studied
# and attendance percentage into a single measure of consistent study effort.
FEATURE_COLS: Final[list[str]] = RAW_FEATURE_COLS + ["engagement_score"]

# STEP 3. Define the test size and random state for reproducibility.

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the case dataset from the data/raw folder."""
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

    selected_cols: list[str] = RAW_FEATURE_COLS + [TARGET_COL]

    # Select only the columns we need.
    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]

    # Drop rows with any missing values.
    df_no_missing: pd.DataFrame = df_selected.dropna()

    # Assign a copy of the no-missing DataFrame to df_clean to avoid SettingWithCopyWarning.
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean


# === Section 5b. Add Derived Feature (MODIFICATION) ===


def add_derived_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Add an engineered feature: engagement_score.

    engagement_score = hours_studied * (attendance_pct / 100)

    Rationale: hours studied alone does not capture whether that effort
    was reinforced by consistent classroom attendance. Combining the two
    signals into one composite feature may carry more predictive value
    than either raw feature alone.
    """
    LOG.info("Adding derived feature: engagement_score")

    df_clean = df_clean.copy()
    df_clean["engagement_score"] = df_clean["hours_studied"] * (
        df_clean["attendance_pct"] / 100
    )

    LOG.debug(
        f"\n{df_clean[['hours_studied', 'attendance_pct', 'engagement_score']].head()}"
    )

    return df_clean


# === Section 6. Train Supervised Model ===


def train_model(
    df_clean: pd.DataFrame,
) -> tuple[LinearRegression, pd.DataFrame, pd.Series, pd.Series[float]]:
    """Train a supervised regression model.

    MODIFICATION: now returns the test set and predictions too,
    so we can build a residuals report and chart (see make_plots
    and save_predictions_report).
    """
    LOG.info("Training LinearRegression model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae: float = mean_absolute_error(y_test, y_pred)
    r2: float = r2_score(y_test, y_pred)

    LOG.info(f"Mean absolute error: {mae:.2f}")
    LOG.info(f"R-squared: {r2:.2f}")

    return model, x_test, y_test, y_pred


# === Section 7. Predict One New Case ===


def predict_example(model: LinearRegression) -> None:
    """Use the trained model to predict one new student score."""
    LOG.info("Predicting one new case")

    new_case = pd.DataFrame(
        [
            {
                "hours_studied": 6.5,
                "practice_quizzes": 4,
                "attendance_pct": 92,
                "sleep_hours": 7.0,
                "prior_score": 72,
                "engagement_score": 6.5 * (92 / 100),
            }
        ]
    )

    predicted_score: float = model.predict(new_case)[0]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted score: {predicted_score:.1f}")


# === Section 7b. Save Predictions Report (MODIFICATION: new output) ===


def save_predictions_report(
    x_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series[float]
) -> None:
    """Save actual vs. predicted results and residuals to a CSV report."""
    LOG.info("Saving predictions report.......")

    report = x_test.copy()
    report["actual_score"] = y_test.to_numpy()
    report["predicted_score"] = y_pred
    report["residual"] = report["actual_score"] - report["predicted_score"]

    output_path = "data/processed/model_predictions.csv"
    report.to_csv(output_path, index=False)

    LOG.info(f"Saved predictions report to: {output_path}")
    LOG.debug(f"\n{report}")


# === Section 8. Create Visualizations ===


def make_plots(
    df_clean: pd.DataFrame,
    model: LinearRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series[float],
) -> None:
    """Create charts for the supervised regression case."""
    LOG.info("Creating chart: hours studied vs score")

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter_plt: Axes = sns.scatterplot(
        data=df_clean,
        x="hours_studied",
        y=TARGET_COL,
        ax=ax,
    )

    scatter_plt.set_title("Hours Studied vs Score (CLOSE chart to continue)")
    scatter_plt.set_xlabel("Hours Studied")
    scatter_plt.set_ylabel("Score")

    LOG.info("Creating chart: model coefficients")

    fig, ax = plt.subplots(figsize=(9, 5))

    LOG.info(f"Got a figure {fig} and axes {ax} from plt.subplots().")

    coefficient_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_,
        }
    ).sort_values("coefficient", ascending=False)

    bar_plt: Axes = sns.barplot(
        data=coefficient_df,
        x="coefficient",
        y="feature",
        ax=ax,
    )

    bar_plt.set_title("Model Coefficients (CLOSE chart to continue)")
    bar_plt.set_xlabel("Coefficient")
    bar_plt.set_ylabel("Feature")

    LOG.info("Creating chart: actual vs predicted (residuals)")

    fig, ax = plt.subplots(figsize=(9, 5))

    residual_plt: Axes = sns.scatterplot(
        x=y_test,
        y=y_pred,
        ax=ax,
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")

    residual_plt.set_title("Actual vs Predicted Score (CLOSE chart to continue)")
    residual_plt.set_xlabel("Actual Score")
    residual_plt.set_ylabel("Predicted Score")


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
    """Main function to run the modified supervised ML workflow."""
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

    LOG.info("Add derived feature.......")
    df_clean = add_derived_features(df_clean)

    LOG.info("Train supervised model....")
    model, x_test, y_test, y_pred = train_model(df_clean)

    LOG.info("Predict one case..........")
    predict_example(model)

    LOG.info("Save predictions report...")
    save_predictions_report(x_test, y_test, y_pred)

    LOG.info("Create charts.............")
    make_plots(df_clean, model, x_test, y_test, y_pred)

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
