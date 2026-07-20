from pathlib import Path

import joblib
import pandas as pd
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# Project paths
project_root = Path(__file__).resolve().parents[1]

train_path = (
    project_root
    / "data"
    / "processed"
    / "model_train_2018_2023.parquet"
)

validation_path = (
    project_root
    / "data"
    / "processed"
    / "model_validation_2024.parquet"
)

model_output_path = (
    project_root
    / "models"
    / "situation_only_logistic_regression.joblib"
)

predictions_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "validation_predictions_2024.parquet"
)

metrics_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "baseline_validation_metrics.csv"
)


# Create output directories if necessary
model_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

predictions_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# Load chronological model data
train_data = pl.read_parquet(
    train_path
)

validation_data = pl.read_parquet(
    validation_path
)


# Categorical variables are one-hot encoded
categorical_features = [
    "season_type",
    "qtr",
    "down",
    "goal_to_go",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_is_home",
    "posteam_opening_kickoff",
    "roof",
    "surface",
]


# Numeric variables are standardized
numeric_features = [
    "week",
    "game_seconds_remaining",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "posteam_spread_line",
    "total_line",
]


model_features = (
    categorical_features
    + numeric_features
)

target_column = "is_pass"


# Confirm all required columns are present
required_columns = (
    model_features
    + [
        target_column,
        "xpass",
    ]
)

missing_train_columns = [
    column
    for column in required_columns
    if column not in train_data.columns
]

missing_validation_columns = [
    column
    for column in required_columns
    if column not in validation_data.columns
]

if missing_train_columns:
    raise ValueError(
        "Training data is missing columns: "
        f"{missing_train_columns}"
    )

if missing_validation_columns:
    raise ValueError(
        "Validation data is missing columns: "
        f"{missing_validation_columns}"
    )


# Convert selected Polars columns to pandas for scikit-learn
X_train = train_data.select(
    model_features
).to_pandas()

y_train = train_data[
    target_column
].to_numpy()

X_validation = validation_data.select(
    model_features
).to_pandas()

y_validation = validation_data[
    target_column
].to_numpy()


# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "one_hot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop=None,
            ),
        ),
    ]
)


# Apply the appropriate preprocessing to each feature type
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
    ]
)


# Logistic-regression pipeline
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)


# Train only on 2018 through 2023
print("TRAINING BASELINE MODEL")
print(
    f"Training plays: {len(X_train):,}"
)

model.fit(
    X_train,
    y_train,
)

print("Training complete")


# Generate 2024 validation probabilities
validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

validation_predictions = (
    validation_probabilities >= 0.5
).astype(int)


# nflverse benchmark probabilities
xpass_probabilities = validation_data[
    "xpass"
].to_numpy()

xpass_predictions = (
    xpass_probabilities >= 0.5
).astype(int)


def calculate_metrics(
    model_name,
    actual,
    probabilities,
    predictions,
):
    """Calculate classification and probability metrics."""

    return {
        "model": model_name,
        "accuracy": round(
            accuracy_score(
                actual,
                predictions,
            ),
            4,
        ),
        "roc_auc": round(
            roc_auc_score(
                actual,
                probabilities,
            ),
            4,
        ),
        "log_loss": round(
            log_loss(
                actual,
                probabilities,
            ),
            4,
        ),
        "brier_score": round(
            brier_score_loss(
                actual,
                probabilities,
            ),
            4,
        ),
    }


# Evaluate the new model and nflverse xpass
model_metrics = calculate_metrics(
    model_name="situation_only_logistic_regression",
    actual=y_validation,
    probabilities=validation_probabilities,
    predictions=validation_predictions,
)

xpass_metrics = calculate_metrics(
    model_name="nflverse_xpass",
    actual=y_validation,
    probabilities=xpass_probabilities,
    predictions=xpass_predictions,
)

metrics_table = pl.DataFrame([
    model_metrics,
    xpass_metrics,
])


# Add play-level model predictions to validation data
validation_results = validation_data.with_columns([
    pl.Series(
        name="model_pass_probability",
        values=validation_probabilities,
    ),
    pl.Series(
        name="model_predicted_call",
        values=validation_predictions,
    ),
    (
        pl.col("is_pass")
        - pl.Series(
            name="model_probability",
            values=validation_probabilities,
        )
    ).alias("model_pass_oe"),
])


# Calculate coach-level validation summaries
coach_validation_summary = (
    validation_results
    .group_by([
        "season",
        "head_coach",
        "posteam",
    ])
    .agg([
        pl.len().alias("plays"),
        pl.col("is_pass")
        .mean()
        .alias("actual_pass_rate"),
        pl.col("model_pass_probability")
        .mean()
        .alias("expected_pass_rate"),
        pl.col("model_pass_oe")
        .mean()
        .alias("model_pass_oe"),
        pl.col("xpass")
        .mean()
        .alias("nflverse_expected_pass_rate"),
        pl.col("pass_oe")
        .mean()
        .alias("nflverse_pass_oe"),
        pl.col("epa")
        .mean()
        .alias("mean_epa"),
        pl.col("success")
        .mean()
        .alias("success_rate"),
    ])
    .with_columns([
        (
            100
            * pl.col("actual_pass_rate")
        )
        .round(2)
        .alias("actual_pass_rate_pct"),

        (
            100
            * pl.col("expected_pass_rate")
        )
        .round(2)
        .alias("expected_pass_rate_pct"),

        (
            100
            * pl.col("model_pass_oe")
        )
        .round(2)
        .alias("model_pass_oe_pct"),

        (
            100
            * pl.col(
                "nflverse_expected_pass_rate"
            )
        )
        .round(2)
        .alias(
            "nflverse_expected_pass_rate_pct"
        ),

        (
            100
            * pl.col("nflverse_pass_oe")
        )
        .round(2)
        .alias("nflverse_pass_oe_pct"),

        (
            100
            * pl.col("success_rate")
        )
        .round(2)
        .alias("success_rate_pct"),
    ])
    .sort(
        "model_pass_oe_pct",
        descending=True,
    )
)


# Save the trained model
joblib.dump(
    model,
    model_output_path,
)


# Save validation predictions
validation_results.write_parquet(
    predictions_output_path
)


# Save metric comparison
metrics_table.write_csv(
    metrics_output_path
)


# Create and save coach validation results
coach_summary_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_validation_summary_2024.csv"
)

coach_validation_summary.write_csv(
    coach_summary_output_path
)


# Print results
print("\nVALIDATION METRICS")
print(metrics_table)


print("\nTOP 10 MOST PASS-HEAVY COACH TENURES")
print(
    coach_validation_summary.select([
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "model_pass_oe_pct",
    ])
    .head(10)
)


print("\nTOP 10 MOST RUN-HEAVY COACH TENURES")
print(
    coach_validation_summary.select([
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "model_pass_oe_pct",
    ])
    .tail(10)
    .sort("model_pass_oe_pct")
)


print("\nSAVED FILES")
print("Model:", model_output_path)
print(
    "Validation predictions:",
    predictions_output_path,
)
print("Metrics:", metrics_output_path)
print(
    "Coach validation summary:",
    coach_summary_output_path,
)


print("\nTEST SET STATUS")
print(
    "The 2025 test set was not loaded or evaluated."
)