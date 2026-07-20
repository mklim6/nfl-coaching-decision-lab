from pathlib import Path
import json

import joblib
import numpy as np
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# Locked project settings
RANDOM_STATE = 42
MINIMUM_COACH_PLAYS = 500


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

test_path = (
    project_root
    / "data"
    / "processed"
    / "model_test_2025.parquet"
)

final_model_path = (
    project_root
    / "models"
    / "final_situation_only_model_2018_2024.joblib"
)

metadata_output_path = (
    project_root
    / "models"
    / "final_model_metadata.json"
)

test_predictions_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "final_test_predictions_2025.parquet"
)

test_metrics_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "final_test_metrics_2025.csv"
)

coach_summary_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "final_coach_summary_2025.csv"
)

coach_season_type_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "final_coach_season_type_summary_2025.csv"
)

calibration_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "final_test_calibration_2025.csv"
)


# Create output directories
final_model_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

test_predictions_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# Load chronological datasets
train_data = pl.read_parquet(
    train_path
)

validation_data = pl.read_parquet(
    validation_path
)

test_data = pl.read_parquet(
    test_path
)


# Combine training and validation after model selection
final_training_data = pl.concat(
    [
        train_data,
        validation_data,
    ],
    how="vertical_relaxed",
)


# Locked feature definitions
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


# Confirm required columns
required_columns = (
    model_features
    + [
        target_column,
        "xpass",
        "pass_oe",
        "epa",
        "success",
        "yards_gained",
    ]
)

for dataset_name, dataset in [
    ("final training", final_training_data),
    ("test", test_data),
]:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} data is missing: "
            f"{missing_columns}"
        )


# Confirm the chronological boundary
if final_training_data[
    "season"
].max() != 2024:
    raise ValueError(
        "Final training data must end in 2024."
    )

if test_data[
    "season"
].min() != 2025:
    raise ValueError(
        "Final test data must contain only 2025."
    )

if test_data[
    "season"
].max() != 2025:
    raise ValueError(
        "Final test data must contain only 2025."
    )


# Convert model inputs to pandas and NumPy
X_final_train = final_training_data.select(
    model_features
).to_pandas()

y_final_train = final_training_data[
    target_column
].to_numpy()

X_test = test_data.select(
    model_features
).to_pandas()

y_test = test_data[
    target_column
].to_numpy()


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
            "ordinal_encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)


# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)


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


# Preprocessor returns categorical columns first
categorical_mask = (
    [True] * len(categorical_features)
    + [False] * len(numeric_features)
)


# Locked final model
final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                l2_regularization=1.0,
                categorical_features=categorical_mask,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# Train on 2018 through 2024
print("TRAINING FINAL MODEL")
print(
    f"Final training plays: "
    f"{len(X_final_train):,}"
)
print("Training seasons: 2018-2024")

final_model.fit(
    X_final_train,
    y_final_train,
)

print("Final training complete")


# Perform the final 2025 evaluation
print("\nEVALUATING ON FINAL 2025 TEST SET")
print(
    f"Test plays: {len(X_test):,}"
)

final_probabilities = (
    final_model.predict_proba(
        X_test
    )[:, 1]
)

final_predictions = (
    final_probabilities >= 0.5
).astype(int)

xpass_probabilities = test_data[
    "xpass"
].to_numpy()

xpass_predictions = (
    xpass_probabilities >= 0.5
).astype(int)


def expected_calibration_error(
    actual,
    probabilities,
    number_of_bins=10,
):
    """Calculate weighted calibration error."""

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_numbers = np.digitize(
        probabilities,
        bin_edges,
        right=False,
    ) - 1

    bin_numbers = np.clip(
        bin_numbers,
        0,
        number_of_bins - 1,
    )

    total_plays = len(actual)
    calibration_error = 0.0

    for bin_number in range(
        number_of_bins
    ):
        mask = (
            bin_numbers == bin_number
        )

        plays = int(mask.sum())

        if plays == 0:
            continue

        mean_probability = (
            probabilities[mask].mean()
        )

        actual_rate = (
            actual[mask].mean()
        )

        calibration_error += (
            plays
            / total_plays
            * abs(
                actual_rate
                - mean_probability
            )
        )

    return calibration_error


def calculate_metrics(
    model_name,
    actual,
    probabilities,
    predictions,
):
    """Calculate final evaluation metrics."""

    return {
        "model": model_name,
        "plays": len(actual),
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
                labels=[0, 1],
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
        "expected_calibration_error": round(
            expected_calibration_error(
                actual,
                probabilities,
            ),
            4,
        ),
        "actual_pass_rate": round(
            actual.mean(),
            4,
        ),
        "expected_pass_rate": round(
            probabilities.mean(),
            4,
        ),
        "mean_pass_oe": round(
            (
                actual.mean()
                - probabilities.mean()
            ),
            4,
        ),
    }


# Compare final model with nflverse
final_metrics = calculate_metrics(
    model_name=(
        "final_hist_gradient_boosting"
    ),
    actual=y_test,
    probabilities=final_probabilities,
    predictions=final_predictions,
)

xpass_metrics = calculate_metrics(
    model_name="nflverse_xpass",
    actual=y_test,
    probabilities=xpass_probabilities,
    predictions=xpass_predictions,
)

metrics_table = pl.DataFrame([
    final_metrics,
    xpass_metrics,
])


# Build calibration table
calibration_rows = []
bin_edges = np.linspace(
    0.0,
    1.0,
    11,
)

for model_name, probabilities in [
    (
        "final_hist_gradient_boosting",
        final_probabilities,
    ),
    (
        "nflverse_xpass",
        xpass_probabilities,
    ),
]:
    bin_numbers = np.digitize(
        probabilities,
        bin_edges,
        right=False,
    ) - 1

    bin_numbers = np.clip(
        bin_numbers,
        0,
        9,
    )

    for bin_number in range(10):
        mask = (
            bin_numbers == bin_number
        )

        plays = int(mask.sum())

        if plays == 0:
            continue

        mean_probability = (
            probabilities[mask].mean()
        )

        actual_pass_rate = (
            y_test[mask].mean()
        )

        calibration_rows.append({
            "model": model_name,
            "probability_bin": (
                f"{bin_number / 10:.1f}-"
                f"{(bin_number + 1) / 10:.1f}"
            ),
            "plays": plays,
            "mean_probability": round(
                mean_probability,
                4,
            ),
            "actual_pass_rate": round(
                actual_pass_rate,
                4,
            ),
            "calibration_error": round(
                (
                    actual_pass_rate
                    - mean_probability
                ),
                4,
            ),
        })

calibration_table = pl.DataFrame(
    calibration_rows
)


# Add play-level predictions
test_results = (
    test_data
    .with_columns([
        pl.Series(
            name="final_model_pass_probability",
            values=final_probabilities,
        ),
        pl.Series(
            name="final_model_predicted_call",
            values=final_predictions,
        ),
    ])
    .with_columns([
        (
            pl.col("is_pass")
            - pl.col(
                "final_model_pass_probability"
            )
        ).alias("final_model_pass_oe")
    ])
)


def build_coach_summary(
    data,
    group_columns,
):
    """Build coach-level descriptive results."""

    return (
        data
        .group_by(group_columns)
        .agg([
            pl.len().alias("plays"),

            pl.col("game_id")
            .n_unique()
            .alias("games"),

            pl.col("is_pass")
            .mean()
            .alias("actual_pass_rate"),

            pl.col(
                "final_model_pass_probability"
            )
            .mean()
            .alias("expected_pass_rate"),

            pl.col(
                "final_model_pass_oe"
            )
            .mean()
            .alias("model_pass_oe"),

            pl.col("xpass")
            .mean()
            .alias(
                "nflverse_expected_pass_rate"
            ),

            pl.col("pass_oe")
            .mean()
            .alias("nflverse_pass_oe"),

            pl.col("epa")
            .mean()
            .alias("mean_epa"),

            pl.col("success")
            .mean()
            .alias("success_rate"),

            pl.col("yards_gained")
            .mean()
            .alias("mean_yards_gained"),
        ])
        .with_columns([
            (
                pl.col("plays")
                >= MINIMUM_COACH_PLAYS
            ).alias(
                "meets_minimum_sample"
            ),

            (
                100
                * pl.col("actual_pass_rate")
            )
            .round(2)
            .alias(
                "actual_pass_rate_pct"
            ),

            (
                100
                * pl.col("expected_pass_rate")
            )
            .round(2)
            .alias(
                "expected_pass_rate_pct"
            ),

            (
                100
                * pl.col("model_pass_oe")
            )
            .round(2)
            .alias(
                "model_pass_oe_pct"
            ),

            (
                100
                * pl.col(
                    "nflverse_pass_oe"
                )
            )
            .round(2)
            .alias(
                "nflverse_pass_oe_pct"
            ),

            (
                100
                * pl.col("success_rate")
            )
            .round(2)
            .alias(
                "success_rate_pct"
            ),
        ])
        .sort(
            "model_pass_oe_pct",
            descending=True,
        )
    )


# Overall coach-team summaries
coach_summary = build_coach_summary(
    test_results,
    [
        "season",
        "head_coach",
        "posteam",
    ],
)


# Separate regular-season and playoff summaries
coach_season_type_summary = (
    build_coach_summary(
        test_results,
        [
            "season",
            "season_type",
            "head_coach",
            "posteam",
        ],
    )
)


# Eligible full-season coach rankings
eligible_coaches = coach_summary.filter(
    pl.col("meets_minimum_sample")
)


# Save final model and outputs
joblib.dump(
    final_model,
    final_model_path,
)

test_results.write_parquet(
    test_predictions_output_path
)

metrics_table.write_csv(
    test_metrics_output_path
)

coach_summary.write_csv(
    coach_summary_output_path
)

coach_season_type_summary.write_csv(
    coach_season_type_output_path
)

calibration_table.write_csv(
    calibration_output_path
)


# Save model metadata
metadata = {
    "model_type": (
        "HistGradientBoostingClassifier"
    ),
    "model_purpose": (
        "Situation-only expected pass "
        "probability model"
    ),
    "training_seasons": [
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    ],
    "test_season": 2025,
    "categorical_features": (
        categorical_features
    ),
    "numeric_features": (
        numeric_features
    ),
    "excluded_from_model": [
        "head_coach",
        "posteam",
        "defteam",
        "shotgun",
        "no_huddle",
        "temp",
        "wind",
        "xpass",
        "pass_oe",
        "epa",
        "success",
        "yards_gained",
    ],
    "hyperparameters": {
        "learning_rate": 0.08,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": RANDOM_STATE,
    },
    "minimum_coach_plays": (
        MINIMUM_COACH_PLAYS
    ),
    "final_test_metrics": {
        "model": final_metrics,
        "nflverse_xpass": xpass_metrics,
    },
}

with open(
    metadata_output_path,
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(
        metadata,
        metadata_file,
        indent=4,
    )


# Print final results
print("\nFINAL 2025 TEST METRICS")
print(metrics_table)


print(
    f"\nCOACH RANKINGS REQUIRE AT LEAST "
    f"{MINIMUM_COACH_PLAYS} PLAYS"
)


print("\nTOP 10 MOST PASS-HEAVY 2025 TENURES")
print(
    eligible_coaches
    .select([
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "model_pass_oe_pct",
        "mean_epa",
        "success_rate_pct",
    ])
    .head(10)
)


print("\nTOP 10 MOST RUN-HEAVY 2025 TENURES")
print(
    eligible_coaches
    .select([
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "model_pass_oe_pct",
        "mean_epa",
        "success_rate_pct",
    ])
    .tail(10)
    .sort("model_pass_oe_pct")
)


print("\nSAVED FILES")
print("Final model:", final_model_path)
print(
    "Model metadata:",
    metadata_output_path,
)
print(
    "Test predictions:",
    test_predictions_output_path,
)
print(
    "Test metrics:",
    test_metrics_output_path,
)
print(
    "Coach summary:",
    coach_summary_output_path,
)
print(
    "Coach season-type summary:",
    coach_season_type_output_path,
)
print(
    "Calibration:",
    calibration_output_path,
)


print("\nFINAL TEST COMPLETE")
print(
    "The selected model has now been evaluated "
    "on the held-out 2025 season."
)