from pathlib import Path

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


# Locked model settings
RANDOM_STATE = 42
MINIMUM_COACH_PLAYS = 500

development_seasons = list(
    range(2018, 2025)
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

final_test_predictions_path = (
    project_root
    / "outputs"
    / "tables"
    / "final_test_predictions_2025.parquet"
)

historical_predictions_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "historical_play_predictions_2018_2025.parquet"
)

season_metrics_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "historical_model_metrics_by_season.csv"
)

coach_season_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_season_summary_2018_2025.csv"
)

coach_season_type_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_season_type_summary_2018_2025.csv"
)


# Ensure output directory exists
historical_predictions_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# Load development data
training_data = pl.read_parquet(
    train_path
)

validation_data = pl.read_parquet(
    validation_path
)

development_data = pl.concat(
    [
        training_data,
        validation_data,
    ],
    how="vertical_relaxed",
)


# Load finalized 2025 predictions
final_test_predictions = pl.read_parquet(
    final_test_predictions_path
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


# Columns present before predictions are added
base_columns = development_data.columns


# Use the same prediction-column order everywhere
prediction_columns = [
    "expected_pass_probability",
    "predicted_is_pass",
    "prediction_source",
    "model_pass_oe",
]


def create_locked_model():
    """Create a new copy of the locked final model."""

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
                    handle_unknown=(
                        "use_encoded_value"
                    ),
                    unknown_value=-1,
                ),
            ),
        ]
    )

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

    categorical_mask = (
        [True] * len(categorical_features)
        + [False] * len(numeric_features)
    )

    return Pipeline(
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
                    categorical_features=(
                        categorical_mask
                    ),
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(
    model_name,
    season,
    actual,
    probabilities,
):
    """Calculate season-level model metrics."""

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "season": season,
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


# Store cross-fitted results and metrics
cross_fitted_seasons = []
metric_rows = []


# Generate out-of-sample predictions for 2018–2024
for held_out_season in development_seasons:
    print(
        f"\nCROSS-FITTING SEASON "
        f"{held_out_season}"
    )

    fold_training_data = (
        development_data.filter(
            pl.col("season")
            != held_out_season
        )
    )

    fold_holdout_data = (
        development_data.filter(
            pl.col("season")
            == held_out_season
        )
    )

    print(
        f"Fold training plays: "
        f"{fold_training_data.height:,}"
    )
    print(
        f"Held-out plays: "
        f"{fold_holdout_data.height:,}"
    )

    X_fold_train = (
        fold_training_data
        .select(model_features)
        .to_pandas()
    )

    y_fold_train = fold_training_data[
        target_column
    ].to_numpy()

    X_fold_holdout = (
        fold_holdout_data
        .select(model_features)
        .to_pandas()
    )

    y_fold_holdout = fold_holdout_data[
        target_column
    ].to_numpy()

    fold_model = create_locked_model()

    fold_model.fit(
        X_fold_train,
        y_fold_train,
    )

    fold_probabilities = (
        fold_model.predict_proba(
            X_fold_holdout
        )[:, 1]
    )

    fold_predictions = (
        fold_probabilities >= 0.5
    ).astype(int)

    fold_results = (
        fold_holdout_data
        .with_columns([
            pl.Series(
                name=(
                    "expected_pass_probability"
                ),
                values=fold_probabilities,
            ),
            pl.Series(
                name="predicted_is_pass",
                values=fold_predictions,
            ),
            pl.lit(
                "leave_one_season_out"
            ).alias(
                "prediction_source"
            ),
        ])
        .with_columns(
            (
                pl.col("is_pass")
                - pl.col(
                    "expected_pass_probability"
                )
            ).alias("model_pass_oe")
        )
        .select(
            base_columns
            + prediction_columns
        )
    )

    cross_fitted_seasons.append(
        fold_results
    )

    metric_rows.append(
        calculate_metrics(
            model_name=(
                "cross_fitted_locked_model"
            ),
            season=held_out_season,
            actual=y_fold_holdout,
            probabilities=(
                fold_probabilities
            ),
        )
    )

    xpass_probabilities = (
        fold_holdout_data[
            "xpass"
        ].to_numpy()
    )

    metric_rows.append(
        calculate_metrics(
            model_name="nflverse_xpass",
            season=held_out_season,
            actual=y_fold_holdout,
            probabilities=(
                xpass_probabilities
            ),
        )
    )

    print(
        f"{held_out_season} predictions complete"
    )


# Combine 2018–2024 cross-fitted predictions
historical_development_predictions = (
    pl.concat(
        cross_fitted_seasons,
        how="vertical_relaxed",
    )
)


# Standardize finalized 2025 column names
test_2025_standardized = (
    final_test_predictions
    .with_columns([
        pl.col(
            "final_model_pass_probability"
        ).alias(
            "expected_pass_probability"
        ),

        pl.col(
            "final_model_predicted_call"
        ).alias(
            "predicted_is_pass"
        ),

        pl.lit(
            "held_out_final_test"
        ).alias(
            "prediction_source"
        ),

        pl.col(
            "final_model_pass_oe"
        ).alias(
            "model_pass_oe"
        ),
    ])
    .select(
        base_columns
        + prediction_columns
    )
)


# Add 2025 metrics
actual_2025 = test_2025_standardized[
    "is_pass"
].to_numpy()

model_probabilities_2025 = (
    test_2025_standardized[
        "expected_pass_probability"
    ].to_numpy()
)

xpass_probabilities_2025 = (
    test_2025_standardized[
        "xpass"
    ].to_numpy()
)

metric_rows.append(
    calculate_metrics(
        model_name="final_locked_model",
        season=2025,
        actual=actual_2025,
        probabilities=(
            model_probabilities_2025
        ),
    )
)

metric_rows.append(
    calculate_metrics(
        model_name="nflverse_xpass",
        season=2025,
        actual=actual_2025,
        probabilities=(
            xpass_probabilities_2025
        ),
    )
)


# Combine all seasons
historical_predictions = (
    pl.concat(
        [
            historical_development_predictions,
            test_2025_standardized,
        ],
        how="vertical_relaxed",
    )
    .sort([
        "season",
        "game_id",
        "play_id",
    ])
)


def build_coach_summary(
    data,
    group_columns,
):
    """Build coach tendency and outcome summaries."""

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
                "expected_pass_probability"
            )
            .mean()
            .alias(
                "expected_pass_rate"
            ),

            pl.col("model_pass_oe")
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
            .alias(
                "mean_yards_gained"
            ),
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
            [
                "season",
                "model_pass_oe_pct",
            ],
            descending=[
                False,
                True,
            ],
        )
    )


# Build historical coach tables
coach_season_summary = (
    build_coach_summary(
        historical_predictions,
        [
            "season",
            "head_coach",
            "posteam",
        ],
    )
)

coach_season_type_summary = (
    build_coach_summary(
        historical_predictions,
        [
            "season",
            "season_type",
            "head_coach",
            "posteam",
        ],
    )
)


# Create season-level metric table
season_metrics = (
    pl.DataFrame(metric_rows)
    .sort([
        "season",
        "model",
    ])
)


# Save outputs
historical_predictions.write_parquet(
    historical_predictions_output_path
)

season_metrics.write_csv(
    season_metrics_output_path
)

coach_season_summary.write_csv(
    coach_season_output_path
)

coach_season_type_summary.write_csv(
    coach_season_type_output_path
)


# Final validation output
print("\nHISTORICAL PREDICTIONS CREATED")
print(
    "Shape:",
    historical_predictions.shape,
)


print("\nPREDICTION COUNTS BY SEASON")
print(
    historical_predictions
    .group_by([
        "season",
        "prediction_source",
    ])
    .len()
    .sort("season")
)


print("\nMODEL METRICS BY SEASON")
print(season_metrics)


print("\nELIGIBLE COACH-SEASON COUNT")
print(
    coach_season_summary
    .filter(
        pl.col("meets_minimum_sample")
    )
    .select(
        pl.len().alias(
            "eligible_coach_seasons"
        )
    )
)


print("\nTOP PASS-HEAVY COACH-SEASONS")
print(
    coach_season_summary
    .filter(
        pl.col("meets_minimum_sample")
    )
    .sort(
        "model_pass_oe_pct",
        descending=True,
    )
    .select([
        "season",
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "model_pass_oe_pct",
        "mean_epa",
        "success_rate_pct",
    ])
    .head(15)
)


print("\nTOP RUN-HEAVY COACH-SEASONS")
print(
    coach_season_summary
    .filter(
        pl.col("meets_minimum_sample")
    )
    .sort(
        "model_pass_oe_pct"
    )
    .select([
        "season",
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "model_pass_oe_pct",
        "mean_epa",
        "success_rate_pct",
    ])
    .head(15)
)


print("\nSAVED FILES")
print(
    "Historical play predictions:",
    historical_predictions_output_path,
)
print(
    "Season metrics:",
    season_metrics_output_path,
)
print(
    "Coach-season summary:",
    coach_season_output_path,
)
print(
    "Coach season-type summary:",
    coach_season_type_output_path,
)