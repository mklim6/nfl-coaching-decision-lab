from pathlib import Path

import joblib
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


# Project settings
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

baseline_model_path = (
    project_root
    / "models"
    / "situation_only_logistic_regression.joblib"
)

candidate_model_path = (
    project_root
    / "models"
    / "situation_only_hist_gradient_boosting.joblib"
)

predictions_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_validation_predictions_2024.parquet"
)

metrics_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_validation_metrics.csv"
)

coach_summary_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_coach_summary_2024.csv"
)


# Create output directories
candidate_model_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

predictions_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# Load data
train_data = pl.read_parquet(
    train_path
)

validation_data = pl.read_parquet(
    validation_path
)


# Feature definitions
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


# Verify required columns
required_columns = (
    model_features
    + [
        target_column,
        "xpass",
        "pass_oe",
        "epa",
        "success",
    ]
)

for dataset_name, dataset in [
    ("training", train_data),
    ("validation", validation_data),
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


# Convert selected data to pandas and NumPy
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


# Encode categorical variables as integer categories
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


# Numeric variables do not require scaling for tree models
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


# The preprocessor returns categorical columns first,
# followed by numeric columns.
categorical_mask = (
    [True] * len(categorical_features)
    + [False] * len(numeric_features)
)


# Nonlinear candidate model
candidate_model = Pipeline(
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


# Train the candidate model
print("TRAINING NONLINEAR CANDIDATE MODEL")
print(
    f"Training plays: {len(X_train):,}"
)

candidate_model.fit(
    X_train,
    y_train,
)

print("Training complete")


# Load the existing logistic baseline
if not baseline_model_path.exists():
    raise FileNotFoundError(
        "The logistic baseline model was not found: "
        f"{baseline_model_path}"
    )

baseline_model = joblib.load(
    baseline_model_path
)


# Generate validation probabilities
candidate_probabilities = (
    candidate_model.predict_proba(
        X_validation
    )[:, 1]
)

baseline_probabilities = (
    baseline_model.predict_proba(
        X_validation
    )[:, 1]
)

xpass_probabilities = validation_data[
    "xpass"
].to_numpy()


# Convert probabilities to predicted classes
candidate_predictions = (
    candidate_probabilities >= 0.5
).astype(int)

baseline_predictions = (
    baseline_probabilities >= 0.5
).astype(int)

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


# Compare all three probability models
metrics_table = pl.DataFrame([
    calculate_metrics(
        model_name="logistic_regression",
        actual=y_validation,
        probabilities=baseline_probabilities,
        predictions=baseline_predictions,
    ),
    calculate_metrics(
        model_name="hist_gradient_boosting",
        actual=y_validation,
        probabilities=candidate_probabilities,
        predictions=candidate_predictions,
    ),
    calculate_metrics(
        model_name="nflverse_xpass",
        actual=y_validation,
        probabilities=xpass_probabilities,
        predictions=xpass_predictions,
    ),
])


# Add play-level predictions
validation_results = (
    validation_data
    .with_columns([
        pl.Series(
            name="logistic_pass_probability",
            values=baseline_probabilities,
        ),
        pl.Series(
            name="candidate_pass_probability",
            values=candidate_probabilities,
        ),
        pl.Series(
            name="candidate_predicted_call",
            values=candidate_predictions,
        ),
    ])
    .with_columns([
        (
            pl.col("is_pass")
            - pl.col(
                "candidate_pass_probability"
            )
        ).alias("candidate_pass_oe"),

        (
            pl.col("is_pass")
            - pl.col(
                "logistic_pass_probability"
            )
        ).alias("logistic_pass_oe"),
    ])
)


# Summarize coach-team tenures
coach_summary = (
    validation_results
    .group_by([
        "season",
        "head_coach",
        "posteam",
    ])
    .agg([
        pl.len().alias("plays"),

        pl.col("game_id")
        .n_unique()
        .alias("games"),

        pl.col("is_pass")
        .mean()
        .alias("actual_pass_rate"),

        pl.col(
            "candidate_pass_probability"
        )
        .mean()
        .alias(
            "candidate_expected_pass_rate"
        ),

        pl.col("candidate_pass_oe")
        .mean()
        .alias("candidate_pass_oe"),

        pl.col(
            "logistic_pass_probability"
        )
        .mean()
        .alias(
            "logistic_expected_pass_rate"
        ),

        pl.col("logistic_pass_oe")
        .mean()
        .alias("logistic_pass_oe"),

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
    ])
    .with_columns([
        (
            pl.col("plays")
            >= MINIMUM_COACH_PLAYS
        ).alias("meets_minimum_sample"),

        (
            100
            * pl.col("actual_pass_rate")
        )
        .round(2)
        .alias("actual_pass_rate_pct"),

        (
            100
            * pl.col(
                "candidate_expected_pass_rate"
            )
        )
        .round(2)
        .alias(
            "candidate_expected_pass_rate_pct"
        ),

        (
            100
            * pl.col("candidate_pass_oe")
        )
        .round(2)
        .alias("candidate_pass_oe_pct"),

        (
            100
            * pl.col(
                "logistic_pass_oe"
            )
        )
        .round(2)
        .alias("logistic_pass_oe_pct"),

        (
            100
            * pl.col(
                "nflverse_pass_oe"
            )
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
        "candidate_pass_oe_pct",
        descending=True,
    )
)


# Apply the minimum sample threshold for rankings
eligible_coaches = coach_summary.filter(
    pl.col("meets_minimum_sample")
)


# Save outputs
joblib.dump(
    candidate_model,
    candidate_model_path,
)

validation_results.write_parquet(
    predictions_output_path
)

metrics_table.write_csv(
    metrics_output_path
)

coach_summary.write_csv(
    coach_summary_output_path
)


# Print model comparison
print("\nVALIDATION MODEL COMPARISON")
print(metrics_table)


print(
    f"\nCOACH RANKINGS REQUIRE AT LEAST "
    f"{MINIMUM_COACH_PLAYS} PLAYS"
)


print("\nTOP 10 MOST PASS-HEAVY TENURES")
print(
    eligible_coaches
    .select([
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "candidate_expected_pass_rate_pct",
        "candidate_pass_oe_pct",
    ])
    .head(10)
)


print("\nTOP 10 MOST RUN-HEAVY TENURES")
print(
    eligible_coaches
    .select([
        "head_coach",
        "posteam",
        "plays",
        "actual_pass_rate_pct",
        "candidate_expected_pass_rate_pct",
        "candidate_pass_oe_pct",
    ])
    .tail(10)
    .sort("candidate_pass_oe_pct")
)


print("\nSAVED FILES")
print(
    "Candidate model:",
    candidate_model_path,
)
print(
    "Validation predictions:",
    predictions_output_path,
)
print(
    "Metrics:",
    metrics_output_path,
)
print(
    "Coach summary:",
    coach_summary_output_path,
)


print("\nTEST SET STATUS")
print(
    "The 2025 test set was not loaded or evaluated."
)