from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


pl.Config.set_tbl_rows(100)

MINIMUM_GROUP_PLAYS = 200


# Project paths
project_root = Path(__file__).resolve().parents[1]

predictions_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_validation_predictions_2024.parquet"
)

calibration_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_calibration_2024.csv"
)

segment_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_segment_metrics_2024.csv"
)

down_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_metrics_by_down_2024.csv"
)

quarter_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_metrics_by_quarter_2024.csv"
)

season_type_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "candidate_metrics_by_season_type_2024.csv"
)


# Load saved validation predictions
validation = pl.read_parquet(
    predictions_path
)


# Probability columns being compared
model_probability_columns = {
    "logistic_regression": (
        "logistic_pass_probability"
    ),
    "hist_gradient_boosting": (
        "candidate_pass_probability"
    ),
    "nflverse_xpass": "xpass",
}


def calculate_metrics(
    actual,
    probabilities,
):
    """Calculate metrics safely for a group of plays."""

    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "plays": len(actual),
        "accuracy": round(
            accuracy_score(
                actual,
                predictions,
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

    if len(np.unique(actual)) == 2:
        metrics["roc_auc"] = round(
            roc_auc_score(
                actual,
                probabilities,
            ),
            4,
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def build_group_metrics(
    data,
    group_column,
):
    """Calculate model metrics within each group."""

    rows = []

    group_values = (
        data[group_column]
        .unique()
        .sort()
        .to_list()
    )

    for group_value in group_values:
        group_data = data.filter(
            pl.col(group_column) == group_value
        )

        if group_data.height < MINIMUM_GROUP_PLAYS:
            continue

        actual = group_data[
            "is_pass"
        ].to_numpy()

        for model_name, probability_column in (
            model_probability_columns.items()
        ):
            probabilities = group_data[
                probability_column
            ].to_numpy()

            metrics = calculate_metrics(
                actual,
                probabilities,
            )

            rows.append({
                "group_variable": group_column,
                "group_value": str(group_value),
                "model": model_name,
                **metrics,
            })

    return pl.DataFrame(rows)


def build_calibration_table(
    data,
):
    """Create decile-based probability calibration results."""

    rows = []

    bin_edges = np.linspace(
        0.0,
        1.0,
        11,
    )

    actual = data[
        "is_pass"
    ].to_numpy()

    for model_name, probability_column in (
        model_probability_columns.items()
    ):
        probabilities = data[
            probability_column
        ].to_numpy()

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
                actual[mask].mean()
            )

            rows.append({
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
                "absolute_calibration_error": round(
                    abs(
                        actual_pass_rate
                        - mean_probability
                    ),
                    4,
                ),
            })

    return pl.DataFrame(rows)


# Add useful situational segments
validation = validation.with_columns([
    pl.when(
        pl.col("score_differential") <= -9
    )
    .then(pl.lit("trailing_9_plus"))
    .when(
        pl.col("score_differential") >= 9
    )
    .then(pl.lit("leading_9_plus"))
    .otherwise(pl.lit("within_8_points"))
    .alias("score_state"),

    pl.when(
        pl.col("game_seconds_remaining") <= 300
    )
    .then(pl.lit("final_5_minutes"))
    .when(
        pl.col("game_seconds_remaining") <= 900
    )
    .then(pl.lit("final_15_minutes"))
    .otherwise(pl.lit("earlier_game"))
    .alias("time_state"),

    pl.when(
        (
            pl.col("game_seconds_remaining") <= 300
        )
        & (
            pl.col("score_differential")
            .abs()
            <= 8
        )
    )
    .then(pl.lit("close_final_5"))
    .otherwise(pl.lit("other_situations"))
    .alias("close_late_state"),

    pl.when(
        pl.col("ydstogo") <= 2
    )
    .then(pl.lit("short_1_to_2"))
    .when(
        pl.col("ydstogo") >= 10
    )
    .then(pl.lit("long_10_plus"))
    .otherwise(pl.lit("medium_3_to_9"))
    .alias("distance_group"),

    pl.when(
        pl.col("yardline_100") <= 20
    )
    .then(pl.lit("red_zone"))
    .otherwise(pl.lit("outside_red_zone"))
    .alias("field_zone"),
])


# Overall model comparison
overall_rows = []

actual = validation[
    "is_pass"
].to_numpy()

for model_name, probability_column in (
    model_probability_columns.items()
):
    probabilities = validation[
        probability_column
    ].to_numpy()

    overall_rows.append({
        "model": model_name,
        **calculate_metrics(
            actual,
            probabilities,
        ),
    })

overall_metrics = pl.DataFrame(
    overall_rows
)


# Calibration analysis
calibration_table = build_calibration_table(
    validation
)

calibration_summary = (
    calibration_table
    .group_by("model")
    .agg([
        (
            (
                pl.col(
                    "absolute_calibration_error"
                )
                * pl.col("plays")
            ).sum()
            / pl.col("plays").sum()
        )
        .round(4)
        .alias(
            "expected_calibration_error"
        ),

        pl.col(
            "absolute_calibration_error"
        )
        .max()
        .round(4)
        .alias(
            "maximum_calibration_error"
        ),
    ])
    .sort(
        "expected_calibration_error"
    )
)


# Grouped audits
down_metrics = build_group_metrics(
    validation,
    "down",
)

quarter_metrics = build_group_metrics(
    validation,
    "qtr",
)

season_type_metrics = build_group_metrics(
    validation,
    "season_type",
)

segment_tables = []

for segment_column in [
    "score_state",
    "time_state",
    "close_late_state",
    "distance_group",
    "field_zone",
]:
    segment_tables.append(
        build_group_metrics(
            validation,
            segment_column,
        )
    )

segment_metrics = pl.concat(
    segment_tables,
    how="vertical_relaxed",
)


# Save audit tables
calibration_table.write_csv(
    calibration_output_path
)

segment_metrics.write_csv(
    segment_output_path
)

down_metrics.write_csv(
    down_output_path
)

quarter_metrics.write_csv(
    quarter_output_path
)

season_type_metrics.write_csv(
    season_type_output_path
)


# Print audit results
print("OVERALL VALIDATION METRICS")
print(overall_metrics)


print("\nCALIBRATION SUMMARY")
print(calibration_summary)


print(
    "\nHISTOGRAM GRADIENT BOOSTING "
    "CALIBRATION BY PROBABILITY BIN"
)
print(
    calibration_table.filter(
        pl.col("model")
        == "hist_gradient_boosting"
    )
)


print("\nPERFORMANCE BY SEASON TYPE")
print(season_type_metrics)


print("\nHISTOGRAM GRADIENT BOOSTING BY DOWN")
print(
    down_metrics.filter(
        pl.col("model")
        == "hist_gradient_boosting"
    )
)


print("\nHISTOGRAM GRADIENT BOOSTING BY QUARTER")
print(
    quarter_metrics.filter(
        pl.col("model")
        == "hist_gradient_boosting"
    )
)


print("\nIMPORTANT SITUATIONAL SEGMENTS")
print(segment_metrics)


print("\nSAVED FILES")
print(
    "Calibration:",
    calibration_output_path,
)
print(
    "Situation segments:",
    segment_output_path,
)
print(
    "Down metrics:",
    down_output_path,
)
print(
    "Quarter metrics:",
    quarter_output_path,
)
print(
    "Season-type metrics:",
    season_type_output_path,
)


print("\nTEST SET STATUS")
print(
    "The audit used only 2024 validation predictions. "
    "The 2025 test set remains untouched."
)