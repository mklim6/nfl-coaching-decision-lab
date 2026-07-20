from pathlib import Path

import numpy as np
import polars as pl


pl.Config.set_tbl_rows(50)

# Reproducible bootstrap settings
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 2000
MINIMUM_COACH_PLAYS = 500


# Project paths
project_root = Path(__file__).resolve().parents[1]

predictions_path = (
    project_root
    / "outputs"
    / "tables"
    / "historical_play_predictions_2018_2025.parquet"
)

coach_summary_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_season_summary_2018_2025.csv"
)

output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_season_uncertainty_2018_2025.csv"
)


# Load historical predictions and existing summaries
predictions = pl.read_parquet(
    predictions_path
)

coach_summary = pl.read_csv(
    coach_summary_path
)


# Aggregate play-level residuals into game clusters
game_clusters = (
    predictions
    .group_by([
        "season",
        "head_coach",
        "posteam",
        "game_id",
    ])
    .agg([
        pl.len().alias(
            "game_plays"
        ),
        pl.col("model_pass_oe")
        .sum()
        .alias(
            "game_residual_sum"
        ),
    ])
    .sort([
        "season",
        "head_coach",
        "posteam",
        "game_id",
    ])
)


# Split into coach-team-season groups
coach_season_groups = (
    game_clusters.partition_by(
        [
            "season",
            "head_coach",
            "posteam",
        ],
        maintain_order=True,
    )
)


rng = np.random.default_rng(
    RANDOM_STATE
)

bootstrap_rows = []


print("BUILDING GAME-CLUSTERED CONFIDENCE INTERVALS")
print(
    f"Bootstrap samples per coach-season: "
    f"{BOOTSTRAP_SAMPLES:,}"
)
print(
    f"Coach-season groups: "
    f"{len(coach_season_groups):,}"
)


for group_number, group in enumerate(
    coach_season_groups,
    start=1,
):
    season = group[
        "season"
    ][0]

    head_coach = group[
        "head_coach"
    ][0]

    posteam = group[
        "posteam"
    ][0]

    game_residual_sums = group[
        "game_residual_sum"
    ].to_numpy()

    game_play_counts = group[
        "game_plays"
    ].to_numpy()

    number_of_games = len(
        game_residual_sums
    )

    total_plays = int(
        game_play_counts.sum()
    )

    observed_mean = (
        game_residual_sums.sum()
        / game_play_counts.sum()
    )

    # Sample complete games with replacement
    sampled_game_indices = rng.integers(
        low=0,
        high=number_of_games,
        size=(
            BOOTSTRAP_SAMPLES,
            number_of_games,
        ),
    )

    sampled_residual_sums = (
        game_residual_sums[
            sampled_game_indices
        ].sum(axis=1)
    )

    sampled_play_counts = (
        game_play_counts[
            sampled_game_indices
        ].sum(axis=1)
    )

    bootstrap_means = (
        sampled_residual_sums
        / sampled_play_counts
    )

    lower_bound = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper_bound = np.percentile(
        bootstrap_means,
        97.5,
    )

    bootstrap_standard_error = (
        bootstrap_means.std(
            ddof=1
        )
    )

    probability_pass_heavy = (
        bootstrap_means > 0
    ).mean()

    probability_run_heavy = (
        bootstrap_means < 0
    ).mean()

    bootstrap_rows.append({
        "season": season,
        "head_coach": head_coach,
        "posteam": posteam,
        "bootstrap_games": (
            number_of_games
        ),
        "bootstrap_plays": total_plays,
        "observed_pass_oe": round(
            observed_mean,
            6,
        ),
        "bootstrap_standard_error": round(
            bootstrap_standard_error,
            6,
        ),
        "ci_95_lower": round(
            lower_bound,
            6,
        ),
        "ci_95_upper": round(
            upper_bound,
            6,
        ),
        "probability_pass_heavy": round(
            probability_pass_heavy,
            4,
        ),
        "probability_run_heavy": round(
            probability_run_heavy,
            4,
        ),
    })

    if (
        group_number % 50 == 0
        or group_number
        == len(coach_season_groups)
    ):
        print(
            f"Completed "
            f"{group_number:,} of "
            f"{len(coach_season_groups):,}"
        )


bootstrap_table = pl.DataFrame(
    bootstrap_rows
)


# Add percentage-point versions and labels
bootstrap_table = (
    bootstrap_table
    .with_columns([
        (
            100
            * pl.col(
                "observed_pass_oe"
            )
        )
        .round(2)
        .alias(
            "observed_pass_oe_pct"
        ),

        (
            100
            * pl.col(
                "bootstrap_standard_error"
            )
        )
        .round(2)
        .alias(
            "bootstrap_standard_error_pct"
        ),

        (
            100
            * pl.col("ci_95_lower")
        )
        .round(2)
        .alias(
            "ci_95_lower_pct"
        ),

        (
            100
            * pl.col("ci_95_upper")
        )
        .round(2)
        .alias(
            "ci_95_upper_pct"
        ),
    ])
    .with_columns([
        (
            (
                pl.col("ci_95_lower")
                > 0
            )
            | (
                pl.col("ci_95_upper")
                < 0
            )
        ).alias(
            "ci_excludes_zero"
        ),

        pl.when(
            pl.col("ci_95_lower") > 0
        )
        .then(
            pl.lit("pass-heavy")
        )
        .when(
            pl.col("ci_95_upper") < 0
        )
        .then(
            pl.lit("run-heavy")
        )
        .otherwise(
            pl.lit("uncertain")
        )
        .alias("tendency_label"),
    ])
)


# Join confidence intervals to the existing coach summary
coach_uncertainty = (
    coach_summary
    .join(
        bootstrap_table,
        on=[
            "season",
            "head_coach",
            "posteam",
        ],
        how="left",
    )
    .with_columns(
        (
            pl.col("plays")
            >= MINIMUM_COACH_PLAYS
        ).alias(
            "meets_minimum_sample"
        )
    )
    .sort([
        "season",
        "model_pass_oe_pct",
    ], descending=[False, True])
)


# Save the final uncertainty table
coach_uncertainty.write_csv(
    output_path
)


# Eligible coach-seasons
eligible_coaches = (
    coach_uncertainty.filter(
        pl.col("meets_minimum_sample")
    )
)

clear_tendencies = (
    eligible_coaches.filter(
        pl.col("ci_excludes_zero")
    )
)


# Validation output
print("\nUNCERTAINTY ANALYSIS COMPLETE")


print("\nSUMMARY COUNTS")
print(
    pl.DataFrame({
        "measure": [
            "all_coach_seasons",
            "eligible_coach_seasons",
            "ci_excludes_zero",
            "pass_heavy",
            "run_heavy",
            "uncertain",
        ],
        "count": [
            coach_uncertainty.height,
            eligible_coaches.height,
            clear_tendencies.height,
            eligible_coaches.filter(
                pl.col("tendency_label")
                == "pass-heavy"
            ).height,
            eligible_coaches.filter(
                pl.col("tendency_label")
                == "run-heavy"
            ).height,
            eligible_coaches.filter(
                pl.col("tendency_label")
                == "uncertain"
            ).height,
        ],
    })
)


print("\nSTRONGEST PASS-HEAVY COACH-SEASONS")
print(
    clear_tendencies
    .filter(
        pl.col("tendency_label")
        == "pass-heavy"
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
        "model_pass_oe_pct",
        "ci_95_lower_pct",
        "ci_95_upper_pct",
        "mean_epa",
        "success_rate_pct",
    ])
    .head(15)
)


print("\nSTRONGEST RUN-HEAVY COACH-SEASONS")
print(
    clear_tendencies
    .filter(
        pl.col("tendency_label")
        == "run-heavy"
    )
    .sort(
        "model_pass_oe_pct"
    )
    .select([
        "season",
        "head_coach",
        "posteam",
        "plays",
        "model_pass_oe_pct",
        "ci_95_lower_pct",
        "ci_95_upper_pct",
        "mean_epa",
        "success_rate_pct",
    ])
    .head(15)
)


print("\nSAVED FILE")
print(output_path)