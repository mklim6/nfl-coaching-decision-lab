from pathlib import Path

import numpy as np
import polars as pl


pl.Config.set_tbl_rows(50)

# Reproducible bootstrap settings
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 2000
MINIMUM_CALLER_PLAYS = 500


# Project paths
project_root = Path(__file__).resolve().parents[1]

predictions_path = (
    project_root
    / "outputs"
    / "tables"
    / "historical_play_predictions_2018_2025.parquet"
)

play_caller_reference_path = (
    project_root
    / "data"
    / "reference"
    / "offensive_play_caller_tenures.csv"
)

output_path = (
    project_root
    / "outputs"
    / "tables"
    / "play_caller_season_uncertainty_2018_2025.csv"
)


# Confirm required inputs exist
required_paths = [
    predictions_path,
    play_caller_reference_path,
]

missing_paths = [
    path
    for path in required_paths
    if not path.exists()
]

if missing_paths:
    missing_list = "\n".join(
        f"- {path}"
        for path in missing_paths
    )

    raise FileNotFoundError(
        "Required input files were not found:\n"
        f"{missing_list}"
    )


print("LOADING HISTORICAL PLAY PREDICTIONS")

predictions = pl.read_parquet(
    predictions_path
)


print("LOADING VERIFIED PLAY-CALLER REFERENCE")

play_caller_reference = (
    pl.read_csv(
        play_caller_reference_path,
        infer_schema_length=10000,
    )
    .filter(
        pl.col("verification_status")
        == "verified"
    )
)


# Expand each verified reference segment to one row per season-team-coach-week.
# This converts the interval match into a normal equality join and makes
# overlap validation straightforward.
reference_by_week = (
    play_caller_reference
    .with_columns(
        pl.int_ranges(
            pl.col("start_week"),
            pl.col("end_week") + 1,
        ).alias("week")
    )
    .explode("week")
    .select([
        "tenure_id",
        "season",
        pl.col("team").alias("posteam"),
        "head_coach",
        "week",
        "offensive_play_caller",
        "caller_role",
        "head_coach_is_play_caller",
        "verification_status",
    ])
)


duplicate_reference_keys = (
    reference_by_week
    .group_by([
        "season",
        "posteam",
        "head_coach",
        "week",
    ])
    .len()
    .filter(pl.col("len") > 1)
    .height
)

if duplicate_reference_keys > 0:
    raise RuntimeError(
        "The verified play-caller reference contains "
        f"{duplicate_reference_keys:,} overlapping weekly keys."
    )


# Attach the verified offensive play caller to every historical play.
caller_predictions = predictions.join(
    reference_by_week,
    on=[
        "season",
        "posteam",
        "head_coach",
        "week",
    ],
    how="left",
    validate="m:1",
)


missing_caller_plays = caller_predictions.filter(
    pl.col("offensive_play_caller").is_null()
).height

if caller_predictions.height != predictions.height:
    raise RuntimeError(
        "The caller join changed the number of play rows. "
        f"Expected {predictions.height:,}; received "
        f"{caller_predictions.height:,}."
    )

if missing_caller_plays > 0:
    raise RuntimeError(
        f"{missing_caller_plays:,} plays did not match a "
        "verified offensive play caller."
    )


# Build the caller-team-season summary used as the base output table.
caller_summary = (
    caller_predictions
    .group_by([
        "season",
        "offensive_play_caller",
        "posteam",
    ])
    .agg([
        pl.len().alias("plays"),
        pl.col("game_id")
        .n_unique()
        .alias("games"),
        pl.col("head_coach")
        .n_unique()
        .alias("head_coach_count"),
        (
            100 * pl.col("is_pass").mean()
        )
        .round(2)
        .alias("actual_pass_rate_pct"),
        (
            100
            * pl.col(
                "expected_pass_probability"
            ).mean()
        )
        .round(2)
        .alias("expected_pass_rate_pct"),
        (
            100
            * pl.col("model_pass_oe").mean()
        )
        .round(2)
        .alias("model_pass_oe_pct"),
        (
            100 * pl.col("pass_oe").mean()
        )
        .round(2)
        .alias("nflverse_pass_oe_pct"),
        pl.col("epa")
        .mean()
        .round(4)
        .alias("mean_epa"),
        (
            100 * pl.col("success").mean()
        )
        .round(2)
        .alias("success_rate_pct"),
        pl.col("yards_gained")
        .mean()
        .round(2)
        .alias("mean_yards_gained"),
    ])
)


# Aggregate play-level residuals into complete-game clusters.
game_clusters = (
    caller_predictions
    .group_by([
        "season",
        "offensive_play_caller",
        "posteam",
        "game_id",
    ])
    .agg([
        pl.len().alias("game_plays"),
        pl.col("model_pass_oe")
        .sum()
        .alias("game_residual_sum"),
    ])
    .sort([
        "season",
        "offensive_play_caller",
        "posteam",
        "game_id",
    ])
)


caller_season_groups = (
    game_clusters.partition_by(
        [
            "season",
            "offensive_play_caller",
            "posteam",
        ],
        maintain_order=True,
    )
)


rng = np.random.default_rng(
    RANDOM_STATE
)

bootstrap_rows = []


print("\nBUILDING GAME-CLUSTERED PLAY-CALLER INTERVALS")
print(
    "Bootstrap samples per caller-season: "
    f"{BOOTSTRAP_SAMPLES:,}"
)
print(
    "Caller-team-season groups: "
    f"{len(caller_season_groups):,}"
)


for group_number, group in enumerate(
    caller_season_groups,
    start=1,
):
    season = group["season"][0]
    offensive_play_caller = group[
        "offensive_play_caller"
    ][0]
    posteam = group["posteam"][0]

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

    # Sample complete games with replacement. Weighting by the resampled
    # number of plays retains the play-level estimand while respecting the
    # within-game dependence structure.
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
        bootstrap_means.std(ddof=1)
    )

    probability_pass_heavy = (
        bootstrap_means > 0
    ).mean()

    probability_run_heavy = (
        bootstrap_means < 0
    ).mean()

    bootstrap_rows.append({
        "season": season,
        "offensive_play_caller": (
            offensive_play_caller
        ),
        "posteam": posteam,
        "bootstrap_games": number_of_games,
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
        == len(caller_season_groups)
    ):
        print(
            f"Completed {group_number:,} of "
            f"{len(caller_season_groups):,}"
        )


bootstrap_table = pl.DataFrame(
    bootstrap_rows
)


# Add percentage-point versions and evidence labels.
bootstrap_table = (
    bootstrap_table
    .with_columns([
        (
            100 * pl.col("observed_pass_oe")
        )
        .round(2)
        .alias("observed_pass_oe_pct"),
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
            100 * pl.col("ci_95_lower")
        )
        .round(2)
        .alias("ci_95_lower_pct"),
        (
            100 * pl.col("ci_95_upper")
        )
        .round(2)
        .alias("ci_95_upper_pct"),
    ])
    .with_columns([
        (
            (pl.col("ci_95_lower") > 0)
            | (pl.col("ci_95_upper") < 0)
        ).alias("ci_excludes_zero"),
        pl.when(pl.col("ci_95_lower") > 0)
        .then(pl.lit("pass-heavy"))
        .when(pl.col("ci_95_upper") < 0)
        .then(pl.lit("run-heavy"))
        .otherwise(pl.lit("uncertain"))
        .alias("tendency_label"),
    ])
)


play_caller_uncertainty = (
    caller_summary
    .join(
        bootstrap_table,
        on=[
            "season",
            "offensive_play_caller",
            "posteam",
        ],
        how="left",
        validate="1:1",
    )
    .with_columns(
        (
            pl.col("plays")
            >= MINIMUM_CALLER_PLAYS
        ).alias("meets_minimum_sample")
    )
    .sort(
        ["season", "model_pass_oe_pct"],
        descending=[False, True],
    )
)


if play_caller_uncertainty[
    "ci_95_lower"
].null_count() > 0:
    raise RuntimeError(
        "At least one caller summary failed to match its "
        "bootstrap result."
    )


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

play_caller_uncertainty.write_csv(
    output_path
)


eligible_callers = (
    play_caller_uncertainty.filter(
        pl.col("meets_minimum_sample")
    )
)

clear_tendencies = (
    eligible_callers.filter(
        pl.col("ci_excludes_zero")
    )
)


print("\nPLAY-CALLER UNCERTAINTY ANALYSIS COMPLETE")


print("\nJOIN VALIDATION")
print(f"Prediction rows: {predictions.height:,}")
print(
    "Caller-enriched rows: "
    f"{caller_predictions.height:,}"
)
print(
    "Missing verified caller plays: "
    f"{missing_caller_plays:,}"
)
print(
    "Overlapping weekly reference keys: "
    f"{duplicate_reference_keys:,}"
)


print("\nSUMMARY COUNTS")
print(
    pl.DataFrame({
        "measure": [
            "all_caller_seasons",
            "eligible_caller_seasons",
            "ci_excludes_zero",
            "pass_heavy",
            "run_heavy",
            "uncertain",
        ],
        "count": [
            play_caller_uncertainty.height,
            eligible_callers.height,
            clear_tendencies.height,
            eligible_callers.filter(
                pl.col("tendency_label")
                == "pass-heavy"
            ).height,
            eligible_callers.filter(
                pl.col("tendency_label")
                == "run-heavy"
            ).height,
            eligible_callers.filter(
                pl.col("tendency_label")
                == "uncertain"
            ).height,
        ],
    })
)


display_columns = [
    "season",
    "offensive_play_caller",
    "posteam",
    "plays",
    "model_pass_oe_pct",
    "ci_95_lower_pct",
    "ci_95_upper_pct",
    "mean_epa",
    "success_rate_pct",
]


print("\nSTRONGEST PASS-HEAVY CALLER-SEASONS")
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
    .select(display_columns)
    .head(15)
)


print("\nSTRONGEST RUN-HEAVY CALLER-SEASONS")
print(
    clear_tendencies
    .filter(
        pl.col("tendency_label")
        == "run-heavy"
    )
    .sort("model_pass_oe_pct")
    .select(display_columns)
    .head(15)
)


print("\nSAVED FILE")
print(output_path)