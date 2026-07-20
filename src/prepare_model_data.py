from pathlib import Path

import polars as pl


pl.Config.set_tbl_rows(50)


# Project paths
project_root = Path(__file__).resolve().parents[1]

input_path = (
    project_root
    / "data"
    / "processed"
    / "play_calls_2018_2025.parquet"
)

processed_data_dir = (
    project_root
    / "data"
    / "processed"
)

train_output_path = (
    processed_data_dir
    / "model_train_2018_2023.parquet"
)

validation_output_path = (
    processed_data_dir
    / "model_validation_2024.parquet"
)

test_output_path = (
    processed_data_dir
    / "model_test_2025.parquet"
)


# Load the audited play-call dataset
play_calls = pl.read_parquet(
    input_path
)


# Create model features
model_data = play_calls.with_columns([
    # Binary prediction target
    pl.when(pl.col("play_call") == "pass")
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .cast(pl.Int8)
    .alias("is_pass"),

    # Whether the offense is the home team
    (
        pl.col("posteam")
        == pl.col("home_team")
    )
    .cast(pl.Int8)
    .alias("posteam_is_home"),

    # Whether the offensive team received the opening kickoff
    pl.when(
        pl.col("posteam")
        == pl.col("home_team")
    )
    .then(
        pl.col("home_opening_kickoff")
    )
    .otherwise(
        1 - pl.col("home_opening_kickoff")
    )
    .cast(pl.Int8)
    .alias("posteam_opening_kickoff"),

    # Express the point spread from the offensive team's
    # perspective instead of always the home team's perspective
    pl.when(
        pl.col("posteam")
        == pl.col("home_team")
    )
    .then(
        pl.col("spread_line")
    )
    .otherwise(
        -pl.col("spread_line")
    )
    .alias("posteam_spread_line"),
])


# Identifiers and context retained for evaluation,
# aggregation, and coach-level analysis
identifier_columns = [
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "game_date",
    "posteam",
    "defteam",
    "head_coach",
    "play_call",
]

# Columns retained for benchmarking and outcome analysis,
# but never used as model inputs
evaluation_columns = [
    "xpass",
    "pass_oe",
    "epa",
    "success",
    "yards_gained",
]

# Situation-only baseline features
#
# These variables describe the game situation without
# directly identifying the coach or offensive team.
model_features = [
    # Game timing
    "week",
    "season_type",
    "qtr",
    "game_seconds_remaining",

    # Down and field position
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",

    # Score and timeout situation
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",

    # Home and kickoff context
    "posteam_is_home",
    "posteam_opening_kickoff",

    # Stadium and pre-game context
    "roof",
    "surface",
    "posteam_spread_line",
    "total_line",
]

# Target column
target_column = [
    "is_pass",
]

# Keep only identifiers, model inputs, and target
model_data = model_data.select(
    identifier_columns
    + [
        feature
        for feature in model_features
        if feature not in identifier_columns
    ]
    + evaluation_columns
    + target_column
)

# Add the chronological split label
model_data = model_data.with_columns(
    pl.when(
        pl.col("season") <= 2023
    )
    .then(
        pl.lit("training")
    )
    .when(
        pl.col("season") == 2024
    )
    .then(
        pl.lit("validation")
    )
    .when(
        pl.col("season") == 2025
    )
    .then(
        pl.lit("test")
    )
    .otherwise(
        pl.lit("unused")
    )
    .alias("data_split")
)


# Create chronological datasets
train_data = model_data.filter(
    pl.col("data_split") == "training"
)

validation_data = model_data.filter(
    pl.col("data_split") == "validation"
)

test_data = model_data.filter(
    pl.col("data_split") == "test"
)


# Confirm no model input contains missing values
missingness_rows = []

for feature in model_features:
    missingness_rows.append({
        "feature": feature,
        "training_nulls": (
            train_data[feature].null_count()
        ),
        "validation_nulls": (
            validation_data[feature].null_count()
        ),
        "test_nulls": (
            test_data[feature].null_count()
        ),
    })

missingness_table = pl.DataFrame(
    missingness_rows
)

total_missing_values = (
    missingness_table.select(
        pl.exclude("feature").sum()
    )
    .sum_horizontal()
    .item()
)

if total_missing_values > 0:
    print("\nMODEL FEATURE MISSINGNESS")
    print(
        missingness_table.filter(
            (
                pl.col("training_nulls") > 0
            )
            | (
                pl.col("validation_nulls") > 0
            )
            | (
                pl.col("test_nulls") > 0
            )
        )
    )

    raise ValueError(
        "Missing values were found in model features."
    )


# Check for overlap between chronological splits
train_games = set(
    train_data["game_id"].unique().to_list()
)

validation_games = set(
    validation_data["game_id"].unique().to_list()
)

test_games = set(
    test_data["game_id"].unique().to_list()
)

if train_games & validation_games:
    raise ValueError(
        "Training and validation games overlap."
    )

if train_games & test_games:
    raise ValueError(
        "Training and test games overlap."
    )

if validation_games & test_games:
    raise ValueError(
        "Validation and test games overlap."
    )


# Save the model datasets
train_data.write_parquet(
    train_output_path
)

validation_data.write_parquet(
    validation_output_path
)

test_data.write_parquet(
    test_output_path
)


# Validation output
print("MODEL DATASETS CREATED")

print("\nMODEL FEATURES")
for feature in model_features:
    print(f"- {feature}")


print("\nSPLIT COUNTS")
print(
    model_data
    .group_by("data_split")
    .agg([
        pl.len().alias("plays"),
        pl.col("game_id")
        .n_unique()
        .alias("games"),
        pl.col("season")
        .min()
        .alias("first_season"),
        pl.col("season")
        .max()
        .alias("last_season"),
    ])
    .sort("first_season")
)


print("\nTARGET BALANCE BY SPLIT")
print(
    model_data
    .group_by([
        "data_split",
        "is_pass",
    ])
    .agg(
        pl.len().alias("plays")
    )
    .with_columns(
        (
            100
            * pl.col("plays")
            / pl.col("plays")
            .sum()
            .over("data_split")
        )
        .round(2)
        .alias("percentage")
    )
    .sort([
        "data_split",
        "is_pass",
    ])
)


print("\nMODEL FEATURE MISSINGNESS")
print(missingness_table)


print("\nSPLIT OVERLAP CHECK")
print("Training-validation overlap: 0 games")
print("Training-test overlap: 0 games")
print("Validation-test overlap: 0 games")


print("\nSAVED FILES")
print("Training:", train_output_path)
print("Validation:", validation_output_path)
print("Test:", test_output_path)
