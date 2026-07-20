from pathlib import Path

import polars as pl


pl.Config.set_tbl_rows(100)

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / "play_calls_2018_2025.parquet"
)

play_calls = pl.read_parquet(data_path)


model_features = [
    "week",
    "season_type",
    "posteam",
    "defteam",
    "head_coach",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "quarter_seconds_remaining",
    "goal_to_go",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "shotgun",
    "no_huddle",
    "home_opening_kickoff",
    "roof",
    "surface",
    "temp",
    "wind",
    "spread_line",
    "total_line",
]


print("DATASET SUMMARY")
print(
    play_calls.select([
        pl.len().alias("total_plays"),
        pl.col("game_id").n_unique().alias("unique_games"),
        pl.col("season").min().alias("first_season"),
        pl.col("season").max().alias("last_season"),
    ])
)


print("\nDUPLICATE PLAY CHECK")
print(
    play_calls
    .group_by([
        "game_id",
        "play_id",
    ])
    .len()
    .filter(pl.col("len") > 1)
    .select(
        pl.len().alias("duplicate_play_ids")
    )
)


print("\nFEATURE MISSINGNESS BY SEASON")

missingness_rows = []

for season in sorted(
    play_calls["season"].unique().to_list()
):
    season_data = play_calls.filter(
        pl.col("season") == season
    )

    for feature in model_features:
        null_count = season_data[feature].null_count()
        null_percentage = (
            100 * null_count / season_data.height
        )

        missingness_rows.append({
            "season": season,
            "feature": feature,
            "null_count": null_count,
            "null_percentage": round(
                null_percentage,
                2,
            ),
        })

missingness_table = pl.DataFrame(
    missingness_rows
)

print(
    missingness_table
    .filter(pl.col("null_count") > 0)
    .sort([
        "season",
        "null_percentage",
    ], descending=[False, True])
)


print("\nWEATHER MISSINGNESS BY SEASON AND ROOF")
print(
    play_calls
    .with_columns(
        (
            pl.col("temp").is_null()
            | pl.col("wind").is_null()
        ).alias("weather_missing")
    )
    .group_by([
        "season",
        "roof",
        "weather_missing",
    ])
    .len()
    .sort([
        "season",
        "roof",
        "weather_missing",
    ])
)


print("\nCOACH-TEAM-SEASON ASSIGNMENTS")
print(
    play_calls
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
    ])
    .sort([
        "season",
        "posteam",
        "games",
    ], descending=[False, False, True])
)


print("\nTEAMS WITH MULTIPLE COACHES IN ONE SEASON")
print(
    play_calls
    .group_by([
        "season",
        "posteam",
    ])
    .agg([
        pl.col("head_coach")
        .n_unique()
        .alias("coach_count"),
        pl.col("head_coach")
        .unique()
        .sort()
        .alias("coaches"),
    ])
    .filter(pl.col("coach_count") > 1)
    .sort([
        "season",
        "posteam",
    ])
)


print("\nCOACHES ASSOCIATED WITH MULTIPLE TEAMS IN ONE SEASON")
print(
    play_calls
    .group_by([
        "season",
        "head_coach",
    ])
    .agg([
        pl.col("posteam")
        .n_unique()
        .alias("team_count"),
        pl.col("posteam")
        .unique()
        .sort()
        .alias("teams"),
    ])
    .filter(pl.col("team_count") > 1)
    .sort([
        "season",
        "head_coach",
    ])
)


print("\nTARGET BALANCE BY SEASON")
print(
    play_calls
    .group_by([
        "season",
        "play_call",
    ])
    .agg(
        pl.len().alias("plays")
    )
    .with_columns(
        (
            100
            * pl.col("plays")
            / pl.col("plays").sum().over("season")
        )
        .round(2)
        .alias("percentage")
    )
    .sort([
        "season",
        "play_call",
    ])
)


print("\nXPASS COVERAGE BY SEASON")
print(
    play_calls
    .group_by("season")
    .agg([
        pl.len().alias("total_plays"),
        pl.col("xpass")
        .is_not_null()
        .sum()
        .alias("plays_with_xpass"),
        (
            100
            * pl.col("xpass").is_not_null().sum()
            / pl.len()
        )
        .round(2)
        .alias("xpass_coverage_percentage"),
    ])
    .sort("season")
)


print("\nCHRONOLOGICAL SPLIT COUNTS")
print(
    play_calls
    .with_columns(
        pl.when(pl.col("season") <= 2023)
        .then(pl.lit("training"))
        .when(pl.col("season") == 2024)
        .then(pl.lit("validation"))
        .when(pl.col("season") == 2025)
        .then(pl.lit("test"))
        .otherwise(pl.lit("unused"))
        .alias("data_split")
    )
    .group_by("data_split")
    .len()
    .sort("data_split")
)