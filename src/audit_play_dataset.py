from pathlib import Path

import polars as pl


pl.Config.set_tbl_rows(30)

project_root = Path(__file__).resolve().parents[1]
data_path = (
    project_root
    / "data"
    / "processed"
    / "play_calls_2025.parquet"
)

play_calls = pl.read_parquet(data_path)


# Potential model features
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


# Calculate missingness and unique-value counts
audit_rows = []

for feature in model_features:
    null_count = play_calls[feature].null_count()
    null_percentage = 100 * null_count / play_calls.height

    audit_rows.append({
        "feature": feature,
        "null_count": null_count,
        "null_percentage": round(null_percentage, 2),
        "unique_values": play_calls[feature].n_unique(),
    })


audit_table = pl.DataFrame(audit_rows)

print("FEATURE AUDIT")
print(
    audit_table.sort(
        "null_percentage",
        descending=True,
    )
)


# Examine whether missing weather values are related to roof type
weather_missingness = (
    play_calls
    .with_columns(
        (
            pl.col("temp").is_null()
            | pl.col("wind").is_null()
        ).alias("weather_missing")
    )
    .group_by([
        "roof",
        "weather_missing",
    ])
    .agg(
        pl.len().alias("plays")
    )
    .sort([
        "roof",
        "weather_missing",
    ])
)

print("\nWEATHER MISSINGNESS BY ROOF")
print(weather_missingness)


print("\nCOACH-TEAM ASSIGNMENTS")
print(
    play_calls
    .group_by([
        "head_coach",
        "posteam",
    ])
    .len()
    .sort([
        "head_coach",
        "posteam",
    ])
)


print("\nCOACHES ASSOCIATED WITH MULTIPLE TEAMS")
print(
    play_calls
    .group_by("head_coach")
    .agg(
        pl.col("posteam").n_unique().alias("team_count"),
        pl.col("posteam").unique().alias("teams"),
    )
    .filter(pl.col("team_count") > 1)
)


print("\nNFLVERSE XPASS COVERAGE")
print(
    play_calls.select([
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
)


print("\nTARGET BALANCE")
print(
    play_calls
    .group_by("play_call")
    .agg([
        pl.len().alias("plays"),
        (
            100
            * pl.len()
            / play_calls.height
        )
        .round(2)
        .alias("percentage"),
    ])
    .sort("play_call")
)