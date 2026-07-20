from pathlib import Path

import nflreadpy as nfl
import polars as pl


pl.Config.set_tbl_rows(50)

seasons = list(
    range(2018, 2026)
)


# Project paths
project_root = Path(__file__).resolve().parents[1]

coach_overrides_path = (
    project_root
    / "data"
    / "reference"
    / "head_coach_overrides.csv"
)

team_game_output_path = (
    project_root
    / "data"
    / "processed"
    / "team_game_results_2018_2025.parquet"
)

team_season_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "team_season_records_2018_2025.csv"
)

coach_season_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_team_season_records_2018_2025.csv"
)

coach_season_type_output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_team_season_type_records_2018_2025.csv"
)


# Confirm reference table exists
if not coach_overrides_path.exists():
    raise FileNotFoundError(
        "Head-coach override file was not found: "
        f"{coach_overrides_path}"
    )


# Create output directories
team_game_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

team_season_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# Load verified coach corrections
coach_overrides = pl.read_csv(
    coach_overrides_path
)


print("LOADING NFL SCHEDULES")

schedules = nfl.load_schedules(
    seasons=seasons
)

# Normalize historical team abbreviations to match
# the identifiers used in the play-prediction dataset.
team_code_map = {
    "OAK": "LV",
}

schedules = schedules.with_columns([
    pl.col("home_team")
    .replace(team_code_map)
    .alias("home_team"),

    pl.col("away_team")
    .replace(team_code_map)
    .alias("away_team"),
])

# Keep only regular-season and postseason games
valid_game_types = [
    "REG",
    "WC",
    "DIV",
    "CON",
    "SB",
]

schedules = schedules.filter(
    pl.col("game_type").is_in(
        valid_game_types
    )
)


# Keep only completed games
schedules = schedules.filter(
    pl.col("home_score").is_not_null()
    & pl.col("away_score").is_not_null()
)


print(
    f"Completed games loaded: "
    f"{schedules.height:,}"
)


# Create one row for the home team
home_results = schedules.select([
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",

    pl.col("home_team").alias(
        "team"
    ),

    pl.col("away_team").alias(
        "opponent"
    ),

    pl.lit(1)
    .cast(pl.Int8)
    .alias("is_home"),

    pl.col("home_score").alias(
        "points_for"
    ),

    pl.col("away_score").alias(
        "points_against"
    ),

    pl.col("home_coach").alias(
        "head_coach"
    ),
])


# Create one row for the away team
away_results = schedules.select([
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",

    pl.col("away_team").alias(
        "team"
    ),

    pl.col("home_team").alias(
        "opponent"
    ),

    pl.lit(0)
    .cast(pl.Int8)
    .alias("is_home"),

    pl.col("away_score").alias(
        "points_for"
    ),

    pl.col("home_score").alias(
        "points_against"
    ),

    pl.col("away_coach").alias(
        "head_coach"
    ),
])


# Combine home and away team perspectives
team_games = pl.concat(
    [
        home_results,
        away_results,
    ],
    how="vertical_relaxed",
)


# Apply verified head-coach corrections
for override in coach_overrides.iter_rows(
    named=True
):
    team_games = team_games.with_columns(
        pl.when(
            (
                pl.col("season")
                == override["season"]
            )
            & (
                pl.col("team")
                == override["posteam"]
            )
            & (
                pl.col("week")
                >= override["start_week"]
            )
            & (
                pl.col("week")
                <= override["end_week"]
            )
        )
        .then(
            pl.lit(
                override["head_coach"]
            )
        )
        .otherwise(
            pl.col("head_coach")
        )
        .alias("head_coach")
    )


# Add season type and game outcomes
team_games = (
    team_games
    .with_columns([
        pl.when(
            pl.col("game_type") == "REG"
        )
        .then(
            pl.lit("REG")
        )
        .otherwise(
            pl.lit("POST")
        )
        .alias("season_type"),

        (
            pl.col("points_for")
            - pl.col("points_against")
        ).alias("point_differential"),
    ])
    .with_columns([
        pl.when(
            pl.col("points_for")
            > pl.col("points_against")
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("win"),

        pl.when(
            pl.col("points_for")
            < pl.col("points_against")
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("loss"),

        pl.when(
            pl.col("points_for")
            == pl.col("points_against")
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("tie"),

        pl.when(
            pl.col("points_for")
            > pl.col("points_against")
        )
        .then(pl.lit("W"))
        .when(
            pl.col("points_for")
            < pl.col("points_against")
        )
        .then(pl.lit("L"))
        .otherwise(pl.lit("T"))
        .alias("game_result"),
    ])
    .sort([
        "season",
        "week",
        "game_id",
        "team",
    ])
)


def build_record_table(
    data,
    group_columns,
):
    """Create team or coach record summaries."""

    return (
        data
        .group_by(group_columns)
        .agg([
            pl.len().alias("games"),

            pl.col("win")
            .sum()
            .alias("wins"),

            pl.col("loss")
            .sum()
            .alias("losses"),

            pl.col("tie")
            .sum()
            .alias("ties"),

            pl.col("points_for")
            .sum()
            .alias("points_for"),

            pl.col("points_against")
            .sum()
            .alias("points_against"),

            pl.col("point_differential")
            .sum()
            .alias(
                "point_differential"
            ),

            pl.col("points_for")
            .mean()
            .alias(
                "points_per_game"
            ),

            pl.col("points_against")
            .mean()
            .alias(
                "points_allowed_per_game"
            ),

            pl.col("point_differential")
            .mean()
            .alias(
                "point_differential_per_game"
            ),
        ])
        .with_columns([
            (
                (
                    pl.col("wins")
                    + 0.5 * pl.col("ties")
                )
                / pl.col("games")
            )
            .round(3)
            .alias("win_percentage"),

            pl.col("points_per_game")
            .round(2),

            pl.col(
                "points_allowed_per_game"
            )
            .round(2),

            pl.col(
                "point_differential_per_game"
            )
            .round(2),
        ])
        .sort(group_columns)
    )


# Team-season records
team_season_records = (
    build_record_table(
        team_games,
        [
            "season",
            "team",
        ],
    )
)


# Coach-team-season records across all games
coach_season_records = (
    build_record_table(
        team_games,
        [
            "season",
            "head_coach",
            "team",
        ],
    )
)


# Coach-team-season records separated by season type
coach_season_type_records = (
    build_record_table(
        team_games,
        [
            "season",
            "season_type",
            "head_coach",
            "team",
        ],
    )
)


# Validate unique team-game rows
duplicate_team_games = (
    team_games
    .group_by([
        "game_id",
        "team",
    ])
    .len()
    .filter(
        pl.col("len") > 1
    )
    .height
)


missing_coaches = (
    team_games[
        "head_coach"
    ].null_count()
)


# Save outputs
team_games.write_parquet(
    team_game_output_path
)

team_season_records.write_csv(
    team_season_output_path
)

coach_season_records.write_csv(
    coach_season_output_path
)

coach_season_type_records.write_csv(
    coach_season_type_output_path
)


# Validation output
print("\nTEAM RESULTS CREATED")


print("\nTEAM-GAME SUMMARY")
print(
    team_games.select([
        pl.len().alias(
            "team_game_rows"
        ),
        pl.col("game_id")
        .n_unique()
        .alias(
            "unique_games"
        ),
        pl.col("team")
        .n_unique()
        .alias(
            "unique_teams"
        ),
        pl.col("season")
        .min()
        .alias(
            "first_season"
        ),
        pl.col("season")
        .max()
        .alias(
            "last_season"
        ),
    ])
)


print("\nVALIDATION")
print(
    f"Duplicate team-game rows: "
    f"{duplicate_team_games}"
)
print(
    f"Missing head coaches: "
    f"{missing_coaches}"
)


print("\nGAMES BY SEASON TYPE")
print(
    team_games
    .group_by("season_type")
    .agg(
        pl.col("game_id")
        .n_unique()
        .alias("games")
    )
    .sort("season_type")
)


print("\nCOACH OVERRIDE VALIDATION")
print(
    team_games
    .filter(
        (
            (pl.col("season") == 2024)
            & pl.col("team").is_in([
                "CHI",
                "NO",
                "NYJ",
            ])
        )
        | (
            (pl.col("season") == 2025)
            & pl.col("team").is_in([
                "NYG",
                "TEN",
            ])
        )
    )
    .group_by([
        "season",
        "team",
        "head_coach",
    ])
    .agg([
        pl.len().alias("games"),
        pl.col("week")
        .min()
        .alias("first_week"),
        pl.col("week")
        .max()
        .alias("last_week"),
        pl.col("win")
        .sum()
        .alias("wins"),
        pl.col("loss")
        .sum()
        .alias("losses"),
        pl.col("tie")
        .sum()
        .alias("ties"),
    ])
    .sort([
        "season",
        "team",
        "first_week",
    ])
)


print("\n2025 TEAM RECORDS")
print(
    team_season_records
    .filter(
        pl.col("season") == 2025
    )
    .sort(
        "win_percentage",
        descending=True,
    )
)


print("\nSAVED FILES")
print(
    "Team-game results:",
    team_game_output_path,
)
print(
    "Team-season records:",
    team_season_output_path,
)
print(
    "Coach-season records:",
    coach_season_output_path,
)
print(
    "Coach season-type records:",
    coach_season_type_output_path,
)