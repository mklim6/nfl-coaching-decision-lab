from pathlib import Path

import nflreadpy as nfl
import polars as pl


# Seasons included in the project
seasons = list(range(2018, 2026))


# Project paths
project_root = Path(__file__).resolve().parents[1]

processed_data_dir = (
    project_root
    / "data"
    / "processed"
)

combined_output_path = (
    processed_data_dir
    / "play_calls_2018_2025.parquet"
)

coach_overrides_path = (
    project_root
    / "data"
    / "reference"
    / "head_coach_overrides.csv"
)


# Make sure the processed-data directory exists
processed_data_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# Load verified head-coach corrections
coach_overrides = pl.read_csv(
    coach_overrides_path
)


# Columns retained in every processed season
selected_columns = [
    # Identifiers
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "game_date",

    # Teams and coach
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "head_coach",

    # Pre-snap situation
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

    # Pre-game context
    "home_opening_kickoff",
    "roof",
    "surface",
    "temp",
    "wind",
    "spread_line",
    "total_line",

    # Prediction target
    "play_call",

    # Outcomes, not model inputs
    "epa",
    "success",
    "yards_gained",

    # nflverse benchmark, not model inputs
    "xpass",
    "pass_oe",

    # Audit fields
    "play_type",
    "qb_dropback",
    "rush_attempt",
    "qb_scramble",
    "sack",
]


def apply_head_coach_overrides(
    play_data,
    season,
):
    """
    Apply verified in-season head-coach changes.

    nflverse remains the default source. The reference table
    overrides only team-week combinations with a verified
    coaching change that nflverse did not capture.
    """

    season_overrides = coach_overrides.filter(
        pl.col("season") == season
    )

    for override in season_overrides.iter_rows(
        named=True
    ):
        play_data = play_data.with_columns(
            pl.when(
                (
                    pl.col("posteam")
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

    return play_data


def build_season_dataset(season):
    """
    Load, clean, and return one season of play-call data.
    """

    print(
        f"\nLoading {season} play-by-play data..."
    )

    pbp = nfl.load_pbp(
        seasons=[season]
    )

    # Assign the offensive team's head coach using the
    # nflverse home-coach and away-coach fields.
    pbp = pbp.with_columns(
        pl.when(
            pl.col("posteam")
            == pl.col("home_team")
        )
        .then(
            pl.col("home_coach")
        )
        .when(
            pl.col("posteam")
            == pl.col("away_team")
        )
        .then(
            pl.col("away_coach")
        )
        .otherwise(None)
        .alias("head_coach")
    )

    # Correct verified in-season coaching changes that
    # are absent from the nflverse coach fields.
    pbp = apply_head_coach_overrides(
        pbp,
        season,
    )

    # Define the intended play call.
    #
    # QB dropbacks include passes, sacks, and scrambles
    # that began as passing plays. Remaining rushing
    # attempts are treated as runs.
    pbp = pbp.with_columns(
        pl.when(
            pl.col("qb_dropback") == 1
        )
        .then(
            pl.lit("pass")
        )
        .when(
            pl.col("rush_attempt") == 1
        )
        .then(
            pl.lit("run")
        )
        .otherwise(None)
        .alias("play_call")
    )

    # Remove plays that do not represent a competitive
    # run/pass decision.
    play_calls = pbp.filter(
        pl.col("play_call").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("down").is_not_null()
        & (
            pl.col("qb_kneel").fill_null(0)
            == 0
        )
        & (
            pl.col("qb_spike").fill_null(0)
            == 0
        )
        & (
            pl.col("aborted_play").fill_null(0)
            == 0
        )
    )

    # Retain only the project columns.
    play_calls = play_calls.select(
        selected_columns
    )

    return play_calls


# Build and save each season separately
season_datasets = []

for season in seasons:
    season_play_calls = build_season_dataset(
        season
    )

    season_output_path = (
        processed_data_dir
        / f"play_calls_{season}.parquet"
    )

    season_play_calls.write_parquet(
        season_output_path
    )

    season_datasets.append(
        season_play_calls
    )

    print(
        f"{season} complete: "
        f"{season_play_calls.height:,} plays"
    )
    print(
        f"Saved to: {season_output_path}"
    )


# Combine all seasons into one dataset
all_play_calls = pl.concat(
    season_datasets,
    how="vertical_relaxed",
)

all_play_calls.write_parquet(
    combined_output_path
)


# Overall validation output
print("\nCOMBINED PLAY-CALL DATASET CREATED")
print(
    "Shape:",
    all_play_calls.shape,
)
print(
    "Saved to:",
    combined_output_path,
)


print("\nPLAY COUNTS BY SEASON")
print(
    all_play_calls
    .group_by("season")
    .len()
    .sort("season")
)


print("\nPLAY-CALL COUNTS BY SEASON")
print(
    all_play_calls
    .group_by([
        "season",
        "play_call",
    ])
    .len()
    .sort([
        "season",
        "play_call",
    ])
)


print("\nSEASON-TYPE COUNTS")
print(
    all_play_calls
    .group_by([
        "season",
        "season_type",
    ])
    .len()
    .sort([
        "season",
        "season_type",
    ])
)


print("\nCOACH COVERAGE BY SEASON")
print(
    all_play_calls
    .group_by("season")
    .agg([
        pl.len().alias(
            "total_plays"
        ),
        pl.col("head_coach")
        .is_not_null()
        .sum()
        .alias(
            "plays_with_head_coach"
        ),
        pl.col("head_coach")
        .n_unique()
        .alias(
            "unique_head_coaches"
        ),
    ])
    .sort("season")
)


print("\nCOACH OVERRIDE VALIDATION")
print(
    all_play_calls
    .filter(
        (
            (pl.col("season") == 2024)
            & pl.col("posteam").is_in([
                "CHI",
                "NO",
                "NYJ",
            ])
        )
        | (
            (pl.col("season") == 2025)
            & pl.col("posteam").is_in([
                "NYG",
                "TEN",
            ])
        )
    )
    .group_by([
        "season",
        "posteam",
        "head_coach",
    ])
    .agg([
        pl.len().alias("plays"),
        pl.col("game_id")
        .n_unique()
        .alias("games"),
        pl.col("week")
        .min()
        .alias("first_week"),
        pl.col("week")
        .max()
        .alias("last_week"),
    ])
    .sort([
        "season",
        "posteam",
        "first_week",
    ])
)


print("\nXPASS COVERAGE BY SEASON")
print(
    all_play_calls
    .group_by("season")
    .agg([
        pl.len().alias(
            "total_plays"
        ),
        pl.col("xpass")
        .is_not_null()
        .sum()
        .alias(
            "plays_with_xpass"
        ),
        (
            100
            * pl.col("xpass")
            .is_not_null()
            .sum()
            / pl.len()
        )
        .round(2)
        .alias(
            "xpass_coverage_percentage"
        ),
    ])
    .sort("season")
)


print("\nTARGET AUDIT")
print(
    all_play_calls
    .group_by([
        "season",
        "play_call",
        "qb_scramble",
        "sack",
    ])
    .len()
    .sort([
        "season",
        "play_call",
        "qb_scramble",
        "sack",
    ])
)