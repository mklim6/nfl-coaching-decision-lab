from pathlib import Path

import polars as pl


pl.Config.set_tbl_rows(50)


# Project paths
project_root = Path(__file__).resolve().parents[1]

team_results_path = (
    project_root
    / "data"
    / "processed"
    / "team_game_results_2018_2025.parquet"
)

output_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_tenures_2018_2025.csv"
)


if not team_results_path.exists():
    raise FileNotFoundError(
        "Team-game results were not found: "
        f"{team_results_path}"
    )


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


print("LOADING TEAM-GAME RESULTS")

team_games = pl.read_parquet(
    team_results_path
)


# Determine each team's final game week in each season
team_season_bounds = (
    team_games
    .group_by([
        "season",
        "team",
    ])
    .agg([
        pl.col("week")
        .min()
        .alias("team_first_week"),

        pl.col("week")
        .max()
        .alias("team_last_week"),

        pl.len()
        .alias("team_games"),
    ])
)


# Create one row per coach-team-season tenure
tenures = (
    team_games
    .group_by([
        "season",
        "head_coach",
        "team",
    ])
    .agg([
        pl.col("week")
        .min()
        .alias("first_week"),

        pl.col("week")
        .max()
        .alias("last_week"),

        pl.len()
        .alias("games"),

        pl.col("game_id")
        .n_unique()
        .alias("unique_games"),

        (
            pl.col("season_type") == "REG"
        )
        .sum()
        .alias("regular_season_games"),

        (
            pl.col("season_type") == "POST"
        )
        .sum()
        .alias("postseason_games"),

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
        .alias("point_differential"),
    ])
    .join(
        team_season_bounds,
        on=[
            "season",
            "team",
        ],
        how="left",
    )
    .with_columns([
        (
            pl.col("first_week")
            > pl.col("team_first_week")
        )
        .alias("started_midseason"),

        (
            pl.col("last_week")
            < pl.col("team_last_week")
        )
        .alias("ended_before_team_season"),

        (
            (
                pl.col("wins")
                + 0.5 * pl.col("ties")
            )
            / pl.col("games")
        )
        .round(3)
        .alias("tenure_win_percentage"),

        (
            pl.col("points_for")
            / pl.col("games")
        )
        .round(2)
        .alias("points_per_game"),

        (
            pl.col("points_against")
            / pl.col("games")
        )
        .round(2)
        .alias("points_allowed_per_game"),

        (
            pl.col("point_differential")
            / pl.col("games")
        )
        .round(2)
        .alias("point_differential_per_game"),
    ])
    .sort([
        "season",
        "team",
        "first_week",
        "head_coach",
    ])
)


# Calculate experience entering each season using only
# seasons contained in the 2018-2025 project window.
tenure_rows = tenures.to_dicts()
experience_rows = []

coach_history = {}

for row in tenure_rows:
    coach = row["head_coach"]
    season = row["season"]

    prior_rows = [
        prior
        for prior in coach_history.get(coach, [])
        if prior["season"] < season
    ]

    prior_seasons = sorted({
        prior["season"]
        for prior in prior_rows
    })

    prior_teams = sorted({
        prior["team"]
        for prior in prior_rows
    })

    prior_games = sum(
        prior["games"]
        for prior in prior_rows
    )

    prior_wins = sum(
        prior["wins"]
        for prior in prior_rows
    )

    prior_losses = sum(
        prior["losses"]
        for prior in prior_rows
    )

    prior_ties = sum(
        prior["ties"]
        for prior in prior_rows
    )

    if prior_games > 0:
        prior_win_percentage = round(
            (
                prior_wins
                + 0.5 * prior_ties
            )
            / prior_games,
            3,
        )
    else:
        prior_win_percentage = None

    all_observed_rows = (
        coach_history.get(coach, [])
        + [row]
    )

    first_observed_season = min(
        observed["season"]
        for observed in all_observed_rows
    )

    experience_rows.append({
        **row,

        "observed_prior_hc_seasons": (
            len(prior_seasons)
        ),

        "observed_prior_hc_games": (
            prior_games
        ),

        "observed_prior_hc_wins": (
            prior_wins
        ),

        "observed_prior_hc_losses": (
            prior_losses
        ),

        "observed_prior_hc_ties": (
            prior_ties
        ),

        "observed_prior_hc_win_percentage": (
            prior_win_percentage
        ),

        "observed_prior_teams": (
            ", ".join(prior_teams)
            if prior_teams
            else None
        ),

                "first_observed_hc_season": (
            first_observed_season
        ),

        "observed_history_starts_at_window_boundary": (
            first_observed_season == 2018
        ),

        "experience_scope": (
            "Observed NFL head-coaching results since 2018 only"
        ),
    })

    coach_history.setdefault(
        coach,
        []
    ).append(row)


coach_tenures = (
    pl.DataFrame(experience_rows)
    .sort([
        "season",
        "team",
        "first_week",
        "head_coach",
    ])
)


# Validation
duplicate_tenures = (
    coach_tenures
    .group_by([
        "season",
        "head_coach",
        "team",
    ])
    .len()
    .filter(
        pl.col("len") > 1
    )
    .height
)

midseason_tenures = (
    coach_tenures
    .filter(
        pl.col("started_midseason")
    )
)

missing_coaches = (
    coach_tenures[
        "head_coach"
    ].null_count()
)


coach_tenures.write_csv(
    output_path
)


print("\nCOACH TENURE DATASET CREATED")

print("\nSUMMARY")
print(
    coach_tenures.select([
        pl.len().alias("tenures"),

        pl.col("head_coach")
        .n_unique()
        .alias("unique_coaches"),

        pl.col("season")
        .min()
        .alias("first_season"),

        pl.col("season")
        .max()
        .alias("last_season"),
    ])
)

print("\nVALIDATION")
print(
    f"Duplicate coach-team-season tenures: "
    f"{duplicate_tenures}"
)
print(
    f"Missing head coaches: "
    f"{missing_coaches}"
)
print(
    f"Midseason-start tenures: "
    f"{midseason_tenures.height}"
)

print("\nMIDSEASON-START TENURES")
print(
    midseason_tenures.select([
        "season",
        "team",
        "head_coach",
        "first_week",
        "last_week",
        "games",
        "wins",
        "losses",
        "ties",
    ])
)

print("\n2025 COACHING EXPERIENCE")
print(
    coach_tenures
    .filter(
        pl.col("season") == 2025
    )
    .select([
        "head_coach",
        "team",
        "observed_prior_hc_seasons",
        "observed_prior_hc_games",
        "observed_prior_hc_wins",
        "observed_prior_hc_losses",
        "observed_prior_hc_ties",
        "observed_history_starts_at_window_boundary",
        "experience_scope",
    ])
    .sort(
        "observed_prior_hc_games",
        descending=True,
    )
)

print("\nSAVED FILE")
print(output_path)