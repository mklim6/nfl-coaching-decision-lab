from pathlib import Path

# Database version 1.6 adds the play-style analysis layer.

import duckdb


# Project paths
project_root = Path(__file__).resolve().parents[1]

database_dir = (
    project_root
    / "database"
)

database_path = (
    database_dir
    / "coaching_lab.duckdb"
)

outputs_dir = (
    project_root
    / "outputs"
    / "tables"
)

processed_data_dir = (
    project_root
    / "data"
    / "processed"
)

reference_data_dir = (
    project_root
    / "data"
    / "reference"
)


# Source files
play_predictions_path = (
    outputs_dir
    / "historical_play_predictions_2018_2025.parquet"
)

coach_season_path = (
    outputs_dir
    / "coach_season_summary_2018_2025.csv"
)

coach_season_type_path = (
    outputs_dir
    / "coach_season_type_summary_2018_2025.csv"
)

coach_uncertainty_path = (
    outputs_dir
    / "coach_season_uncertainty_2018_2025.csv"
)

play_style_predictions_path = (
    outputs_dir
    / "play_style_predictions_2018_2025.parquet"
)

play_caller_uncertainty_path = (
    outputs_dir
    / "play_caller_season_uncertainty_2018_2025.csv"
)

season_metrics_path = (
    outputs_dir
    / "historical_model_metrics_by_season.csv"
)

final_test_metrics_path = (
    outputs_dir
    / "final_test_metrics_2025.csv"
)

team_game_results_path = (
    processed_data_dir
    / "team_game_results_2018_2025.parquet"
)

team_season_records_path = (
    outputs_dir
    / "team_season_records_2018_2025.csv"
)

coach_team_season_records_path = (
    outputs_dir
    / "coach_team_season_records_2018_2025.csv"
)

coach_team_season_type_records_path = (
    outputs_dir
    / "coach_team_season_type_records_2018_2025.csv"
)

coach_tenures_path = (
    outputs_dir
    / "coach_tenures_2018_2025.csv"
)

play_caller_reference_path = (
    reference_data_dir
    / "offensive_play_caller_tenures.csv"
)


# Confirm every required source file exists
required_files = [
    play_predictions_path,
    play_style_predictions_path,
    coach_season_path,
    coach_season_type_path,
    coach_uncertainty_path,
    play_caller_uncertainty_path,
    season_metrics_path,
    final_test_metrics_path,
    team_game_results_path,
    team_season_records_path,
    coach_team_season_records_path,
    coach_team_season_type_records_path,
    coach_tenures_path,
    play_caller_reference_path,
]

missing_files = [
    path
    for path in required_files
    if not path.exists()
]

if missing_files:
    missing_list = "\n".join(
        str(path)
        for path in missing_files
    )

    raise FileNotFoundError(
        "Required database source files "
        "were not found:\n"
        f"{missing_list}"
    )


# Create database directory
database_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# Connect to DuckDB
connection = duckdb.connect(
    str(database_path)
)


# Remove dependent views before replacing tables
views_to_drop = [
    "play_style_predictions_with_callers",
    "play_predictions_with_callers",
    "play_caller_season_summary",
    "play_caller_season_type_summary",
    "play_caller_career_summary",
    "play_caller_team_career_summary",
    "eligible_coach_seasons",
    "clear_coach_tendencies",
    "coach_career_summary",
    "coach_team_career_summary",
    "team_season_summary",
    "league_season_summary",
]

for view_name in views_to_drop:
    connection.execute(
        f"DROP VIEW IF EXISTS {view_name}"
    )


# Load source files into physical DuckDB tables
print("LOADING PLAY-LEVEL PREDICTIONS")

connection.execute(
    """
    CREATE OR REPLACE TABLE play_predictions AS
    SELECT *
    FROM read_parquet(?)
    """,
    [str(play_predictions_path)],
)


print("LOADING COACH-SEASON SUMMARIES")

connection.execute(
    """
    CREATE OR REPLACE TABLE coach_season_summary AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(coach_season_path)],
)


print("LOADING COACH SEASON-TYPE SUMMARIES")

connection.execute(
    """
    CREATE OR REPLACE TABLE
        coach_season_type_summary AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(coach_season_type_path)],
)


print("LOADING COACH UNCERTAINTY RESULTS")

connection.execute(
    """
    CREATE OR REPLACE TABLE coach_uncertainty AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(coach_uncertainty_path)],
)


print("LOADING PLAY-STYLE PREDICTIONS")

connection.execute(
    """
    CREATE OR REPLACE TABLE
        play_style_predictions AS
    SELECT *
    FROM read_parquet(?)
    """,
    [str(play_style_predictions_path)],
)


print("LOADING PLAY-CALLER UNCERTAINTY RESULTS")

connection.execute(
    """
    CREATE OR REPLACE TABLE
        play_caller_uncertainty AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(play_caller_uncertainty_path)],
)


print("LOADING HISTORICAL MODEL METRICS")

connection.execute(
    """
    CREATE OR REPLACE TABLE model_metrics_by_season AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(season_metrics_path)],
)


print("LOADING FINAL TEST METRICS")

connection.execute(
    """
    CREATE OR REPLACE TABLE final_test_metrics AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(final_test_metrics_path)],
)


print("LOADING TEAM-GAME RESULTS")

connection.execute(
    """
    CREATE OR REPLACE TABLE team_game_results AS
    SELECT *
    FROM read_parquet(?)
    """,
    [str(team_game_results_path)],
)


print("LOADING TEAM-SEASON RECORDS")

connection.execute(
    """
    CREATE OR REPLACE TABLE team_season_records AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(team_season_records_path)],
)


print("LOADING COACH-TEAM-SEASON RECORDS")

connection.execute(
    """
    CREATE OR REPLACE TABLE
        coach_team_season_records AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(coach_team_season_records_path)],
)


print("LOADING COACH TEAM-SEASON-TYPE RECORDS")

connection.execute(
    """
    CREATE OR REPLACE TABLE
        coach_team_season_type_records AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(coach_team_season_type_records_path)],
)


print("LOADING COACH TENURES")

connection.execute(
    """
    CREATE OR REPLACE TABLE coach_tenures AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(coach_tenures_path)],
)


print("LOADING OFFENSIVE PLAY-CALLER REFERENCE")

connection.execute(
    """
    CREATE OR REPLACE TABLE
        offensive_play_caller_tenures AS
    SELECT *
    FROM read_csv_auto(
        ?,
        header = true
    )
    """,
    [str(play_caller_reference_path)],
)


# Attach the verified play caller where one is available. The fallback remains
# explicit so the database will not silently present an unverified caller as
# verified if a future reference row is incomplete.
connection.execute(
    """
    CREATE VIEW play_predictions_with_callers AS
    SELECT
        p.*,
        r.tenure_id AS play_caller_tenure_id,
        r.offensive_play_caller,
        r.caller_role,
        r.head_coach_is_play_caller,
        r.verification_status AS
            play_caller_verification_status,
        r.source_title AS play_caller_source_title,
        r.source_publisher AS play_caller_source_publisher,
        r.source_url AS play_caller_source_url,
        r.source_date AS play_caller_source_date,
        r.date_verified AS play_caller_date_verified,
        r.verification_notes AS play_caller_notes,

        CASE
            WHEN
                r.verification_status = 'verified'
                AND r.offensive_play_caller IS NOT NULL
            THEN r.offensive_play_caller
            ELSE p.head_coach
        END AS attributed_coach,

        CASE
            WHEN
                r.verification_status = 'verified'
                AND r.offensive_play_caller IS NOT NULL
            THEN 'verified_offensive_play_caller'
            ELSE 'head_coach_fallback'
        END AS attribution_type

    FROM play_predictions AS p
    LEFT JOIN offensive_play_caller_tenures AS r
        ON p.season = r.season
        AND p.posteam = r.team
        AND p.head_coach = r.head_coach
        AND p.week BETWEEN
            r.start_week AND r.end_week
    """
)


# Combine the verified caller attribution with the enriched play-style fields.
connection.execute(
    """
    CREATE VIEW play_style_predictions_with_callers AS
    SELECT
        s.*,
        p.play_caller_tenure_id,
        p.offensive_play_caller,
        p.caller_role,
        p.head_coach_is_play_caller,
        p.play_caller_verification_status,
        p.play_caller_source_title,
        p.play_caller_source_publisher,
        p.play_caller_source_url,
        p.play_caller_source_date,
        p.play_caller_date_verified,
        p.play_caller_notes,
        p.attributed_coach,
        p.attribution_type
    FROM play_style_predictions AS s
    INNER JOIN play_predictions_with_callers AS p
        ON s.game_id = p.game_id
        AND s.play_id = p.play_id
        AND s.season = p.season
    """
)


# Verified offensive play-caller summaries. These views deliberately include
# only verified caller assignments. They do not reuse the head-coach bootstrap
# confidence intervals because those intervals were clustered and estimated
# for different attribution groups.
connection.execute(
    """
    CREATE VIEW play_caller_season_summary AS
    SELECT
        season,
        offensive_play_caller,
        posteam,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(DISTINCT head_coach) AS head_coaches,
        STRING_AGG(
            DISTINCT head_coach,
            ', '
            ORDER BY head_coach
        ) AS associated_head_coaches,

        ROUND(100 * AVG(is_pass), 2)
            AS actual_pass_rate_pct,
        ROUND(100 * AVG(expected_pass_probability), 2)
            AS expected_pass_rate_pct,
        ROUND(100 * AVG(model_pass_oe), 2)
            AS model_pass_oe_pct,
        ROUND(100 * AVG(pass_oe), 2)
            AS nflverse_pass_oe_pct,
        ROUND(AVG(epa), 4) AS mean_epa,
        ROUND(100 * AVG(success), 2)
            AS success_rate_pct,
        ROUND(AVG(yards_gained), 2)
            AS mean_yards_gained

    FROM play_predictions_with_callers
    WHERE
        attribution_type =
            'verified_offensive_play_caller'
    GROUP BY
        season,
        offensive_play_caller,
        posteam
    """
)


connection.execute(
    """
    CREATE VIEW play_caller_season_type_summary AS
    SELECT
        season,
        season_type,
        offensive_play_caller,
        posteam,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(DISTINCT head_coach) AS head_coaches,
        STRING_AGG(
            DISTINCT head_coach,
            ', '
            ORDER BY head_coach
        ) AS associated_head_coaches,

        ROUND(100 * AVG(is_pass), 2)
            AS actual_pass_rate_pct,
        ROUND(100 * AVG(expected_pass_probability), 2)
            AS expected_pass_rate_pct,
        ROUND(100 * AVG(model_pass_oe), 2)
            AS model_pass_oe_pct,
        ROUND(100 * AVG(pass_oe), 2)
            AS nflverse_pass_oe_pct,
        ROUND(AVG(epa), 4) AS mean_epa,
        ROUND(100 * AVG(success), 2)
            AS success_rate_pct,
        ROUND(AVG(yards_gained), 2)
            AS mean_yards_gained

    FROM play_predictions_with_callers
    WHERE
        attribution_type =
            'verified_offensive_play_caller'
    GROUP BY
        season,
        season_type,
        offensive_play_caller,
        posteam
    """
)


connection.execute(
    """
    CREATE VIEW play_caller_career_summary AS
    SELECT
        offensive_play_caller,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(DISTINCT season) AS seasons,
        COUNT(DISTINCT posteam) AS teams,
        COUNT(DISTINCT head_coach) AS head_coaches,
        MIN(season) AS first_season,
        MAX(season) AS last_season,

        ROUND(100 * AVG(is_pass), 2)
            AS actual_pass_rate_pct,
        ROUND(100 * AVG(expected_pass_probability), 2)
            AS expected_pass_rate_pct,
        ROUND(100 * AVG(model_pass_oe), 2)
            AS model_pass_oe_pct,
        ROUND(100 * AVG(pass_oe), 2)
            AS nflverse_pass_oe_pct,
        ROUND(AVG(epa), 4) AS mean_epa,
        ROUND(100 * AVG(success), 2)
            AS success_rate_pct,
        ROUND(AVG(yards_gained), 2)
            AS mean_yards_gained

    FROM play_predictions_with_callers
    WHERE
        attribution_type =
            'verified_offensive_play_caller'
    GROUP BY offensive_play_caller
    """
)


connection.execute(
    """
    CREATE VIEW play_caller_team_career_summary AS
    SELECT
        offensive_play_caller,
        posteam,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(DISTINCT season) AS seasons,
        COUNT(DISTINCT head_coach) AS head_coaches,
        MIN(season) AS first_season,
        MAX(season) AS last_season,

        ROUND(100 * AVG(is_pass), 2)
            AS actual_pass_rate_pct,
        ROUND(100 * AVG(expected_pass_probability), 2)
            AS expected_pass_rate_pct,
        ROUND(100 * AVG(model_pass_oe), 2)
            AS model_pass_oe_pct,
        ROUND(100 * AVG(pass_oe), 2)
            AS nflverse_pass_oe_pct,
        ROUND(AVG(epa), 4) AS mean_epa,
        ROUND(100 * AVG(success), 2)
            AS success_rate_pct,
        ROUND(AVG(yards_gained), 2)
            AS mean_yards_gained

    FROM play_predictions_with_callers
    WHERE
        attribution_type =
            'verified_offensive_play_caller'
    GROUP BY
        offensive_play_caller,
        posteam
    """
)


# Database metadata
connection.execute(
    """
    CREATE OR REPLACE TABLE database_metadata AS
    SELECT
        CURRENT_TIMESTAMP AS built_at,
        2018 AS first_season,
        2025 AS last_season,
        500 AS minimum_coach_plays,
        '1.6' AS database_version
    """
)


# Useful filtered views
connection.execute(
    """
    CREATE VIEW eligible_coach_seasons AS
    SELECT *
    FROM coach_uncertainty
    WHERE meets_minimum_sample = true
    """
)


connection.execute(
    """
    CREATE VIEW clear_coach_tendencies AS
    SELECT *
    FROM coach_uncertainty
    WHERE
        meets_minimum_sample = true
        AND ci_excludes_zero = true
    """
)


# Multi-season coach summaries
connection.execute(
    """
    CREATE VIEW coach_career_summary AS
    SELECT
        head_coach,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(DISTINCT season) AS seasons,
        COUNT(DISTINCT posteam) AS teams,
        MIN(season) AS first_season,
        MAX(season) AS last_season,

        ROUND(
            100 * AVG(is_pass),
            2
        ) AS actual_pass_rate_pct,

        ROUND(
            100 * AVG(
                expected_pass_probability
            ),
            2
        ) AS expected_pass_rate_pct,

        ROUND(
            100 * AVG(model_pass_oe),
            2
        ) AS model_pass_oe_pct,

        ROUND(
            100 * AVG(pass_oe),
            2
        ) AS nflverse_pass_oe_pct,

        ROUND(
            AVG(epa),
            4
        ) AS mean_epa,

        ROUND(
            100 * AVG(success),
            2
        ) AS success_rate_pct,

        ROUND(
            AVG(yards_gained),
            2
        ) AS mean_yards_gained

    FROM play_predictions
    GROUP BY head_coach
    """
)


# Coach-team career summaries
connection.execute(
    """
    CREATE VIEW coach_team_career_summary AS
    SELECT
        head_coach,
        posteam,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(DISTINCT season) AS seasons,
        MIN(season) AS first_season,
        MAX(season) AS last_season,

        ROUND(
            100 * AVG(is_pass),
            2
        ) AS actual_pass_rate_pct,

        ROUND(
            100 * AVG(
                expected_pass_probability
            ),
            2
        ) AS expected_pass_rate_pct,

        ROUND(
            100 * AVG(model_pass_oe),
            2
        ) AS model_pass_oe_pct,

        ROUND(
            AVG(epa),
            4
        ) AS mean_epa,

        ROUND(
            100 * AVG(success),
            2
        ) AS success_rate_pct,

        ROUND(
            AVG(yards_gained),
            2
        ) AS mean_yards_gained

    FROM play_predictions
    GROUP BY
        head_coach,
        posteam
    """
)


# Team-season summaries
connection.execute(
    """
    CREATE VIEW team_season_summary AS
    SELECT
        season,
        posteam,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,

        ROUND(
            100 * AVG(is_pass),
            2
        ) AS actual_pass_rate_pct,

        ROUND(
            100 * AVG(
                expected_pass_probability
            ),
            2
        ) AS expected_pass_rate_pct,

        ROUND(
            100 * AVG(model_pass_oe),
            2
        ) AS model_pass_oe_pct,

        ROUND(
            AVG(epa),
            4
        ) AS mean_epa,

        ROUND(
            100 * AVG(success),
            2
        ) AS success_rate_pct,

        ROUND(
            AVG(yards_gained),
            2
        ) AS mean_yards_gained

    FROM play_predictions
    GROUP BY
        season,
        posteam
    """
)


# League season summaries
connection.execute(
    """
    CREATE VIEW league_season_summary AS
    SELECT
        season,
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,

        ROUND(
            100 * AVG(is_pass),
            2
        ) AS actual_pass_rate_pct,

        ROUND(
            100 * AVG(
                expected_pass_probability
            ),
            2
        ) AS expected_pass_rate_pct,

        ROUND(
            100 * AVG(model_pass_oe),
            2
        ) AS model_pass_oe_pct,

        ROUND(
            AVG(epa),
            4
        ) AS mean_epa,

        ROUND(
            100 * AVG(success),
            2
        ) AS success_rate_pct,

        ROUND(
            AVG(yards_gained),
            2
        ) AS mean_yards_gained

    FROM play_predictions
    GROUP BY season
    """
)


# Indexes for common Streamlit filters
indexes = [
    (
        "idx_play_season",
        "season"
    ),
    (
        "idx_play_coach",
        "head_coach"
    ),
    (
        "idx_play_team",
        "posteam"
    ),
    (
        "idx_play_season_type",
        "season_type"
    ),
    (
        "idx_play_down",
        "down"
    ),
    (
        "idx_play_quarter",
        "qtr"
    ),
    (
        "idx_play_game",
        "game_id"
    ),
]

for index_name, column_name in indexes:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
            {index_name}
        ON play_predictions(
            {column_name}
        )
        """
    )


# Indexes for common Play Style filters
style_indexes = [
    ("idx_style_season", "season"),
    ("idx_style_coach", "head_coach"),
    ("idx_style_team", "posteam"),
    ("idx_style_play_call", "play_call"),
    ("idx_style_formation", "formation_style"),
    ("idx_style_pass_depth", "pass_depth_bucket"),
    ("idx_style_run_direction", "run_direction"),
]

for index_name, column_name in style_indexes:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
            {index_name}
        ON play_style_predictions(
            {column_name}
        )
        """
    )


# Indexes for team-result filters and joins
team_result_indexes = [
    (
        "idx_team_game_game",
        "game_id"
    ),
    (
        "idx_team_game_season",
        "season"
    ),
    (
        "idx_team_game_team",
        "team"
    ),
    (
        "idx_team_game_coach",
        "head_coach"
    ),
    (
        "idx_team_game_season_type",
        "season_type"
    ),
]

for index_name, column_name in team_result_indexes:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
            {index_name}
        ON team_game_results(
            {column_name}
        )
        """
    )


# Indexes for coaching-tenure filters and joins
tenure_indexes = [
    (
        "idx_tenure_season",
        "season"
    ),
    (
        "idx_tenure_coach",
        "head_coach"
    ),
    (
        "idx_tenure_team",
        "team"
    ),
]

for index_name, column_name in tenure_indexes:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
            {index_name}
        ON coach_tenures(
            {column_name}
        )
        """
    )


# Indexes for play-caller reference joins and filters
play_caller_indexes = [
    (
        "idx_play_caller_season",
        "season"
    ),
    (
        "idx_play_caller_team",
        "team"
    ),
    (
        "idx_play_caller_head_coach",
        "head_coach"
    ),
    (
        "idx_play_caller_name",
        "offensive_play_caller"
    ),
    (
        "idx_play_caller_status",
        "verification_status"
    ),
]

for index_name, column_name in play_caller_indexes:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
            {index_name}
        ON offensive_play_caller_tenures(
            {column_name}
        )
        """
    )


# Database validation
play_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_predictions
    """
).fetchone()[0]

unique_games = connection.execute(
    """
    SELECT COUNT(DISTINCT game_id)
    FROM play_predictions
    """
).fetchone()[0]

season_range = connection.execute(
    """
    SELECT
        MIN(season),
        MAX(season)
    FROM play_predictions
    """
).fetchone()

duplicate_plays = connection.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            game_id,
            play_id,
            COUNT(*) AS play_count
        FROM play_predictions
        GROUP BY
            game_id,
            play_id
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

missing_predictions = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_predictions
    WHERE
        expected_pass_probability
        IS NULL
    """
).fetchone()[0]

team_game_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM team_game_results
    """
).fetchone()[0]

team_result_unique_games = connection.execute(
    """
    SELECT COUNT(DISTINCT game_id)
    FROM team_game_results
    """
).fetchone()[0]

team_result_unique_teams = connection.execute(
    """
    SELECT COUNT(DISTINCT team)
    FROM team_game_results
    """
).fetchone()[0]

duplicate_team_games = connection.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            game_id,
            team,
            COUNT(*) AS row_count
        FROM team_game_results
        GROUP BY
            game_id,
            team
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

missing_team_result_coaches = connection.execute(
    """
    SELECT COUNT(*)
    FROM team_game_results
    WHERE head_coach IS NULL
    """
).fetchone()[0]

team_season_record_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM team_season_records
    """
).fetchone()[0]

coach_team_season_record_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM coach_team_season_records
    """
).fetchone()[0]

coach_team_season_type_record_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM coach_team_season_type_records
    """
).fetchone()[0]

coach_tenure_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM coach_tenures
    """
).fetchone()[0]

unique_tenure_coaches = connection.execute(
    """
    SELECT COUNT(DISTINCT head_coach)
    FROM coach_tenures
    """
).fetchone()[0]

duplicate_tenures = connection.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            season,
            head_coach,
            team,
            COUNT(*) AS row_count
        FROM coach_tenures
        GROUP BY
            season,
            head_coach,
            team
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

missing_tenure_coaches = connection.execute(
    """
    SELECT COUNT(*)
    FROM coach_tenures
    WHERE head_coach IS NULL
    """
).fetchone()[0]

play_caller_reference_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM offensive_play_caller_tenures
    """
).fetchone()[0]

verified_play_caller_segments = connection.execute(
    """
    SELECT COUNT(*)
    FROM offensive_play_caller_tenures
    WHERE verification_status = 'verified'
    """
).fetchone()[0]

joined_play_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_predictions_with_callers
    """
).fetchone()[0]

duplicate_joined_plays = connection.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            game_id,
            play_id,
            COUNT(*) AS row_count
        FROM play_predictions_with_callers
        GROUP BY
            game_id,
            play_id
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

missing_reference_matches = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_predictions_with_callers
    WHERE play_caller_tenure_id IS NULL
    """
).fetchone()[0]

style_play_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_style_predictions
    """
).fetchone()[0]

style_caller_play_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_style_predictions_with_callers
    """
).fetchone()[0]

missing_style_formations = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_style_predictions
    WHERE formation_style IS NULL
    """
).fetchone()[0]

duplicate_style_plays = connection.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            game_id,
            play_id,
            COUNT(*) AS row_count
        FROM play_style_predictions
        GROUP BY
            game_id,
            play_id
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

fallback_play_attributions = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_predictions_with_callers
    WHERE
        attribution_type !=
            'verified_offensive_play_caller'
    """
).fetchone()[0]

verified_caller_plays = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_predictions_with_callers
    WHERE
        attribution_type =
            'verified_offensive_play_caller'
    """
).fetchone()[0]

verified_caller_seasons = connection.execute(
    """
    SELECT COUNT(DISTINCT season)
    FROM play_predictions_with_callers
    WHERE
        attribution_type =
            'verified_offensive_play_caller'
    """
).fetchone()[0]

play_caller_season_summary_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_season_summary
    """
).fetchone()[0]

play_caller_season_type_summary_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_season_type_summary
    """
).fetchone()[0]

play_caller_career_summary_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_career_summary
    """
).fetchone()[0]

play_caller_team_career_summary_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_team_career_summary
    """
).fetchone()[0]

missing_summary_callers = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_season_summary
    WHERE offensive_play_caller IS NULL
    """
).fetchone()[0]

play_caller_uncertainty_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_uncertainty
    """
).fetchone()[0]

eligible_play_caller_uncertainty_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_uncertainty
    WHERE meets_minimum_sample = true
    """
).fetchone()[0]

missing_play_caller_intervals = connection.execute(
    """
    SELECT COUNT(*)
    FROM play_caller_uncertainty
    WHERE
        ci_95_lower_pct IS NULL
        OR ci_95_upper_pct IS NULL
    """
).fetchone()[0]

duplicate_play_caller_intervals = connection.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT
            season,
            offensive_play_caller,
            posteam,
            COUNT(*) AS row_count
        FROM play_caller_uncertainty
        GROUP BY
            season,
            offensive_play_caller,
            posteam
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

table_names = connection.execute(
    """
    SHOW TABLES
    """
).fetchall()


print("\nDATABASE CREATED SUCCESSFULLY")
print("Saved to:", database_path)


print("\nDATABASE VALIDATION")
print(
    f"Play-level rows: {play_count:,}"
)
print(
    f"Unique games: {unique_games:,}"
)
print(
    f"Season range: "
    f"{season_range[0]}-"
    f"{season_range[1]}"
)
print(
    f"Duplicate plays: "
    f"{duplicate_plays:,}"
)
print(
    f"Missing predictions: "
    f"{missing_predictions:,}"
)

print("\nPLAY-STYLE VALIDATION")
print(
    f"Style play rows: {style_play_count:,}"
)
print(
    "Caller-enriched style rows: "
    f"{style_caller_play_count:,}"
)
print(
    "Missing formation labels: "
    f"{missing_style_formations:,}"
)
print(
    "Duplicate style plays: "
    f"{duplicate_style_plays:,}"
)


print("\nTEAM-RESULT VALIDATION")
print(
    f"Team-game rows: "
    f"{team_game_count:,}"
)
print(
    f"Unique result games: "
    f"{team_result_unique_games:,}"
)
print(
    f"Unique team identifiers: "
    f"{team_result_unique_teams:,}"
)
print(
    f"Duplicate team-game rows: "
    f"{duplicate_team_games:,}"
)
print(
    f"Missing team-result coaches: "
    f"{missing_team_result_coaches:,}"
)
print(
    f"Team-season record rows: "
    f"{team_season_record_count:,}"
)
print(
    f"Coach-team-season record rows: "
    f"{coach_team_season_record_count:,}"
)
print(
    "Coach team-season-type record rows: "
    f"{coach_team_season_type_record_count:,}"
)


print("\nCOACH-TENURE VALIDATION")
print(
    f"Coach-tenure rows: "
    f"{coach_tenure_count:,}"
)
print(
    f"Unique tenure coaches: "
    f"{unique_tenure_coaches:,}"
)
print(
    f"Duplicate coach-team-season tenures: "
    f"{duplicate_tenures:,}"
)
print(
    f"Missing tenure coaches: "
    f"{missing_tenure_coaches:,}"
)


print("\nPLAY-CALLER VALIDATION")
print(
    f"Reference segments: "
    f"{play_caller_reference_count:,}"
)
print(
    f"Verified reference segments: "
    f"{verified_play_caller_segments:,}"
)
print(
    f"Joined play rows: "
    f"{joined_play_count:,}"
)
print(
    f"Duplicate joined plays: "
    f"{duplicate_joined_plays:,}"
)
print(
    f"Plays with verified callers: "
    f"{verified_caller_plays:,}"
)
print(
    f"Verified caller seasons: "
    f"{verified_caller_seasons:,}"
)
print(
    f"Plays missing reference matches: "
    f"{missing_reference_matches:,}"
)
print(
    f"Plays using head-coach fallback: "
    f"{fallback_play_attributions:,}"
)
print(
    f"Play-caller season summary rows: "
    f"{play_caller_season_summary_count:,}"
)
print(
    "Play-caller season-type summary rows: "
    f"{play_caller_season_type_summary_count:,}"
)
print(
    f"Play-caller career summary rows: "
    f"{play_caller_career_summary_count:,}"
)
print(
    "Play-caller team-career summary rows: "
    f"{play_caller_team_career_summary_count:,}"
)
print(
    f"Missing callers in season summary: "
    f"{missing_summary_callers:,}"
)

print("\nPLAY-CALLER UNCERTAINTY VALIDATION")
print(
    f"Caller-season interval rows: "
    f"{play_caller_uncertainty_count:,}"
)
print(
    f"Eligible caller-season intervals: "
    f"{eligible_play_caller_uncertainty_count:,}"
)
print(
    f"Missing caller intervals: "
    f"{missing_play_caller_intervals:,}"
)
print(
    f"Duplicate caller intervals: "
    f"{duplicate_play_caller_intervals:,}"
)


# Stop rather than silently publishing a database with a broken or
# many-to-many play-caller join.
validation_errors = []

if style_play_count != play_count:
    validation_errors.append(
        "The play-style table row count does not match "
        "play_predictions."
    )

if style_caller_play_count != play_count:
    validation_errors.append(
        "The caller-enriched play-style view row count does "
        "not match play_predictions."
    )

if missing_style_formations > 0:
    validation_errors.append(
        "The play-style table contains missing formation labels."
    )

if duplicate_style_plays > 0:
    validation_errors.append(
        "The play-style table contains duplicate plays."
    )

if joined_play_count != play_count:
    validation_errors.append(
        "The caller-enriched view does not have the same "
        "number of rows as play_predictions."
    )

if duplicate_joined_plays > 0:
    validation_errors.append(
        "The caller-enriched view contains duplicate plays."
    )

if missing_reference_matches > 0:
    validation_errors.append(
        "Some plays did not match a play-caller "
        "reference segment."
    )

if fallback_play_attributions > 0:
    validation_errors.append(
        "Some plays are still using head-coach fallback "
        "attribution."
    )

if verified_caller_plays != play_count:
    validation_errors.append(
        "The number of verified play-caller plays does not "
        "match play_predictions."
    )

if verified_caller_seasons != 8:
    validation_errors.append(
        "Verified play-caller attribution does not cover all "
        "eight seasons from 2018 through 2025."
    )

if play_caller_season_summary_count == 0:
    validation_errors.append(
        "The play-caller season summary is empty."
    )

if play_caller_career_summary_count == 0:
    validation_errors.append(
        "The play-caller career summary is empty."
    )

if missing_summary_callers > 0:
    validation_errors.append(
        "The play-caller season summary contains missing "
        "caller names."
    )

if play_caller_uncertainty_count != (
    play_caller_season_summary_count
):
    validation_errors.append(
        "The play-caller uncertainty row count does not "
        "match the play-caller season summary."
    )

if missing_play_caller_intervals > 0:
    validation_errors.append(
        "The play-caller uncertainty table contains "
        "missing confidence intervals."
    )

if duplicate_play_caller_intervals > 0:
    validation_errors.append(
        "The play-caller uncertainty table contains "
        "duplicate caller-team-season rows."
    )

if validation_errors:
    error_message = "\n".join(
        f"- {message}"
        for message in validation_errors
    )

    connection.close()

    raise RuntimeError(
        "Play-caller database validation failed:\n"
        f"{error_message}"
    )


print("\nDATABASE TABLES AND VIEWS")
for table_name in table_names:
    print(
        f"- {table_name[0]}"
    )


# Close database connection
connection.close()