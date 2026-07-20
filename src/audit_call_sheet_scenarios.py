from pathlib import Path

import duckdb
import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

database_path = (
    project_root
    / "database"
    / "coaching_lab.duckdb"
)

output_directory = (
    project_root
    / "outputs"
    / "tables"
)

scenario_output_path = (
    output_directory
    / "call_sheet_scenario_match_audit.csv"
)

entity_output_path = (
    output_directory
    / "call_sheet_entity_coverage_audit.csv"
)


if not database_path.exists():
    raise FileNotFoundError(
        "The coaching database was not found: "
        f"{database_path}"
    )


output_directory.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Scenario definitions
#
# quarter_seconds_remaining is the number of seconds remaining in the quarter.
# yardline_100 is the number of yards from the opponent's end zone.
# -----------------------------------------------------------------------------

SCENARIOS = [
    {
        "scenario": "Opening-drive first down",
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 75,
        "qtr": 1,
        "quarter_seconds_remaining": 780,
        "score_differential": 0,
        "goal_to_go": 0,
    },
    {
        "scenario": "Second-and-medium",
        "down": 2,
        "ydstogo": 6,
        "yardline_100": 60,
        "qtr": 2,
        "quarter_seconds_remaining": 480,
        "score_differential": 0,
        "goal_to_go": 0,
    },
    {
        "scenario": "Third-and-short",
        "down": 3,
        "ydstogo": 2,
        "yardline_100": 45,
        "qtr": 2,
        "quarter_seconds_remaining": 600,
        "score_differential": 0,
        "goal_to_go": 0,
    },
    {
        "scenario": "Third-and-seven in plus territory",
        "down": 3,
        "ydstogo": 7,
        "yardline_100": 38,
        "qtr": 2,
        "quarter_seconds_remaining": 480,
        "score_differential": 0,
        "goal_to_go": 0,
    },
    {
        "scenario": "Fourth-and-one decision",
        "down": 4,
        "ydstogo": 1,
        "yardline_100": 42,
        "qtr": 4,
        "quarter_seconds_remaining": 300,
        "score_differential": -3,
        "goal_to_go": 0,
    },
    {
        "scenario": "Red-zone second down",
        "down": 2,
        "ydstogo": 7,
        "yardline_100": 15,
        "qtr": 2,
        "quarter_seconds_remaining": 420,
        "score_differential": 0,
        "goal_to_go": 0,
    },
    {
        "scenario": "Two-minute comeback",
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 70,
        "qtr": 4,
        "quarter_seconds_remaining": 90,
        "score_differential": -4,
        "goal_to_go": 0,
    },
    {
        "scenario": "Protecting a late lead",
        "down": 2,
        "ydstogo": 7,
        "yardline_100": 60,
        "qtr": 4,
        "quarter_seconds_remaining": 360,
        "score_differential": 7,
        "goal_to_go": 0,
    },
]


MATCH_TIERS = [
    "Strict",
    "Balanced",
    "Broad",
]


def distance_bounds(distance):
    """Return the football distance bucket containing the target distance."""

    if distance <= 3:
        return 1, 3, "Short (1-3)"

    if distance <= 6:
        return 4, 6, "Medium (4-6)"

    if distance <= 10:
        return 7, 10, "Long (7-10)"

    return 11, 99, "Very long (11+)"


def field_zone_bounds(yardline_100):
    """Return the field-position zone containing the target yardline."""

    if yardline_100 <= 20:
        return 1, 20, "Red zone"

    if yardline_100 <= 50:
        return 21, 50, "Plus territory"

    if yardline_100 <= 80:
        return 51, 80, "Own territory"

    return 81, 99, "Backed up"


def score_bounds(score_differential):
    """Return the game-score state containing the target differential."""

    if score_differential <= -9:
        return -99, -9, "Trailing by 9+"

    if score_differential <= -1:
        return -8, -1, "Trailing by 1-8"

    if score_differential == 0:
        return 0, 0, "Tied"

    if score_differential <= 8:
        return 1, 8, "Leading by 1-8"

    return 9, 99, "Leading by 9+"


def target_game_seconds(scenario):
    """Convert a quarter clock into regulation game seconds remaining."""

    quarter = scenario["qtr"]
    quarter_seconds = scenario[
        "quarter_seconds_remaining"
    ]

    if quarter == 1:
        return 2700 + quarter_seconds

    if quarter == 2:
        return 1800 + quarter_seconds

    if quarter == 3:
        return 900 + quarter_seconds

    # Fourth-quarter and overtime rows already use the remaining
    # seconds shown for that period in the source data.
    return quarter_seconds


def build_match_filter(scenario, match_tier):
    """Build a parameterized SQL filter for one scenario and match tier."""

    distance_low, distance_high, distance_label = distance_bounds(
        scenario["ydstogo"]
    )

    field_low, field_high, field_label = field_zone_bounds(
        scenario["yardline_100"]
    )

    score_low, score_high, score_label = score_bounds(
        scenario["score_differential"]
    )

    game_seconds_target = target_game_seconds(
        scenario
    )

    conditions = [
        "down = ?",
        "goal_to_go = ?",
    ]

    parameters = [
        scenario["down"],
        scenario["goal_to_go"],
    ]

    if match_tier == "Strict":
        conditions.extend([
            "ydstogo BETWEEN ? AND ?",
            "yardline_100 BETWEEN ? AND ?",
            "qtr = ?",
            "game_seconds_remaining BETWEEN ? AND ?",
            "score_differential BETWEEN ? AND ?",
        ])

        parameters.extend([
            max(1, scenario["ydstogo"] - 1),
            scenario["ydstogo"] + 1,
            max(1, scenario["yardline_100"] - 10),
            min(99, scenario["yardline_100"] + 10),
            scenario["qtr"],
            max(
                0,
                game_seconds_target - 180,
            ),
            min(
                3600,
                game_seconds_target + 180,
            ),
            scenario["score_differential"] - 3,
            scenario["score_differential"] + 3,
        ])

    elif match_tier == "Balanced":
        conditions.extend([
            "ydstogo BETWEEN ? AND ?",
            "yardline_100 BETWEEN ? AND ?",
            "qtr = ?",
            "game_seconds_remaining BETWEEN ? AND ?",
            "score_differential BETWEEN ? AND ?",
        ])

        parameters.extend([
            distance_low,
            distance_high,
            field_low,
            field_high,
            scenario["qtr"],
            max(
                0,
                game_seconds_target - 300,
            ),
            min(
                3600,
                game_seconds_target + 300,
            ),
            score_low,
            score_high,
        ])

    elif match_tier == "Broad":
        target_half = (
            1
            if scenario["qtr"] in (1, 2)
            else 2
        )

        half_quarters = (
            (1, 2)
            if target_half == 1
            else (3, 4, 5)
        )

        quarter_placeholders = ", ".join(
            ["?"] * len(half_quarters)
        )

        conditions.extend([
            "ydstogo BETWEEN ? AND ?",
            "yardline_100 BETWEEN ? AND ?",
            f"qtr IN ({quarter_placeholders})",
            "score_differential BETWEEN ? AND ?",
        ])

        parameters.extend([
            distance_low,
            distance_high,
            field_low,
            field_high,
            *half_quarters,
            score_low,
            score_high,
        ])

    else:
        raise ValueError(
            f"Unknown match tier: {match_tier}"
        )

    where_clause = (
        "WHERE "
        + " AND ".join(conditions)
    )

    metadata = {
        "distance_bucket": distance_label,
        "field_zone": field_label,
        "score_state": score_label,
    }

    return where_clause, parameters, metadata


def coverage_label(plays, games, coach_groups_25):
    """Assign a UX-oriented coverage label, not a significance claim."""

    if (
        plays >= 1000
        and games >= 100
        and coach_groups_25 >= 16
    ):
        return "Strong simulator coverage"

    if (
        plays >= 400
        and games >= 50
        and coach_groups_25 >= 8
    ):
        return "Usable with sample warning"

    return "Thin coverage"


# -----------------------------------------------------------------------------
# Run audit
# -----------------------------------------------------------------------------

connection = duckdb.connect(
    str(database_path),
    read_only=True,
)


required_columns = {
    "season",
    "game_id",
    "play_id",
    "posteam",
    "head_coach",
    "offensive_play_caller",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "goal_to_go",
    "score_differential",
    "is_pass",
    "expected_pass_probability",
    "model_pass_oe",
    "epa",
    "success",
    "formation_style",
    "tempo_style",
    "pass_depth_bucket",
    "pass_direction",
    "run_direction",
    "run_gap_style",
}


available_columns = {
    row[1]
    for row in connection.execute(
        "PRAGMA table_info('play_style_predictions_with_callers')"
    ).fetchall()
}


missing_columns = sorted(
    required_columns - available_columns
)


if missing_columns:
    raise ValueError(
        "The simulator audit cannot run because these database "
        "columns are missing: "
        + ", ".join(missing_columns)
    )


scenario_rows = []
entity_rows = []


print("AUDITING CALL-SHEET SCENARIO COVERAGE")
print("Data seasons: 2018-2025")
print(
    "Match tiers: Strict, Balanced, Broad"
)


for scenario in SCENARIOS:
    print(
        "\nSCENARIO: "
        + scenario["scenario"]
    )

    for match_tier in MATCH_TIERS:
        (
            where_clause,
            parameters,
            metadata,
        ) = build_match_filter(
            scenario,
            match_tier,
        )

        league_row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS plays,
                COUNT(DISTINCT game_id) AS games,
                COUNT(DISTINCT season) AS seasons,
                COUNT(DISTINCT posteam) AS teams,
                AVG(is_pass) AS actual_pass_rate,
                AVG(expected_pass_probability)
                    AS expected_pass_rate,
                AVG(model_pass_oe) AS pass_oe,
                AVG(epa) AS mean_epa,
                AVG(success) AS success_rate
            FROM play_style_predictions_with_callers
            {where_clause}
            """,
            parameters,
        ).fetchone()

        (
            plays,
            games,
            seasons,
            teams,
            actual_pass_rate,
            expected_pass_rate,
            pass_oe,
            mean_epa,
            success_rate,
        ) = league_row

        entity_coverage = connection.execute(
            f"""
            WITH entity_samples AS (
                SELECT
                    'Head coach' AS attribution,
                    head_coach AS entity,
                    posteam,
                    COUNT(*) AS plays,
                    COUNT(DISTINCT game_id) AS games
                FROM play_style_predictions_with_callers
                {where_clause}
                GROUP BY head_coach, posteam

                UNION ALL

                SELECT
                    'Verified offensive play caller'
                        AS attribution,
                    offensive_play_caller AS entity,
                    posteam,
                    COUNT(*) AS plays,
                    COUNT(DISTINCT game_id) AS games
                FROM play_style_predictions_with_callers
                {where_clause}
                GROUP BY offensive_play_caller, posteam
            )
            SELECT
                attribution,
                COUNT(*) AS entity_team_groups,
                SUM(CASE WHEN plays >= 25 THEN 1 ELSE 0 END)
                    AS groups_25_plus,
                SUM(CASE WHEN plays >= 50 THEN 1 ELSE 0 END)
                    AS groups_50_plus,
                SUM(CASE WHEN plays >= 100 THEN 1 ELSE 0 END)
                    AS groups_100_plus,
                SUM(CASE WHEN plays >= 200 THEN 1 ELSE 0 END)
                    AS groups_200_plus,
                MEDIAN(plays) AS median_group_plays,
                MAX(plays) AS maximum_group_plays,
                MEDIAN(games) AS median_group_games,
                MAX(games) AS maximum_group_games
            FROM entity_samples
            WHERE entity IS NOT NULL
            GROUP BY attribution
            ORDER BY attribution
            """,
            parameters + parameters,
        ).fetchall()

        coverage_by_attribution = {}

        for coverage_row in entity_coverage:
            (
                attribution,
                entity_team_groups,
                groups_25_plus,
                groups_50_plus,
                groups_100_plus,
                groups_200_plus,
                median_group_plays,
                maximum_group_plays,
                median_group_games,
                maximum_group_games,
            ) = coverage_row

            coverage_by_attribution[
                attribution
            ] = {
                "groups_25_plus": int(
                    groups_25_plus
                ),
            }

            entity_rows.append({
                "scenario": scenario["scenario"],
                "match_tier": match_tier,
                "attribution": attribution,
                "entity_team_groups": int(
                    entity_team_groups
                ),
                "groups_25_plus": int(
                    groups_25_plus
                ),
                "groups_50_plus": int(
                    groups_50_plus
                ),
                "groups_100_plus": int(
                    groups_100_plus
                ),
                "groups_200_plus": int(
                    groups_200_plus
                ),
                "median_group_plays": float(
                    median_group_plays
                ),
                "maximum_group_plays": int(
                    maximum_group_plays
                ),
                "median_group_games": float(
                    median_group_games
                ),
                "maximum_group_games": int(
                    maximum_group_games
                ),
            })

        coach_groups_25 = coverage_by_attribution.get(
            "Head coach",
            {},
        ).get(
            "groups_25_plus",
            0,
        )

        scenario_rows.append({
            "scenario": scenario["scenario"],
            "match_tier": match_tier,
            "target_down": scenario["down"],
            "target_ydstogo": scenario["ydstogo"],
            "target_yardline_100": scenario["yardline_100"],
            "target_quarter": scenario["qtr"],
            "target_quarter_seconds_remaining": (
                scenario["quarter_seconds_remaining"]
            ),
            "target_score_differential": (
                scenario["score_differential"]
            ),
            "distance_bucket": metadata["distance_bucket"],
            "field_zone": metadata["field_zone"],
            "score_state": metadata["score_state"],
            "league_plays": int(plays),
            "league_games": int(games),
            "seasons": int(seasons),
            "teams": int(teams),
            "actual_pass_rate_pct": (
                round(100 * actual_pass_rate, 2)
                if actual_pass_rate is not None
                else None
            ),
            "expected_pass_rate_pct": (
                round(100 * expected_pass_rate, 2)
                if expected_pass_rate is not None
                else None
            ),
            "pass_oe_pct": (
                round(100 * pass_oe, 2)
                if pass_oe is not None
                else None
            ),
            "mean_epa": (
                round(mean_epa, 4)
                if mean_epa is not None
                else None
            ),
            "success_rate_pct": (
                round(100 * success_rate, 2)
                if success_rate is not None
                else None
            ),
            "coverage_label": coverage_label(
                int(plays),
                int(games),
                coach_groups_25,
            ),
        })

        print(
            f"  {match_tier:<8} | "
            f"{int(plays):>6,} plays | "
            f"{int(games):>5,} games | "
            f"{coach_groups_25:>2} coach-team groups "
            "with 25+ plays"
        )


scenario_audit = pl.DataFrame(
    scenario_rows
)

entity_audit = pl.DataFrame(
    entity_rows
)


scenario_audit.write_csv(
    scenario_output_path
)

entity_audit.write_csv(
    entity_output_path
)


print("\nSCENARIO COVERAGE SUMMARY")
print(
    scenario_audit.select([
        "scenario",
        "match_tier",
        "league_plays",
        "league_games",
        "actual_pass_rate_pct",
        "expected_pass_rate_pct",
        "pass_oe_pct",
        "coverage_label",
    ])
)


print("\nENTITY COVERAGE SUMMARY")
print(
    entity_audit.select([
        "scenario",
        "match_tier",
        "attribution",
        "groups_25_plus",
        "groups_50_plus",
        "groups_100_plus",
        "groups_200_plus",
        "median_group_plays",
        "maximum_group_plays",
    ])
)


print("\nSAVED FILES")
print(scenario_output_path)
print(entity_output_path)


connection.close()