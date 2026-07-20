from pathlib import Path

import nflreadpy as nfl
import polars as pl


pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)


SEASONS = list(range(2018, 2026))


# Fields are audited against the play population where they are meaningful.
FIELD_GROUPS = {
    "pass_calls": [
        "pass_length",
        "pass_location",
        "air_yards",
    ],
    "run_calls": [
        "run_location",
        "run_gap",
    ],
    "all_calls": [
        "shotgun",
        "no_huddle",
        "play_action",
    ],
}


project_root = Path(__file__).resolve().parents[1]

output_path = (
    project_root
    / "outputs"
    / "tables"
    / "play_style_field_audit_2018_2025.csv"
)


def define_play_call(pbp: pl.DataFrame) -> pl.DataFrame:
    """Apply the same intended run-pass target used by the project."""

    return pbp.with_columns(
        pl.when(pl.col("qb_dropback") == 1)
        .then(pl.lit("pass"))
        .when(pl.col("rush_attempt") == 1)
        .then(pl.lit("run"))
        .otherwise(None)
        .alias("play_call")
    )


def filter_competitive_calls(
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    """Retain the project's competitive run-pass decisions."""

    return pbp.filter(
        pl.col("play_call").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("down").is_not_null()
        & (pl.col("qb_kneel").fill_null(0) == 0)
        & (pl.col("qb_spike").fill_null(0) == 0)
        & (pl.col("aborted_play").fill_null(0) == 0)
    )


def applicable_plays(
    plays: pl.DataFrame,
    field_group: str,
) -> pl.DataFrame:
    """Select the appropriate denominator for a style field."""

    if field_group == "pass_calls":
        return plays.filter(
            pl.col("play_call") == "pass"
        )

    if field_group == "run_calls":
        return plays.filter(
            pl.col("play_call") == "run"
        )

    return plays


def format_unique_values(
    data: pl.DataFrame,
    field: str,
) -> str:
    """Create a compact, deterministic preview of observed values."""

    values = (
        data
        .select(pl.col(field).drop_nulls().unique())
        .to_series()
        .to_list()
    )

    values = sorted(
        values,
        key=lambda value: str(value),
    )

    preview = values[:20]
    text = " | ".join(
        str(value)
        for value in preview
    )

    if len(values) > len(preview):
        text += " | ..."

    return text


audit_rows = []
season_summary_rows = []


print("AUDITING PLAY-STYLE FIELD COVERAGE")
print(
    "Seasons: "
    + ", ".join(str(season) for season in SEASONS)
)


for season in SEASONS:
    print(f"\nLOADING {season} PLAY-BY-PLAY")

    pbp = nfl.load_pbp(seasons=[season])
    available_columns = set(pbp.columns)

    required_target_columns = {
        "qb_dropback",
        "rush_attempt",
        "posteam",
        "defteam",
        "down",
        "qb_kneel",
        "qb_spike",
        "aborted_play",
    }

    missing_target_columns = sorted(
        required_target_columns
        - available_columns
    )

    if missing_target_columns:
        raise RuntimeError(
            f"{season} is missing target/filter columns: "
            + ", ".join(missing_target_columns)
        )

    plays = filter_competitive_calls(
        define_play_call(pbp)
    )

    pass_calls = plays.filter(
        pl.col("play_call") == "pass"
    ).height

    run_calls = plays.filter(
        pl.col("play_call") == "run"
    ).height

    season_summary_rows.append({
        "season": season,
        "competitive_plays": plays.height,
        "pass_calls": pass_calls,
        "run_calls": run_calls,
    })

    print(
        f"Competitive calls: {plays.height:,} | "
        f"Pass: {pass_calls:,} | Run: {run_calls:,}"
    )

    for field_group, fields in FIELD_GROUPS.items():
        denominator_data = applicable_plays(
            plays,
            field_group,
        )

        denominator = denominator_data.height

        for field in fields:
            field_available = (
                field in available_columns
            )

            if field_available:
                non_null_count = denominator_data[
                    field
                ].is_not_null().sum()

                null_count = (
                    denominator - non_null_count
                )

                coverage_percentage = (
                    100 * non_null_count / denominator
                    if denominator
                    else 0.0
                )

                unique_count = denominator_data[
                    field
                ].drop_nulls().n_unique()

                unique_values = format_unique_values(
                    denominator_data,
                    field,
                )
            else:
                non_null_count = 0
                null_count = denominator
                coverage_percentage = 0.0
                unique_count = 0
                unique_values = ""

            audit_rows.append({
                "season": season,
                "field": field,
                "field_group": field_group,
                "available_in_schema": field_available,
                "applicable_plays": denominator,
                "non_null_plays": int(non_null_count),
                "null_plays": int(null_count),
                "coverage_percentage": round(
                    coverage_percentage,
                    2,
                ),
                "unique_count": int(unique_count),
                "unique_values_preview": unique_values,
            })


audit = pl.DataFrame(audit_rows)
season_summary = pl.DataFrame(
    season_summary_rows
)


overall_audit = (
    audit
    .group_by([
        "field",
        "field_group",
    ])
    .agg([
        pl.col("available_in_schema")
        .all()
        .alias("available_all_seasons"),
        pl.col("applicable_plays")
        .sum()
        .alias("applicable_plays"),
        pl.col("non_null_plays")
        .sum()
        .alias("non_null_plays"),
        pl.col("null_plays")
        .sum()
        .alias("null_plays"),
        pl.col("coverage_percentage")
        .min()
        .alias("minimum_season_coverage_pct"),
        pl.col("coverage_percentage")
        .max()
        .alias("maximum_season_coverage_pct"),
    ])
    .with_columns(
        (
            100
            * pl.col("non_null_plays")
            / pl.col("applicable_plays")
        )
        .round(2)
        .alias("overall_coverage_pct")
    )
    .sort([
        "field_group",
        "field",
    ])
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.write_csv(output_path)


print("\nSEASON PLAY COUNTS")
print(season_summary)


print("\nOVERALL FIELD COVERAGE")
print(overall_audit)


print("\nFIELDS MISSING FROM AT LEAST ONE SEASON")
print(
    overall_audit.filter(
        ~pl.col("available_all_seasons")
    )
)


print("\nLOW-COVERAGE FIELDS")
print(
    overall_audit.filter(
        pl.col("overall_coverage_pct") < 80
    )
)


print("\nSAVED DETAILED AUDIT")
print(output_path)