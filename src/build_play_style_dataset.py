from pathlib import Path

import nflreadpy as nfl
import polars as pl


pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)


SEASONS = list(range(2018, 2026))

STYLE_COLUMNS = [
    "pass_length",
    "pass_location",
    "air_yards",
    "run_location",
    "run_gap",
    "shotgun",
    "no_huddle",
    "sack",
    "qb_scramble",
]


project_root = Path(__file__).resolve().parents[1]

predictions_path = (
    project_root
    / "outputs"
    / "tables"
    / "historical_play_predictions_2018_2025.parquet"
)

output_path = (
    project_root
    / "outputs"
    / "tables"
    / "play_style_predictions_2018_2025.parquet"
)


if not predictions_path.exists():
    raise FileNotFoundError(
        "Historical predictions were not found: "
        f"{predictions_path}"
    )


print("LOADING HISTORICAL PLAY PREDICTIONS")

predictions = pl.read_parquet(
    predictions_path
)


style_frames = []


print("\nLOADING PLAY-STYLE FIELDS")

for season in SEASONS:
    print(f"Loading {season} play-by-play...")

    pbp = nfl.load_pbp(seasons=[season])
    available_columns = set(pbp.columns)

    required_keys = {
        "game_id",
        "play_id",
        "season",
    }

    missing_keys = sorted(
        required_keys - available_columns
    )

    if missing_keys:
        raise RuntimeError(
            f"{season} is missing join keys: "
            + ", ".join(missing_keys)
        )

    missing_style_columns = [
        column
        for column in STYLE_COLUMNS
        if column not in available_columns
    ]

    if missing_style_columns:
        raise RuntimeError(
            f"{season} is missing required style columns: "
            + ", ".join(missing_style_columns)
        )

    season_styles = pbp.select([
        "game_id",
        "play_id",
        "season",
        *STYLE_COLUMNS,
    ])

    duplicate_style_keys = (
        season_styles
        .group_by([
            "game_id",
            "play_id",
        ])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_style_keys > 0:
        raise RuntimeError(
            f"{season} contains "
            f"{duplicate_style_keys:,} duplicate "
            "game-play style keys."
        )

    style_frames.append(season_styles)

    print(
        f"{season} style rows loaded: "
        f"{season_styles.height:,}"
    )


style_data = pl.concat(
    style_frames,
    how="vertical_relaxed",
)


print("\nJOINING STYLE FIELDS TO MODEL PREDICTIONS")

style_predictions = predictions.join(
    style_data,
    on=[
        "game_id",
        "play_id",
        "season",
    ],
    how="left",
    validate="1:1",
)


# Create analysis-friendly labels while retaining every raw source field.
style_predictions = (
    style_predictions
    .with_columns([
        pl.when(pl.col("shotgun") == 1)
        .then(pl.lit("Shotgun"))
        .when(pl.col("shotgun") == 0)
        .then(pl.lit("Under center"))
        .otherwise(pl.lit("Unknown"))
        .alias("formation_style"),

        pl.when(pl.col("no_huddle") == 1)
        .then(pl.lit("No huddle"))
        .when(pl.col("no_huddle") == 0)
        .then(pl.lit("Standard tempo"))
        .otherwise(pl.lit("Unknown"))
        .alias("tempo_style"),

        pl.when(
            (pl.col("play_call") == "pass")
            & (pl.col("sack").fill_null(0) == 1)
        )
        .then(pl.lit("Sack"))
        .when(
            (pl.col("play_call") == "pass")
            & (
                pl.col("qb_scramble")
                .fill_null(0)
                == 1
            )
        )
        .then(pl.lit("Scramble"))
        .when(
            (pl.col("play_call") == "pass")
            & pl.col("air_yards").is_not_null()
        )
        .then(pl.lit("Throw"))
        .when(pl.col("play_call") == "pass")
        .then(pl.lit("Unclassified dropback"))
        .otherwise(None)
        .alias("dropback_result"),

        pl.when(
            (pl.col("play_call") == "pass")
            & pl.col("air_yards").is_not_null()
            & (pl.col("air_yards") < 0)
        )
        .then(pl.lit("Behind line"))
        .when(
            (pl.col("play_call") == "pass")
            & pl.col("air_yards").is_between(
                0,
                9,
                closed="both",
            )
        )
        .then(pl.lit("Short (0-9)"))
        .when(
            (pl.col("play_call") == "pass")
            & pl.col("air_yards").is_between(
                10,
                19,
                closed="both",
            )
        )
        .then(pl.lit("Intermediate (10-19)"))
        .when(
            (pl.col("play_call") == "pass")
            & (pl.col("air_yards") >= 20)
        )
        .then(pl.lit("Deep (20+)"))
        .otherwise(None)
        .alias("pass_depth_bucket"),

        pl.when(
            (pl.col("play_call") == "pass")
            & pl.col("pass_location").is_not_null()
        )
        .then(
            pl.col("pass_location")
            .str.to_titlecase()
        )
        .otherwise(None)
        .alias("pass_direction"),

        pl.when(
            (pl.col("play_call") == "run")
            & pl.col("run_location").is_not_null()
        )
        .then(
            pl.col("run_location")
            .str.to_titlecase()
        )
        .otherwise(None)
        .alias("run_direction"),

        pl.when(
            (pl.col("play_call") == "run")
            & pl.col("run_gap").is_not_null()
        )
        .then(
            pl.col("run_gap")
            .str.to_titlecase()
        )
        .otherwise(None)
        .alias("run_gap_style"),
    ])
)


missing_style_matches = style_predictions[
    "shotgun"
].null_count()

duplicate_output_plays = (
    style_predictions
    .group_by([
        "game_id",
        "play_id",
    ])
    .len()
    .filter(pl.col("len") > 1)
    .height
)


if style_predictions.height != predictions.height:
    raise RuntimeError(
        "The style join changed the number of prediction rows. "
        f"Expected {predictions.height:,}; received "
        f"{style_predictions.height:,}."
    )

if missing_style_matches > 0:
    raise RuntimeError(
        f"{missing_style_matches:,} prediction rows did not "
        "receive style fields."
    )

if duplicate_output_plays > 0:
    raise RuntimeError(
        f"The output contains {duplicate_output_plays:,} "
        "duplicate plays."
    )


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

style_predictions.write_parquet(
    output_path
)


def print_distribution(
    title: str,
    field: str,
    play_call: str | None = None,
) -> None:
    """Print category counts and percentages for validation."""

    data = style_predictions

    if play_call is not None:
        data = data.filter(
            pl.col("play_call") == play_call
        )

    data = data.filter(
        pl.col(field).is_not_null()
    )

    print(f"\n{title}")

    if data.is_empty():
        print("No classified plays")
        return

    print(
        data
        .group_by(field)
        .agg(pl.len().alias("plays"))
        .with_columns(
            (
                100 * pl.col("plays")
                / data.height
            )
            .round(2)
            .alias("percentage")
        )
        .sort("plays", descending=True)
    )


print("\nPLAY-STYLE DATASET CREATED")
print("Shape:", style_predictions.shape)


print("\nJOIN VALIDATION")
print(f"Prediction rows: {predictions.height:,}")
print(
    "Style-enriched rows: "
    f"{style_predictions.height:,}"
)
print(
    "Missing style matches: "
    f"{missing_style_matches:,}"
)
print(
    "Duplicate output plays: "
    f"{duplicate_output_plays:,}"
)


print_distribution(
    "FORMATION DISTRIBUTION",
    "formation_style",
)

print_distribution(
    "TEMPO DISTRIBUTION",
    "tempo_style",
)

print_distribution(
    "DROPBACK RESULT DISTRIBUTION",
    "dropback_result",
    play_call="pass",
)

print_distribution(
    "CHARTED PASS DEPTH DISTRIBUTION",
    "pass_depth_bucket",
    play_call="pass",
)

print_distribution(
    "CHARTED PASS DIRECTION DISTRIBUTION",
    "pass_direction",
    play_call="pass",
)

print_distribution(
    "RUN DIRECTION DISTRIBUTION",
    "run_direction",
    play_call="run",
)

print_distribution(
    "CHARTED RUN GAP DISTRIBUTION",
    "run_gap_style",
    play_call="run",
)


print("\nSAVED FILE")
print(output_path)