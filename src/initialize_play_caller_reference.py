from datetime import date
from pathlib import Path

import polars as pl


# Project paths
project_root = Path(__file__).resolve().parents[1]

coach_tenures_path = (
    project_root
    / "outputs"
    / "tables"
    / "coach_tenures_2018_2025.csv"
)

output_path = (
    project_root
    / "data"
    / "reference"
    / "offensive_play_caller_tenures.csv"
)


if not coach_tenures_path.exists():
    raise FileNotFoundError(
        "Coach-tenure dataset was not found: "
        f"{coach_tenures_path}"
    )


# Protect future manual research from accidental replacement
if output_path.exists():
    raise FileExistsError(
        "The offensive play-caller reference file already "
        "exists and was not overwritten:\n"
        f"{output_path}\n\n"
        "This protection prevents manually verified research "
        "from being erased."
    )


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


print("LOADING COACH TENURES")

coach_tenures = pl.read_csv(
    coach_tenures_path
)


reference_rows = []

for row in coach_tenures.iter_rows(
    named=True
):
    tenure_id = (
        f"{row['season']}_"
        f"{row['team']}_"
        f"{row['first_week']}_"
        f"{row['last_week']}_"
        f"{row['head_coach']}"
    )

    reference_rows.append({
        "tenure_id": tenure_id,
        "season": row["season"],
        "team": row["team"],
        "start_week": row["first_week"],
        "end_week": row["last_week"],
        "head_coach": row["head_coach"],
        "head_coach_started_midseason": (
            row["started_midseason"]
        ),

        # Fields to populate through manual verification
        "offensive_play_caller": None,
        "caller_role": None,
        "head_coach_is_play_caller": None,

        # Verification documentation
        "verification_status": "unverified",
        "source_url": None,
        "source_title": None,
        "source_publisher": None,
        "source_date": None,
        "date_verified": None,
        "verification_notes": (
            "Manual verification required. Split this row "
            "if play-calling responsibility changed during "
            "the listed week range."
        ),
    })


play_caller_reference = (
    pl.DataFrame(
        reference_rows,
        schema_overrides={
            "offensive_play_caller": pl.String,
            "caller_role": pl.String,
            "head_coach_is_play_caller": pl.Boolean,
            "source_url": pl.String,
            "source_title": pl.String,
            "source_publisher": pl.String,
            "source_date": pl.String,
            "date_verified": pl.String,
        },
    )
    .sort([
        "season",
        "team",
        "start_week",
        "head_coach",
    ])
)


play_caller_reference.write_csv(
    output_path
)


print("\nOFFENSIVE PLAY-CALLER REFERENCE CREATED")

print("\nSUMMARY")
print(
    play_caller_reference.select([
        pl.len().alias("reference_rows"),

        pl.col("season")
        .min()
        .alias("first_season"),

        pl.col("season")
        .max()
        .alias("last_season"),

        pl.col("team")
        .n_unique()
        .alias("unique_teams"),

        pl.col("head_coach")
        .n_unique()
        .alias("unique_head_coaches"),
    ])
)


print("\nVERIFICATION STATUS")
print(
    play_caller_reference
    .group_by("verification_status")
    .len()
    .sort("verification_status")
)


print("\n2025 RESEARCH QUEUE")
print(
    play_caller_reference
    .filter(
        pl.col("season") == 2025
    )
    .select([
        "season",
        "team",
        "start_week",
        "end_week",
        "head_coach",
        "offensive_play_caller",
        "verification_status",
    ])
    .sort([
        "team",
        "start_week",
    ])
)


print("\nIMPORTANT")
print(
    "Do not rerun this initializer after adding verified "
    "research. It intentionally refuses to overwrite the "
    "reference file."
)

print("\nSAVED FILE")
print(output_path)