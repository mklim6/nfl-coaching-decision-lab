from pathlib import Path

import polars as pl


pl.Config.set_tbl_rows(50)


project_root = Path(__file__).resolve().parents[1]

reference_path = (
    project_root
    / "data"
    / "reference"
    / "offensive_play_caller_tenures.csv"
)


if not reference_path.exists():
    raise FileNotFoundError(
        "Play-caller reference file was not found: "
        f"{reference_path}"
    )


reference = pl.read_csv(reference_path)


required_columns = [
    "tenure_id",
    "season",
    "team",
    "start_week",
    "end_week",
    "head_coach",
    "offensive_play_caller",
    "caller_role",
    "head_coach_is_play_caller",
    "verification_status",
    "source_url",
    "source_title",
    "source_publisher",
    "source_date",
    "date_verified",
    "verification_notes",
]

missing_columns = [
    column
    for column in required_columns
    if column not in reference.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing: "
        + ", ".join(missing_columns)
    )


duplicate_ids = (
    reference
    .group_by("tenure_id")
    .len()
    .filter(pl.col("len") > 1)
)

invalid_week_ranges = reference.filter(
    pl.col("start_week") > pl.col("end_week")
)

verified_rows = reference.filter(
    pl.col("verification_status") == "verified"
)

verified_missing_evidence = verified_rows.filter(
    pl.col("offensive_play_caller").is_null()
    | pl.col("caller_role").is_null()
    | pl.col("head_coach_is_play_caller").is_null()
    | pl.col("source_url").is_null()
    | pl.col("source_title").is_null()
    | pl.col("source_publisher").is_null()
    | pl.col("source_date").is_null()
    | pl.col("date_verified").is_null()
)


reference_2025 = reference.filter(
    pl.col("season") == 2025
)

unverified_2025 = reference_2025.filter(
    pl.col("verification_status") != "verified"
)


play_caller_changes_2025 = (
    reference_2025
    .group_by("team")
    .agg([
        pl.len().alias("segments"),
        pl.col("offensive_play_caller")
        .n_unique()
        .alias("unique_callers"),
        pl.col("offensive_play_caller")
        .unique()
        .sort()
        .alias("callers"),
    ])
    .filter(pl.col("unique_callers") > 1)
    .sort("team")
)


print("PLAY-CALLER REFERENCE AUDIT")

print("\nOVERALL SUMMARY")
print(
    reference.select([
        pl.len().alias("rows"),
        pl.col("season").min().alias("first_season"),
        pl.col("season").max().alias("last_season"),
        pl.col("team").n_unique().alias("unique_teams"),
        pl.col("offensive_play_caller")
        .drop_nulls()
        .n_unique()
        .alias("verified_caller_names"),
    ])
)

print("\nSTATUS COUNTS")
print(
    reference
    .group_by("verification_status")
    .len()
    .sort("verification_status")
)

print("\nVALIDATION")
print(f"Duplicate tenure IDs: {duplicate_ids.height}")
print(f"Invalid week ranges: {invalid_week_ranges.height}")
print(
    "Verified rows missing evidence: "
    f"{verified_missing_evidence.height}"
)

print("\n2025 VALIDATION")
print(f"2025 segments: {reference_2025.height}")
print(
    "2025 teams: "
    f"{reference_2025['team'].n_unique()}"
)
print(f"Unverified 2025 segments: {unverified_2025.height}")

print("\n2025 TEAMS WITH PLAY-CALLER CHANGES")
print(play_caller_changes_2025)

print("\n2025 VERIFIED SEGMENTS")
print(
    reference_2025.select([
        "team",
        "start_week",
        "end_week",
        "head_coach",
        "offensive_play_caller",
        "caller_role",
        "head_coach_is_play_caller",
        "verification_status",
    ])
    .sort([
        "team",
        "start_week",
    ])
)


if duplicate_ids.height > 0:
    raise ValueError("Duplicate tenure IDs were found.")

if invalid_week_ranges.height > 0:
    raise ValueError("Invalid week ranges were found.")

if verified_missing_evidence.height > 0:
    raise ValueError(
        "Verified rows with missing evidence were found."
    )

if unverified_2025.height > 0:
    raise ValueError(
        "Unverified 2025 play-caller segments remain."
    )

print("\nAUDIT PASSED")