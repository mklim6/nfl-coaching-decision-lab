from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import py_compile
import subprocess

import duckdb


project_root = Path(__file__).resolve().parents[1]

app_path = project_root / "app" / "app.py"
database_path = (
    project_root
    / "database"
    / "coaching_lab.duckdb"
)
requirements_path = project_root / "requirements.txt"
readme_path = project_root / "README.md"
gitignore_path = project_root / ".gitignore"

required_paths = [
    app_path,
    database_path,
    requirements_path,
    readme_path,
    gitignore_path,
]

required_tables = {
    "play_predictions",
    "play_predictions_with_callers",
    "play_style_predictions_with_callers",
    "coach_uncertainty",
    "play_caller_uncertainty",
    "coach_tenures",
    "team_game_results",
    "final_test_metrics",
    "model_metrics_by_season",
}

packages = [
    "streamlit",
    "duckdb",
    "pandas",
    "plotly",
]


print("NFL COACHING DECISION LAB DEPLOYMENT AUDIT")

print("\nREQUIRED FILES")
missing_paths = []
for path in required_paths:
    exists = path.exists()
    print(f"{'PASS' if exists else 'MISSING'}: {path}")
    if not exists:
        missing_paths.append(path)

if missing_paths:
    raise FileNotFoundError(
        "Required deployment files are missing."
    )


print("\nPACKAGE VERSIONS")
missing_packages = []
for package in packages:
    try:
        package_version = version(package)
        print(f"{package}=={package_version}")
    except PackageNotFoundError:
        print(f"MISSING: {package}")
        missing_packages.append(package)

if missing_packages:
    raise ModuleNotFoundError(
        "Required deployment packages are missing."
    )


print("\nAPP SYNTAX")
py_compile.compile(
    str(app_path),
    doraise=True,
)
print("PASS: app/app.py compiled successfully")


print("\nDATABASE VALIDATION")
connection = duckdb.connect(
    str(database_path),
    read_only=True,
)

available_tables = {
    row[0]
    for row in connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        """
    ).fetchall()
}

missing_tables = sorted(
    required_tables - available_tables
)

print(f"Available tables and views: {len(available_tables):,}")
print(
    "Missing required tables: "
    + (", ".join(missing_tables) if missing_tables else "0")
)

if missing_tables:
    raise ValueError(
        "The deployment database is missing required tables."
    )

validation = connection.execute(
    """
    SELECT
        COUNT(*) AS plays,
        COUNT(DISTINCT game_id) AS games,
        COUNT(*) - COUNT(DISTINCT CONCAT(game_id, '-', play_id))
            AS duplicate_plays,
        SUM(
            CASE
                WHEN expected_pass_probability IS NULL THEN 1
                ELSE 0
            END
        ) AS missing_predictions,
        MIN(season) AS first_season,
        MAX(season) AS last_season
    FROM play_predictions
    """
).fetchone()

print(f"Play rows: {validation[0]:,}")
print(f"Unique games: {validation[1]:,}")
print(f"Duplicate plays: {validation[2]:,}")
print(f"Missing predictions: {validation[3]:,}")
print(f"Season range: {validation[4]}-{validation[5]}")

if validation[2] != 0 or validation[3] != 0:
    raise ValueError(
        "Database play-level validation failed."
    )

connection.close()


print("\nGITHUB FILE-SIZE CHECK")
large_files = []
large_lfs_files = []

for path in project_root.rglob("*"):
    if not path.is_file():
        continue

    if ".git" in path.parts:
        continue

    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb < 95:
        continue

    relative_path = path.relative_to(
        project_root
    )

    attribute_check = subprocess.run(
        [
            "git",
            "check-attr",
            "filter",
            "--",
            relative_path.as_posix(),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if attribute_check.stdout.strip().endswith(
        ": lfs"
    ):
        large_lfs_files.append(
            (path, size_mb)
        )
    else:
        large_files.append(
            (path, size_mb)
        )


for path, size_mb in large_lfs_files:
    print(
        f"PASS VIA GIT LFS: "
        f"{size_mb:.1f} MB - {path}"
    )


if large_files:
    for path, size_mb in large_files:
        print(
            f"TOO LARGE: "
            f"{size_mb:.1f} MB - {path}"
        )

    raise ValueError(
        "One or more large files are not tracked "
        "through Git LFS."
    )


if not large_lfs_files:
    print(
        "PASS: no individual file is "
        "95 MB or larger"
    )


print("\nDEPLOYMENT AUDIT PASSED")