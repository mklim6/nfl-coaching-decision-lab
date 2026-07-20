from pathlib import Path

import polars as pl


project_root = Path(__file__).resolve().parents[1]

team_results_path = (
    project_root
    / "data"
    / "processed"
    / "team_game_results_2018_2025.parquet"
)

play_predictions_path = (
    project_root
    / "outputs"
    / "tables"
    / "historical_play_predictions_2018_2025.parquet"
)


team_results = pl.read_parquet(team_results_path)
play_predictions = pl.read_parquet(play_predictions_path)


result_teams = sorted(
    team_results
    .get_column("team")
    .drop_nulls()
    .unique()
    .to_list()
)

play_teams = sorted(
    play_predictions
    .get_column("posteam")
    .drop_nulls()
    .unique()
    .to_list()
)


print("TEAM-RESULT IDENTIFIERS")
print(result_teams)
print("Count:", len(result_teams))

print("\nPLAY-PREDICTION IDENTIFIERS")
print(play_teams)
print("Count:", len(play_teams))

print("\nONLY IN TEAM RESULTS")
print(sorted(set(result_teams) - set(play_teams)))

print("\nONLY IN PLAY PREDICTIONS")
print(sorted(set(play_teams) - set(result_teams)))

print("\nTEAM IDENTIFIERS BY SEASON")
print(
    team_results
    .group_by("season")
    .agg(
        pl.col("team")
        .unique()
        .sort()
        .alias("teams")
    )
    .sort("season")
)