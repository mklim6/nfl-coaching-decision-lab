import nflreadpy as nfl


# Load one season first so we can inspect the available fields
pbp = nfl.load_pbp(seasons=[2025])
schedules = nfl.load_schedules(seasons=[2025])


print("PLAY-BY-PLAY DATA")
print("Shape:", pbp.shape)
print("Number of columns:", len(pbp.columns))
print("Columns:")
print(pbp.columns)


print("\nSCHEDULE DATA")
print("Shape:", schedules.shape)
print("Columns:")
print(schedules.columns)


# Check for fields we expect to use
candidate_fields = [
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "goal_to_go",
    "shotgun",
    "no_huddle",
    "qb_kneel",
    "qb_spike",
    "pass",
    "rush",
    "epa",
    "success",
    "home_team",
    "away_team",
]

available_fields = [
    field for field in candidate_fields
    if field in pbp.columns
]

missing_fields = [
    field for field in candidate_fields
    if field not in pbp.columns
]


print("\nAVAILABLE CANDIDATE FIELDS")
print(available_fields)

print("\nMISSING CANDIDATE FIELDS")
print(missing_fields)

print("\nSAMPLE PLAYS")
print(
    pbp.select(available_fields).head(10)
)