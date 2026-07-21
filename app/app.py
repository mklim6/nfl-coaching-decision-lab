from pathlib import Path
import random

# NFL Coaching Decision Lab with realistic randomized Make the Call sessions.

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="NFL Coaching Decision Lab",
    layout="wide",
)


# Project paths
project_root = Path(__file__).resolve().parents[1]

database_path = (
    project_root
    / "database"
    / "coaching_lab.duckdb"
)


if not database_path.exists():
    st.error(
        "The coaching database was not found. "
        "Run src/build_database.py first."
    )
    st.stop()


# Use a connection dedicated to this Streamlit script run.
# A shared cached DuckDB connection can mix query results
# when Streamlit reruns overlap.
connection = duckdb.connect(
    str(database_path),
    read_only=True,
)


@st.cache_data
def load_filter_options():
    seasons = connection.execute(
        """
        SELECT DISTINCT season
        FROM play_predictions
        ORDER BY season
        """
    ).fetchdf()["season"].tolist()

    coaches = connection.execute(
        """
        SELECT DISTINCT head_coach
        FROM play_predictions
        WHERE head_coach IS NOT NULL
        ORDER BY head_coach
        """
    ).fetchdf()["head_coach"].tolist()

    play_callers = connection.execute(
        """
        SELECT DISTINCT offensive_play_caller
        FROM play_predictions_with_callers
        WHERE
            attribution_type =
                'verified_offensive_play_caller'
            AND offensive_play_caller IS NOT NULL
        ORDER BY offensive_play_caller
        """
    ).fetchdf()["offensive_play_caller"].tolist()

    teams = connection.execute(
        """
        SELECT DISTINCT posteam
        FROM play_predictions
        WHERE posteam IS NOT NULL
        ORDER BY posteam
        """
    ).fetchdf()["posteam"].tolist()

    bounds = connection.execute(
        """
        SELECT
            MIN(ydstogo) AS min_distance,
            MAX(ydstogo) AS max_distance,
            MIN(yardline_100) AS min_yardline,
            MAX(yardline_100) AS max_yardline,
            MIN(score_differential) AS min_score,
            MAX(score_differential) AS max_score
        FROM play_predictions
        """
    ).fetchdf().iloc[0]

    return {
        "seasons": seasons,
        "coaches": coaches,
        "play_callers": play_callers,
        "teams": teams,
        "bounds": bounds,
    }


filter_options = load_filter_options()
bounds = filter_options["bounds"]


def apply_exploration_preset(preset_name):
    """Apply a curated story to the shared sidebar filters."""

    full_distance = (
        int(bounds["min_distance"]),
        int(bounds["max_distance"]),
    )
    full_field = (
        int(bounds["min_yardline"]),
        int(bounds["max_yardline"]),
    )
    full_score = (
        int(bounds["min_score"]),
        int(bounds["max_score"]),
    )

    presets = {
        "reid": {
            "filter_seasons": list(filter_options["seasons"]),
            "filter_season_types": ["REG", "POST"],
            "filter_coaches": ["Andy Reid"],
            "filter_teams": [],
            "filter_downs": [1, 2, 3, 4],
            "filter_quarters": [1, 2, 3, 4, 5],
            "filter_distance": full_distance,
            "filter_field": full_field,
            "filter_score": full_score,
        },
        "ravens": {
            "filter_seasons": [2024, 2025],
            "filter_season_types": ["REG", "POST"],
            "filter_coaches": ["John Harbaugh"],
            "filter_teams": ["BAL"],
            "filter_downs": [1, 2, 3, 4],
            "filter_quarters": [1, 2, 3, 4, 5],
            "filter_distance": full_distance,
            "filter_field": full_field,
            "filter_score": full_score,
        },
        "postseason": {
            "filter_seasons": list(filter_options["seasons"]),
            "filter_season_types": ["POST"],
            "filter_coaches": [],
            "filter_teams": [],
            "filter_downs": [1, 2, 3, 4],
            "filter_quarters": [1, 2, 3, 4, 5],
            "filter_distance": full_distance,
            "filter_field": full_field,
            "filter_score": full_score,
        },
        "third_down": {
            "filter_seasons": list(filter_options["seasons"]),
            "filter_season_types": ["REG", "POST"],
            "filter_coaches": [],
            "filter_teams": [],
            "filter_downs": [3],
            "filter_quarters": [1, 2, 3, 4, 5],
            "filter_distance": (3, 10),
            "filter_field": full_field,
            "filter_score": (-8, 8),
        },
        "reset": {
            "filter_seasons": [2025],
            "filter_season_types": ["REG", "POST"],
            "filter_coaches": [],
            "filter_teams": [],
            "filter_downs": [1, 2, 3, 4],
            "filter_quarters": [1, 2, 3, 4, 5],
            "filter_distance": full_distance,
            "filter_field": full_field,
            "filter_score": full_score,
        },
    }

    for key, value in presets[preset_name].items():
        st.session_state[key] = value


st.title("NFL Coaching Decision Lab")

st.caption(
    "Explore NFL run-pass decisions relative to a "
    "league-wide situation-only expectation model."
)


# Sidebar filters
st.sidebar.header("Filters")


selected_seasons = st.sidebar.multiselect(
    "Seasons",
    options=filter_options["seasons"],
    default=[2025],
    key="filter_seasons",
)


selected_season_types = st.sidebar.multiselect(
    "Season type",
    options=["REG", "POST"],
    default=["REG", "POST"],
    key="filter_season_types",
    format_func=lambda value: (
        "Regular season"
        if value == "REG"
        else "Postseason"
    ),
)


selected_coaches = st.sidebar.multiselect(
    "Head coaches",
    options=filter_options["coaches"],
    key="filter_coaches",
)


selected_teams = st.sidebar.multiselect(
    "Offensive teams",
    options=filter_options["teams"],
    key="filter_teams",
)


selected_downs = st.sidebar.multiselect(
    "Down",
    options=[1, 2, 3, 4],
    default=[1, 2, 3, 4],
    key="filter_downs",
)


selected_quarters = st.sidebar.multiselect(
    "Quarter",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5],
    key="filter_quarters",
    format_func=lambda value: (
        "Overtime"
        if value == 5
        else f"Quarter {value}"
    ),
)


distance_bounds = st.sidebar.slider(
    "Yards to go",
    min_value=int(bounds["min_distance"]),
    max_value=int(bounds["max_distance"]),
    value=(
        int(bounds["min_distance"]),
        int(bounds["max_distance"]),
    ),
    key="filter_distance",
)


yardline_bounds = st.sidebar.slider(
    "Yards from opponent end zone",
    min_value=int(bounds["min_yardline"]),
    max_value=int(bounds["max_yardline"]),
    value=(
        int(bounds["min_yardline"]),
        int(bounds["max_yardline"]),
    ),
    key="filter_field",
)


score_bounds = st.sidebar.slider(
    "Offense score differential",
    min_value=int(bounds["min_score"]),
    max_value=int(bounds["max_score"]),
    value=(
        int(bounds["min_score"]),
        int(bounds["max_score"]),
    ),
    key="filter_score",
)


minimum_plays = st.sidebar.number_input(
    "Minimum plays for rankings",
    min_value=1,
    max_value=5000,
    value=500,
    step=50,
)

st.sidebar.caption(
    "Sample guidance: 500+ plays is the stronger default; "
    "200-499 plays is a limited exploratory sample; fewer "
    "than 200 plays should be interpreted with substantial caution."
)

if minimum_plays < 200:
    st.sidebar.warning(
        "The selected minimum is below 200 plays. Rankings at "
        "this level are highly sensitive to a relatively small "
        "number of games and should be treated as exploratory."
    )
elif minimum_plays < 500:
    st.sidebar.info(
        "The selected minimum is below the stronger 500-play "
        "default. Results are available for exploration, but "
        "sample sizes are more limited."
    )


def build_filter_query(
    coach_override=None,
    coach_column="head_coach",
    use_sidebar_coaches=True,
):
    """Build safe parameterized SQL filters."""

    conditions = []
    parameters = []

    def add_list_filter(
        column_name,
        selected_values,
    ):
        if not selected_values:
            return

        placeholders = ", ".join(
            ["?"] * len(selected_values)
        )

        conditions.append(
            f"{column_name} IN ({placeholders})"
        )

        parameters.extend(
            selected_values
        )

    add_list_filter(
        "season",
        selected_seasons,
    )

    add_list_filter(
        "season_type",
        selected_season_types,
    )

    if coach_override is not None:
        add_list_filter(
            coach_column,
            coach_override,
        )
    elif use_sidebar_coaches:
        add_list_filter(
            "head_coach",
            selected_coaches,
        )

    add_list_filter(
        "posteam",
        selected_teams,
    )

    add_list_filter(
        "down",
        selected_downs,
    )

    add_list_filter(
        "qtr",
        selected_quarters,
    )

    conditions.extend([
        "ydstogo BETWEEN ? AND ?",
        "yardline_100 BETWEEN ? AND ?",
        "score_differential BETWEEN ? AND ?",
    ])

    parameters.extend([
        distance_bounds[0],
        distance_bounds[1],
        yardline_bounds[0],
        yardline_bounds[1],
        score_bounds[0],
        score_bounds[1],
    ])

    where_clause = (
        "WHERE "
        + " AND ".join(conditions)
    )

    return where_clause, parameters


def build_record_filter_query(
    coach_override=None,
):
    """Build filters for game-level team records."""

    conditions = []
    parameters = []

    def add_list_filter(
        column_name,
        selected_values,
    ):
        if not selected_values:
            return

        placeholders = ", ".join(
            ["?"] * len(selected_values)
        )

        conditions.append(
            f"{column_name} IN ({placeholders})"
        )

        parameters.extend(selected_values)

    add_list_filter(
        "season",
        selected_seasons,
    )

    add_list_filter(
        "season_type",
        selected_season_types,
    )

    if coach_override is None:
        add_list_filter(
            "head_coach",
            selected_coaches,
        )
    else:
        add_list_filter(
            "head_coach",
            coach_override,
        )

    add_list_filter(
        "team",
        selected_teams,
    )

    if conditions:
        where_clause = (
            "WHERE "
            + " AND ".join(conditions)
        )
    else:
        where_clause = ""

    return where_clause, parameters


def load_coach_team_records(
    coach_override=None,
):
    """Aggregate records for selected coach-team samples."""

    record_where, record_parameters = (
        build_record_filter_query(
            coach_override=coach_override
        )
    )

    return connection.execute(
        f"""
        SELECT
            head_coach,
            team AS posteam,
            COUNT(*) AS record_games,
            SUM(win) AS wins,
            SUM(loss) AS losses,
            SUM(tie) AS ties,

            ROUND(
                (
                    SUM(win)
                    + 0.5 * SUM(tie)
                ) / COUNT(*),
                3
            ) AS win_percentage,

            ROUND(
                SUM(points_for) / COUNT(*),
                2
            ) AS points_per_game,

            ROUND(
                SUM(points_against) / COUNT(*),
                2
            ) AS points_allowed_per_game,

            ROUND(
                SUM(point_differential)
                / COUNT(*),
                2
            ) AS point_differential_per_game

        FROM team_game_results
        {record_where}

        GROUP BY
            head_coach,
            team
        """,
        record_parameters,
    ).fetchdf()


def load_coach_team_records_by_season(
    coach_override=None,
):
    """Aggregate records by coach, team, and season."""

    record_where, record_parameters = (
        build_record_filter_query(
            coach_override=coach_override
        )
    )

    return connection.execute(
        f"""
        SELECT
            season,
            head_coach,
            team AS posteam,
            COUNT(*) AS record_games,
            SUM(win) AS wins,
            SUM(loss) AS losses,
            SUM(tie) AS ties,

            ROUND(
                (
                    SUM(win)
                    + 0.5 * SUM(tie)
                ) / COUNT(*),
                3
            ) AS win_percentage,

            ROUND(
                SUM(points_for) / COUNT(*),
                2
            ) AS points_per_game,

            ROUND(
                SUM(points_against) / COUNT(*),
                2
            ) AS points_allowed_per_game,

            ROUND(
                SUM(point_differential)
                / COUNT(*),
                2
            ) AS point_differential_per_game

        FROM team_game_results
        {record_where}

        GROUP BY
            season,
            head_coach,
            team
        """,
        record_parameters,
    ).fetchdf()


def load_coach_team_records_by_season_type(
    coach_override=None,
):
    """Aggregate records by coach, team, and season type."""

    record_where, record_parameters = (
        build_record_filter_query(
            coach_override=coach_override
        )
    )

    return connection.execute(
        f"""
        SELECT
            season_type,
            head_coach,
            team AS posteam,
            COUNT(*) AS games,
            SUM(win) AS wins,
            SUM(loss) AS losses,
            SUM(tie) AS ties,

            ROUND(
                (
                    SUM(win)
                    + 0.5 * SUM(tie)
                ) / COUNT(*),
                3
            ) AS win_percentage,

            ROUND(
                SUM(points_for) / COUNT(*),
                2
            ) AS points_per_game,

            ROUND(
                SUM(points_against) / COUNT(*),
                2
            ) AS points_allowed_per_game,

            ROUND(
                SUM(point_differential)
                / COUNT(*),
                2
            ) AS point_differential_per_game

        FROM team_game_results
        {record_where}

        GROUP BY
            season_type,
            head_coach,
            team

        ORDER BY
            head_coach,
            team,
            season_type
        """,
        record_parameters,
    ).fetchdf()


def load_team_records(
    group_by_season=False,
    group_by_season_type=False,
):
    """Load team outcomes as context without assigning them to a caller."""

    conditions = []
    parameters = []

    def add_list_filter(column_name, selected_values):
        if not selected_values:
            return

        placeholders = ", ".join(
            ["?"] * len(selected_values)
        )
        conditions.append(
            f"{column_name} IN ({placeholders})"
        )
        parameters.extend(selected_values)

    add_list_filter("season", selected_seasons)
    add_list_filter(
        "season_type",
        selected_season_types,
    )
    add_list_filter("team", selected_teams)

    record_where = (
        "WHERE " + " AND ".join(conditions)
        if conditions
        else ""
    )

    dimensions = []
    if group_by_season:
        dimensions.append("season")
    if group_by_season_type:
        dimensions.append("season_type")
    dimensions.append("team AS posteam")

    select_dimensions = ",\n            ".join(
        dimensions
    )
    group_dimensions = []
    if group_by_season:
        group_dimensions.append("season")
    if group_by_season_type:
        group_dimensions.append("season_type")
    group_dimensions.append("team")
    group_clause = ",\n            ".join(
        group_dimensions
    )

    return connection.execute(
        f"""
        SELECT
            {select_dimensions},
            COUNT(*) AS record_games,
            SUM(win) AS wins,
            SUM(loss) AS losses,
            SUM(tie) AS ties,
            ROUND(
                (SUM(win) + 0.5 * SUM(tie))
                / COUNT(*),
                3
            ) AS win_percentage,
            ROUND(AVG(points_for), 2)
                AS points_per_game,
            ROUND(AVG(points_against), 2)
                AS points_allowed_per_game,
            ROUND(AVG(point_differential), 2)
                AS point_differential_per_game
        FROM team_game_results
        {record_where}
        GROUP BY
            {group_clause}
        """,
        parameters,
    ).fetchdf()


def format_record(row):
    """Format wins, losses, and ties for display."""

    if row["wins"] != row["wins"]:
        return "N/A"

    wins = int(row["wins"])
    losses = int(row["losses"])
    ties = int(row["ties"])

    if ties:
        return f"{wins}-{losses}-{ties}"

    return f"{wins}-{losses}"


def format_prior_record(row):
    """Format the observed record entering a season."""

    if (
        row["observed_prior_hc_wins"]
        != row["observed_prior_hc_wins"]
    ):
        return "N/A"

    wins = int(row["observed_prior_hc_wins"])
    losses = int(row["observed_prior_hc_losses"])
    ties = int(row["observed_prior_hc_ties"])

    if ties:
        return f"{wins}-{losses}-{ties}"

    return f"{wins}-{losses}"


def load_latest_tenure_context(
    coach_override=None,
):
    """Load experience entering the latest selected season."""

    conditions = []
    parameters = []

    def add_list_filter(
        column_name,
        selected_values,
    ):
        if not selected_values:
            return

        placeholders = ", ".join(
            ["?"] * len(selected_values)
        )

        conditions.append(
            f"{column_name} IN ({placeholders})"
        )

        parameters.extend(selected_values)

    add_list_filter(
        "season",
        selected_seasons,
    )

    if coach_override is None:
        add_list_filter(
            "head_coach",
            selected_coaches,
        )
    else:
        add_list_filter(
            "head_coach",
            coach_override,
        )

    add_list_filter(
        "team",
        selected_teams,
    )

    if conditions:
        tenure_where = (
            "WHERE "
            + " AND ".join(conditions)
        )
    else:
        tenure_where = ""

    return connection.execute(
        f"""
        SELECT
            season AS experience_season,
            head_coach,
            team AS posteam,
            first_week,
            last_week,
            started_midseason,
            observed_prior_hc_seasons,
            observed_prior_hc_games,
            observed_prior_hc_wins,
            observed_prior_hc_losses,
            observed_prior_hc_ties,
            observed_prior_hc_win_percentage,
            first_observed_hc_season,
            observed_history_starts_at_window_boundary,
            experience_scope

        FROM coach_tenures
        {tenure_where}

        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY
                head_coach,
                team
            ORDER BY
                season DESC
        ) = 1
        """,
        parameters,
    ).fetchdf()


where_clause, query_parameters = (
    build_filter_query()
)


(
    overview_tab,
    rankings_tab,
    comparison_tab,
    trends_tab,
    simulator_tab,
    challenge_tab,
    style_tab,
    attribution_tab,
    model_tab,
    methodology_tab,
) = st.tabs([
    "Overview",
    "Coach Rankings",
    "Coach Comparison",
    "Historical Trends",
    "Situation Lab",
    "Make the Call",
    "Play Style",
    "Attribution Analysis",
    "Model Performance",
    "Methodology",
])


# Overview tab
with overview_tab:
    st.subheader("Welcome to the Coaching Decision Lab")
    st.write(
        "Turn eight seasons of NFL play-by-play into questions you can "
        "actually explore. Compare decision-makers, recreate game "
        "situations, or test your football instincts."
    )

    route_columns = st.columns(3)
    with route_columns[0]:
        with st.container(border=True):
            st.markdown("#### Compare decision-makers")
            st.caption(
                "Put coaches or verified offensive play callers side by "
                "side, with context-adjusted tendencies and outcomes."
            )
            st.markdown("Use the **Coach Comparison** tab above.")

    with route_columns[1]:
        with st.container(border=True):
            st.markdown("#### Build a game situation")
            st.caption(
                "Choose down, distance, field position, clock, and score "
                "to find comparable historical calls."
            )
            st.markdown("Open the **Situation Lab** tab above.")

    with route_columns[2]:
        with st.container(border=True):
            st.markdown("#### Test your instincts")
            st.caption(
                "Guess the historical majority call across a randomized "
                "deck without repeating a situation."
            )
            st.markdown("Play in the **Make the Call** tab above.")

    st.markdown("### Start with a story")
    st.caption(
        "Each button applies a curated set of sidebar filters. From there, "
        "every chart and table remains fully interactive."
    )

    story_columns = st.columns(4)
    with story_columns[0]:
        with st.container(border=True):
            st.markdown("**The Andy Reid profile**")
            st.caption(
                "Follow Reid's context-adjusted passing tendency across "
                "the complete 2018-2025 window."
            )
            st.button(
                "Explore Reid",
                on_click=apply_exploration_preset,
                args=("reid",),
                width="stretch",
            )

    with story_columns[1]:
        with st.container(border=True):
            st.markdown("**Baltimore's run identity**")
            st.caption(
                "Zoom in on Baltimore under John Harbaugh during the two "
                "most recent seasons."
            )
            st.button(
                "Explore Baltimore",
                on_click=apply_exploration_preset,
                args=("ravens",),
                width="stretch",
            )

    with story_columns[2]:
        with st.container(border=True):
            st.markdown("**Postseason decisions**")
            st.caption(
                "See how league play calling changes when only playoff "
                "games remain in the sample."
            )
            st.button(
                "Explore playoffs",
                on_click=apply_exploration_preset,
                args=("postseason",),
                width="stretch",
            )

    with story_columns[3]:
        with st.container(border=True):
            st.markdown("**Third-and-manageable**")
            st.caption(
                "Study third-and-3 through third-and-10 in one-score game "
                "states across all seasons."
            )
            st.button(
                "Explore third down",
                on_click=apply_exploration_preset,
                args=("third_down",),
                width="stretch",
            )

    st.button(
        "Reset to the 2025 overview",
        on_click=apply_exploration_preset,
        args=("reset",),
    )

    st.divider()
    st.markdown("### Current selection")

    overview = connection.execute(
        f"""
        SELECT
            COUNT(*) AS plays,
            COUNT(DISTINCT game_id) AS games,
            AVG(is_pass) AS actual_pass_rate,
            AVG(expected_pass_probability)
                AS expected_pass_rate,
            AVG(model_pass_oe) AS pass_oe,
            AVG(epa) AS mean_epa,
            AVG(success) AS success_rate,
            AVG(yards_gained) AS yards_per_play
        FROM play_predictions
        {where_clause}
        """,
        query_parameters,
    ).fetchdf().iloc[0]

    if overview["plays"] == 0:
        st.warning(
            "No plays match the selected filters."
        )
    else:
        row_one = st.columns(4)

        row_one[0].metric(
            "Plays",
            f"{int(overview['plays']):,}",
        )

        row_one[1].metric(
            "Games",
            f"{int(overview['games']):,}",
        )

        row_one[2].metric(
            "Actual pass rate",
            f"{100 * overview['actual_pass_rate']:.2f}%",
        )

        row_one[3].metric(
            "Expected pass rate",
            f"{100 * overview['expected_pass_rate']:.2f}%",
        )

        row_two = st.columns(4)

        row_two[0].metric(
            "Pass rate over expected",
            f"{100 * overview['pass_oe']:+.2f} pts",
        )

        row_two[1].metric(
            "EPA per play",
            f"{overview['mean_epa']:.3f}",
        )

        row_two[2].metric(
            "Success rate",
            f"{100 * overview['success_rate']:.2f}%",
        )

        row_two[3].metric(
            "Yards per play",
            f"{overview['yards_per_play']:.2f}",
        )

        st.divider()

        distribution = connection.execute(
            f"""
            SELECT
                play_call,
                COUNT(*) AS plays
            FROM play_predictions
            {where_clause}
            GROUP BY play_call
            ORDER BY play_call
            """,
            query_parameters,
        ).fetchdf()

        distribution_figure = px.pie(
            distribution,
            names="play_call",
            values="plays",
            title="Run-Pass Distribution",
            color="play_call",
            color_discrete_map={
                "pass": "#2471A3",
                "run": "#E67E22",
            },
            hole=0.45,
        )

        st.plotly_chart(
            distribution_figure,
            width="stretch",
        )


# Rankings tab
with rankings_tab:
    st.subheader("Coach-Team Tendencies")

    st.caption(
        "Positive values indicate more passing than "
        "expected. Negative values indicate more rushing. "
        "Records follow season and season-type filters, "
        "but not situation filters."
    )

    rankings = connection.execute(
        f"""
        SELECT
            head_coach,
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
            ) AS pass_oe_pct,

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
            ) AS yards_per_play

        FROM play_predictions
        {where_clause}

        GROUP BY
            head_coach,
            posteam

        HAVING COUNT(*) >= ?

        ORDER BY pass_oe_pct DESC
        """,
        query_parameters + [minimum_plays],
    ).fetchdf()

    ranking_records = load_coach_team_records()

    rankings = rankings.merge(
        ranking_records,
        on=[
            "head_coach",
            "posteam",
        ],
        how="left",
    )

    if rankings.empty:
        st.warning(
            "No coach-team samples meet the "
            "minimum-play requirement."
        )
    else:
        rankings["record"] = rankings.apply(
            format_record,
            axis=1,
        )

        rankings["label"] = (
            rankings["head_coach"]
            + " ("
            + rankings["posteam"]
            + ")"
        )

        ranking_figure = px.bar(
            rankings,
            x="pass_oe_pct",
            y="label",
            orientation="h",
            color="pass_oe_pct",
            color_continuous_scale=[
                "#C0392B",
                "#F4F6F7",
                "#2471A3",
            ],
            color_continuous_midpoint=0,
            title=(
                "Pass Rate Over Expected "
                "by Coach-Team"
            ),
            labels={
                "pass_oe_pct": (
                    "Pass OE "
                    "(percentage points)"
                ),
                "label": "",
            },
        )

        ranking_figure.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            },
            height=max(
                500,
                28 * len(rankings),
            ),
        )

        ranking_figure.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
        )

        st.plotly_chart(
            ranking_figure,
            width="stretch",
        )

        display_rankings = rankings[
            [
                "head_coach",
                "posteam",
                "plays",
                "games",
                "record_games",
                "record",
                "win_percentage",
                "points_per_game",
                "points_allowed_per_game",
                "point_differential_per_game",
                "actual_pass_rate_pct",
                "expected_pass_rate_pct",
                "pass_oe_pct",
                "mean_epa",
                "success_rate_pct",
                "yards_per_play",
            ]
        ].rename(
            columns={
                "head_coach": "Head Coach",
                "posteam": "Team",
                "plays": "Plays",
                "games": "Sample Games",
                "record_games": "Team Games",
                "record": "Record",
                "win_percentage": "PCT",
                "points_per_game": "Points/Game",
                "points_allowed_per_game": (
                    "Points Allowed/Game"
                ),
                "point_differential_per_game": (
                    "Point Diff/Game"
                ),
                "actual_pass_rate_pct": (
                    "Actual Pass %"
                ),
                "expected_pass_rate_pct": (
                    "Expected Pass %"
                ),
                "pass_oe_pct": "Pass OE",
                "mean_epa": "EPA/Play",
                "success_rate_pct": "Success %",
                "yards_per_play": "Yards/Play",
            }
        )

        st.dataframe(
            display_rankings,
            width="stretch",
            hide_index=True,
        )


# Coach comparison tab
with comparison_tab:
    st.subheader("Compare Decision-Makers")

    comparison_attribution = st.radio(
        "Attribute play-calling decisions to",
        options=[
            "Head coach",
            "Verified offensive play caller",
        ],
        horizontal=True,
        key="comparison_attribution",
    )

    comparison_is_caller = (
        comparison_attribution
        == "Verified offensive play caller"
    )

    comparison_options = (
        filter_options["play_callers"]
        if comparison_is_caller
        else filter_options["coaches"]
    )

    comparison_entity_column = (
        "offensive_play_caller"
        if comparison_is_caller
        else "head_coach"
    )

    comparison_source = (
        "play_predictions_with_callers"
        if comparison_is_caller
        else "play_predictions"
    )

    comparison_entity_label = (
        "Offensive Play Caller"
        if comparison_is_caller
        else "Head Coach"
    )

    default_comparison_candidates = (
        ["Andy Reid", "Sean McVay"]
        if comparison_is_caller
        else ["Andy Reid", "John Harbaugh"]
    )

    default_comparison_coaches = [
        coach
        for coach in default_comparison_candidates
        if coach in comparison_options
    ]

    comparison_entities = st.multiselect(
        (
            "Select two to five offensive play callers"
            if comparison_is_caller
            else "Select two to five head coaches"
        ),
        options=comparison_options,
        default=default_comparison_coaches,
        max_selections=5,
        key=(
            "comparison_play_callers"
            if comparison_is_caller
            else "comparison_coaches"
        ),
    )

    if len(comparison_entities) < 2:
        st.info(
            "Select at least two decision-makers."
        )
    else:
        (
            comparison_where_clause,
            comparison_parameters,
        ) = build_filter_query(
            coach_override=comparison_entities,
            coach_column=comparison_entity_column,
            use_sidebar_coaches=(
                not comparison_is_caller
            ),
        )

        comparison_data = connection.execute(
            f"""
            SELECT
                {comparison_entity_column}
                    AS head_coach,
                posteam,
                COUNT(*) AS plays,
                COUNT(DISTINCT game_id)
                    AS games,

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
                ) AS pass_oe_pct,

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
                ) AS yards_per_play

            FROM {comparison_source}
            {comparison_where_clause}

            GROUP BY
                {comparison_entity_column},
                posteam

            HAVING COUNT(*) >= ?

            ORDER BY pass_oe_pct DESC
            """,
            (
                comparison_parameters
                + [minimum_plays]
            ),
        ).fetchdf()

        if comparison_is_caller:
            comparison_records = load_team_records()
            comparison_merge_keys = ["posteam"]
        else:
            comparison_records = load_coach_team_records(
                coach_override=comparison_entities
            )
            comparison_merge_keys = [
                "head_coach",
                "posteam",
            ]

        comparison_data = comparison_data.merge(
            comparison_records,
            on=comparison_merge_keys,
            how="left",
        )

        if comparison_is_caller:
            comparison_data[
                "experience_season"
            ] = float("nan")
            comparison_data[
                "observed_prior_hc_seasons"
            ] = float("nan")
            comparison_data[
                "observed_prior_hc_games"
            ] = float("nan")
            comparison_data[
                "observed_prior_hc_wins"
            ] = float("nan")
            comparison_data[
                "observed_prior_hc_losses"
            ] = float("nan")
            comparison_data[
                "observed_prior_hc_ties"
            ] = float("nan")
        else:
            latest_tenure_context = (
                load_latest_tenure_context(
                    coach_override=comparison_entities
                )
            )

            comparison_data = comparison_data.merge(
                latest_tenure_context,
                on=[
                    "head_coach",
                    "posteam",
                ],
                how="left",
            )

        if comparison_data.empty:
            st.warning(
                "No comparison samples meet the "
                "current filters."
            )
        else:
            comparison_data["record"] = (
                comparison_data.apply(
                    format_record,
                    axis=1,
                )
            )

            if comparison_is_caller:
                comparison_data[
                    "observed_prior_record"
                ] = "Not applicable"
            else:
                comparison_data[
                    "observed_prior_record"
                ] = comparison_data.apply(
                    format_prior_record,
                    axis=1,
                )

            st.caption(
                "Team records follow the selected seasons "
                "and season types. They do not change with "
                "down, quarter, distance, field position, "
                "or score filters."
            )

            comparison_data["coach_team"] = (
                comparison_data["head_coach"]
                + " ("
                + comparison_data["posteam"]
                + ")"
            )

            summary_columns = st.columns(
                len(comparison_data)
            )

            for column, (_, row) in zip(
                summary_columns,
                comparison_data.iterrows(),
            ):
                with column:
                    st.markdown(
                        f"#### {row['head_coach']}"
                    )

                    st.caption(
                        f"{row['posteam']} | "
                        f"{int(row['plays']):,} plays"
                    )

                    if row["experience_season"] == row["experience_season"]:
                        st.markdown(
                            (
                                "**Observed HC experience "
                                f"entering {int(row['experience_season'])}**"
                            )
                        )

                        st.caption(
                            (
                                f"{int(row['observed_prior_hc_seasons'])} "
                                "prior seasons | "
                                f"{int(row['observed_prior_hc_games']):,} "
                                "games | "
                                f"{row['observed_prior_record']} record"
                            )
                        )

                        st.caption(
                            "Scope: NFL head-coaching results observed since 2018."
                        )

                    st.metric(
                        "Pass OE",
                        f"{row['pass_oe_pct']:+.2f} pts",
                    )

                    st.metric(
                        "Team record",
                        row["record"],
                        help=(
                            "Record during the selected "
                            "seasons and season types."
                        ),
                    )

                    if row["win_percentage"] == row["win_percentage"]:
                        st.metric(
                            "Win percentage",
                            (
                                f"{100 * row['win_percentage']:.1f}%"
                            ),
                        )

                    st.metric(
                        "EPA per play",
                        f"{row['mean_epa']:.3f}",
                    )

                    st.metric(
                        "Success rate",
                        (
                            f"{row['success_rate_pct']:.2f}%"
                        ),
                    )

                    if row["point_differential_per_game"] == row["point_differential_per_game"]:
                        st.metric(
                            "Point differential/game",
                            (
                                f"{row['point_differential_per_game']:+.2f}"
                            ),
                        )

            record_display = comparison_data[
                [
                    "head_coach",
                    "posteam",
                    "record_games",
                    "record",
                    "win_percentage",
                    "points_per_game",
                    "points_allowed_per_game",
                    "point_differential_per_game",
                ]
            ].rename(
                columns={
                    "head_coach": comparison_entity_label,
                    "posteam": "Team",
                    "record_games": "Games",
                    "record": "Record",
                    "win_percentage": "PCT",
                    "points_per_game": "Points/Game",
                    "points_allowed_per_game": (
                        "Points Allowed/Game"
                    ),
                    "point_differential_per_game": (
                        "Point Diff/Game"
                    ),
                }
            )

            st.markdown("#### Team Results")

            st.dataframe(
                record_display,
                width="stretch",
                hide_index=True,
            )

            if comparison_is_caller:
                season_type_records = load_team_records(
                    group_by_season_type=True
                )
                season_type_records = (
                    comparison_data[
                        ["head_coach", "posteam"]
                    ]
                    .drop_duplicates()
                    .merge(
                        season_type_records,
                        on="posteam",
                        how="left",
                    )
                )
                season_type_records[
                    "games"
                ] = season_type_records[
                    "record_games"
                ]
            else:
                season_type_records = (
                    load_coach_team_records_by_season_type(
                        coach_override=comparison_entities
                    )
                )

            if not season_type_records.empty:
                season_type_records["record"] = (
                    season_type_records.apply(
                        format_record,
                        axis=1,
                    )
                )

                season_type_records[
                    "season_type"
                ] = season_type_records[
                    "season_type"
                ].replace({
                    "REG": "Regular season",
                    "POST": "Postseason",
                })

                season_type_display = (
                    season_type_records[
                        [
                            "head_coach",
                            "posteam",
                            "season_type",
                            "games",
                            "record",
                            "win_percentage",
                            "points_per_game",
                            "points_allowed_per_game",
                            "point_differential_per_game",
                        ]
                    ]
                    .rename(
                        columns={
                            "head_coach": comparison_entity_label,
                            "posteam": "Team",
                            "season_type": "Season Type",
                            "games": "Games",
                            "record": "Record",
                            "win_percentage": "PCT",
                            "points_per_game": "Points/Game",
                            "points_allowed_per_game": (
                                "Points Allowed/Game"
                            ),
                            "point_differential_per_game": (
                                "Point Diff/Game"
                            ),
                        }
                    )
                )

                with st.expander(
                    "Regular-season and postseason split"
                ):
                    st.dataframe(
                        season_type_display,
                        width="stretch",
                        hide_index=True,
                    )

            pass_rate_data = (
                comparison_data.melt(
                    id_vars=["coach_team"],
                    value_vars=[
                        "actual_pass_rate_pct",
                        "expected_pass_rate_pct",
                    ],
                    var_name="rate_type",
                    value_name="pass_rate",
                )
            )

            pass_rate_data["rate_type"] = (
                pass_rate_data["rate_type"]
                .replace({
                    "actual_pass_rate_pct": (
                        "Actual pass rate"
                    ),
                    "expected_pass_rate_pct": (
                        "Expected pass rate"
                    ),
                })
            )

            pass_rate_figure = px.bar(
                pass_rate_data,
                x="coach_team",
                y="pass_rate",
                color="rate_type",
                barmode="group",
                title=(
                    "Actual vs. Expected Pass Rate"
                ),
                labels={
                    "coach_team": "",
                    "pass_rate": "Pass Rate (%)",
                    "rate_type": "",
                },
            )

            st.plotly_chart(
                pass_rate_figure,
                width="stretch",
            )

            outcome_figure = px.scatter(
                comparison_data,
                x="pass_oe_pct",
                y="mean_epa",
                size="plays",
                color="coach_team",
                hover_name="coach_team",
                title=(
                    "Pass Tendency and EPA"
                ),
                labels={
                    "pass_oe_pct": (
                        "Pass OE "
                        "(percentage points)"
                    ),
                    "mean_epa": "EPA per Play",
                        "coach_team": "Decision-Maker-Team",
                },
            )

            outcome_figure.add_vline(
                x=0,
                line_dash="dash",
                line_color="gray",
            )

            outcome_figure.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
            )

            st.plotly_chart(
                outcome_figure,
                width="stretch",
            )

            if comparison_is_caller:
                st.subheader(
                    "Full-Season Play-Caller Confidence Intervals"
                )

                st.caption(
                    "Intervals use complete caller-team-season "
                    "samples and resample entire games. They do "
                    "not change with situation filters."
                )

                uncertainty_lookup_entities = (
                    comparison_entities
                )
                uncertainty_entity_column = (
                    "offensive_play_caller"
                )
                uncertainty_source = (
                    "play_caller_uncertainty"
                )
                uncertainty_title = (
                    "Play-Caller-Season Pass Tendency with "
                    "95% Confidence Intervals"
                )
            else:
                st.subheader(
                    "Full-Season Confidence Intervals"
                )

                st.caption(
                    "Confidence intervals use complete "
                    "coach-seasons and do not change with "
                    "the situation filters."
                )
                uncertainty_lookup_entities = (
                    comparison_entities
                )
                uncertainty_entity_column = "head_coach"
                uncertainty_source = "coach_uncertainty"
                uncertainty_title = (
                    "Coach-Season Pass Tendency with "
                    "95% Confidence Intervals"
                )

            coach_placeholders = ", ".join(
                ["?"] * len(
                    uncertainty_lookup_entities
                )
            )

            season_placeholders = ", ".join(
                ["?"] * len(
                    selected_seasons
                )
            )

            uncertainty_conditions = [
                (
                    f"{uncertainty_entity_column} IN "
                    f"({coach_placeholders})"
                ),
                "meets_minimum_sample = true",
            ]

            uncertainty_parameters = list(
                uncertainty_lookup_entities
            )

            if selected_seasons:
                uncertainty_conditions.append(
                    "season IN "
                    f"({season_placeholders})"
                )

                uncertainty_parameters.extend(
                    selected_seasons
                )

            uncertainty_where = (
                "WHERE "
                + " AND ".join(
                    uncertainty_conditions
                )
            )

            uncertainty_data = (
                connection.execute(
                    f"""
                    SELECT
                        season,
                        {uncertainty_entity_column}
                            AS head_coach,
                        posteam,
                        plays,
                        model_pass_oe_pct,
                        ci_95_lower_pct,
                        ci_95_upper_pct,
                        tendency_label

                    FROM {uncertainty_source}
                    {uncertainty_where}

                    ORDER BY
                        season,
                        model_pass_oe_pct DESC
                    """,
                    uncertainty_parameters,
                ).fetchdf()
            )

            if not uncertainty_data.empty:
                uncertainty_data["label"] = (
                    uncertainty_data["head_coach"]
                    + " ("
                    + uncertainty_data["posteam"]
                    + ", "
                    + uncertainty_data[
                        "season"
                    ].astype(str)
                    + ")"
                )

                uncertainty_data["error_plus"] = (
                    uncertainty_data[
                        "ci_95_upper_pct"
                    ]
                    - uncertainty_data[
                        "model_pass_oe_pct"
                    ]
                )

                uncertainty_data["error_minus"] = (
                    uncertainty_data[
                        "model_pass_oe_pct"
                    ]
                    - uncertainty_data[
                        "ci_95_lower_pct"
                    ]
                )

                confidence_figure = go.Figure()

                color_map = {
                    "pass-heavy": "#2471A3",
                    "run-heavy": "#C0392B",
                    "uncertain": "#7F8C8D",
                }

                for tendency in [
                    "pass-heavy",
                    "run-heavy",
                    "uncertain",
                ]:
                    subset = uncertainty_data[
                        uncertainty_data[
                            "tendency_label"
                        ]
                        == tendency
                    ]

                    if subset.empty:
                        continue

                    confidence_figure.add_trace(
                        go.Scatter(
                            x=subset[
                                "model_pass_oe_pct"
                            ],
                            y=subset["label"],
                            mode="markers",
                            name=tendency.title(),
                            marker={
                                "size": 10,
                                "color": color_map[
                                    tendency
                                ],
                            },
                            error_x={
                                "type": "data",
                                "array": subset[
                                    "error_plus"
                                ],
                                "arrayminus": subset[
                                    "error_minus"
                                ],
                                "visible": True,
                            },
                        )
                    )

                confidence_figure.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="gray",
                )

                confidence_figure.update_layout(
                    title=uncertainty_title,
                    xaxis_title=(
                        "Pass OE "
                        "(percentage points)"
                    ),
                    yaxis_title="",
                    height=max(
                        500,
                        28 * len(
                            uncertainty_data
                        ),
                    ),
                )

                st.plotly_chart(
                    confidence_figure,
                    width="stretch",
                )


# Historical trends tab
with trends_tab:
    st.subheader("Historical Tendencies")

    trend_attribution = st.radio(
        "Attribute historical decisions to",
        options=[
            "Head coach",
            "Verified offensive play caller",
        ],
        horizontal=True,
        key="trend_attribution",
    )

    trends_are_callers = (
        trend_attribution
        == "Verified offensive play caller"
    )

    trend_entity_column = (
        "offensive_play_caller"
        if trends_are_callers
        else "head_coach"
    )

    trend_source = (
        "play_predictions_with_callers"
        if trends_are_callers
        else "play_predictions"
    )

    trend_entity_label = (
        "Offensive Play Caller"
        if trends_are_callers
        else "Head Coach"
    )

    if trends_are_callers:
        default_trend_callers = [
            caller
            for caller in [
                "Andy Reid",
                "Sean McVay",
            ]
            if caller in filter_options["play_callers"]
        ]

        selected_trend_callers = st.multiselect(
            "Select offensive play callers to highlight",
            options=filter_options["play_callers"],
            default=default_trend_callers,
            key="historical_play_callers",
        )
    else:
        selected_trend_callers = None

    trend_where_clause, trend_query_parameters = (
        build_filter_query(
            coach_column=trend_entity_column,
            use_sidebar_coaches=(
                not trends_are_callers
            )
        )
    )

    trends = connection.execute(
        f"""
        SELECT
            season,
            {trend_entity_column} AS head_coach,
            posteam,
            COUNT(*) AS plays,
            COUNT(DISTINCT game_id) AS sample_games,

            ROUND(
                100 * AVG(model_pass_oe),
                2
            ) AS pass_oe_pct,

            ROUND(
                AVG(epa),
                4
            ) AS mean_epa,

            ROUND(
                100 * AVG(success),
                2
            ) AS success_rate_pct

        FROM {trend_source}
        {trend_where_clause}

        GROUP BY
            season,
            {trend_entity_column},
            posteam

        HAVING COUNT(*) >= ?

        ORDER BY
            season,
            head_coach
        """,
        trend_query_parameters + [minimum_plays],
    ).fetchdf()

    selected_period = connection.execute(
        f"""
        SELECT
            {trend_entity_column} AS decision_maker,
            posteam,
            MIN(season) AS first_season,
            MAX(season) AS last_season,
            COUNT(DISTINCT season) AS seasons_represented,
            COUNT(DISTINCT game_id) AS games,
            COUNT(*) AS plays,
            ROUND(
                100 * AVG(model_pass_oe),
                2
            ) AS pass_oe_pct,
            ROUND(
                AVG(epa),
                4
            ) AS mean_epa,
            ROUND(
                100 * AVG(success),
                2
            ) AS success_rate_pct
        FROM {trend_source}
        {trend_where_clause}
        GROUP BY
            {trend_entity_column},
            posteam
        HAVING COUNT(*) >= ?
        ORDER BY
            pass_oe_pct DESC,
            plays DESC
        """,
        trend_query_parameters + [minimum_plays],
    ).fetchdf()

    if not selected_period.empty:
        selected_period["sample_quality"] = (
            selected_period["plays"].apply(
                lambda plays: (
                    "Stronger sample (500+)"
                    if plays >= 500
                    else (
                        "Limited sample (200-499)"
                        if plays >= 200
                        else "Small exploratory sample (<200)"
                    )
                )
            )
        )

    st.markdown("### Selected-Period Aggregate")
    st.caption(
        "This table pools the selected seasons into one sample for "
        "each decision-maker and team. It answers a different "
        "question from the season-by-season chart below and can "
        "provide a larger sample for down-specific analysis."
    )

    if selected_period.empty:
        st.warning(
            "No pooled selected-period samples meet the current "
            f"{minimum_plays:,}-play minimum."
        )
    else:
        selected_period_display = selected_period.rename(
            columns={
                "decision_maker": trend_entity_label,
                "posteam": "Team",
                "first_season": "First Season",
                "last_season": "Last Season",
                "seasons_represented": "Seasons",
                "games": "Games",
                "plays": "Plays",
                "pass_oe_pct": "Pass OE",
                "mean_epa": "EPA/Play",
                "success_rate_pct": "Success %",
                "sample_quality": "Sample Quality",
            }
        )

        st.dataframe(
            selected_period_display,
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Season-by-Season Trends")
    st.caption(
        "Each point below is a separate decision-maker-team-season "
        "sample and must independently meet the selected play minimum."
    )

    if trends_are_callers:
        trend_records = load_team_records(
            group_by_season=True
        )
        trend_merge_keys = [
            "season",
            "posteam",
        ]
    else:
        trend_records = (
            load_coach_team_records_by_season()
        )
        trend_merge_keys = [
            "season",
            "head_coach",
            "posteam",
        ]

    trends = trends.merge(
        trend_records,
        on=trend_merge_keys,
        how="left",
    )

    if trends_are_callers:
        trends["first_week"] = None
        trends["started_midseason"] = False
        trends["observed_prior_hc_seasons"] = None
        trends["observed_prior_hc_games"] = None
        trends["observed_prior_hc_wins"] = None
        trends["observed_prior_hc_losses"] = None
        trends["observed_prior_hc_ties"] = None
    else:
        tenure_history = connection.execute(
            """
            SELECT
                season,
                head_coach,
                team AS posteam,
                first_week,
                last_week,
                started_midseason,
                observed_prior_hc_seasons,
                observed_prior_hc_games,
                observed_prior_hc_wins,
                observed_prior_hc_losses,
                observed_prior_hc_ties,
                observed_prior_hc_win_percentage,
                first_observed_hc_season,
                experience_scope
            FROM coach_tenures
            """
        ).fetchdf()

        trends = trends.merge(
            tenure_history,
            on=[
                "season",
                "head_coach",
                "posteam",
            ],
            how="left",
        )

    if trends.empty:
        st.warning(
            "No historical samples meet the "
            "selected requirements."
        )
    else:
        trends["sample_quality"] = trends["plays"].apply(
            lambda plays: (
                "Stronger sample (500+)"
                if plays >= 500
                else (
                    "Limited sample (200-499)"
                    if plays >= 200
                    else "Small exploratory sample (<200)"
                )
            )
        )

        trends["record"] = trends.apply(
            format_record,
            axis=1,
        )

        if trends_are_callers:
            trends["observed_prior_record"] = (
                "Not applicable"
            )
            trends["tenure_start"] = (
                "See verified caller segment"
            )
        else:
            trends["observed_prior_record"] = (
                trends.apply(
                    format_prior_record,
                    axis=1,
                )
            )

            trends["tenure_start"] = trends.apply(
                lambda row: (
                    f"Week {int(row['first_week'])}"
                    if row["started_midseason"]
                    else "Season start"
                ),
                axis=1,
            )

        trends["coach_team"] = (
            trends["head_coach"]
            + " ("
            + trends["posteam"]
            + ")"
        )

        if trends_are_callers:
            if selected_trend_callers:
                chart_trends = trends[
                    trends["head_coach"].isin(
                        selected_trend_callers
                    )
                ].copy()
            else:
                chart_trends = trends.iloc[0:0].copy()
        else:
            chart_trends = trends.copy()

        # Treat seasons as discrete labels in charts. This prevents
        # Plotly from displaying meaningless half-year values such as
        # 2024.5 or 2025.5 when only one or two seasons are selected.
        chart_trends["season_label"] = (
            chart_trends["season"].astype(str)
        )

        if not selected_coaches and not trends_are_callers:
            st.info(
                "Select coaches in the sidebar "
                "for a clearer chart."
            )

        if (
            trends_are_callers
            and not selected_trend_callers
        ):
            st.info(
                "Select offensive play callers above for a "
                "clearer historical chart."
            )

        trend_figure = px.line(
            chart_trends,
            x="season",
            y="pass_oe_pct",
            color="coach_team",
            markers=True,
            hover_name="coach_team",
            hover_data={
                "posteam": True,
                "plays": ":,",
                "sample_games": ":,",
                "pass_oe_pct": ":+.2f",
                "mean_epa": ":.3f",
                "success_rate_pct": ":.2f",
                "sample_quality": True,
            },
            title=(
                "Pass Rate Over Expected by Season"
            ),
            labels={
                "season": "Season",
                "pass_oe_pct": "Pass OE",
                "coach_team": "Decision-Maker-Team",
                "posteam": "Team",
                "sample_games": "Sample Games",
                "sample_quality": "Sample Quality",
            },
        )

        trend_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        trend_figure.update_xaxes(
            tickmode="linear",
            dtick=1,
            tickformat="d",
        )

        st.caption(
            "Interactive charts: hover over any point to identify "
            "the decision-maker, team, season, sample size, Pass OE, "
            "EPA per play, success rate, and sample-quality label."
        )

        st.plotly_chart(
            trend_figure,
            width="stretch",
        )

        scatter_figure = px.scatter(
            chart_trends,
            x="pass_oe_pct",
            y="mean_epa",
            color="season_label",
            hover_name="coach_team",
            size="plays",
            hover_data={
                "posteam": True,
                "plays": ":,",
                "sample_games": ":,",
                "pass_oe_pct": ":+.2f",
                "mean_epa": ":.3f",
                "success_rate_pct": ":.2f",
                "sample_quality": True,
            },
            title=(
                "Play-Calling Tendency and EPA"
            ),
            labels={
                "pass_oe_pct": (
                    "Pass OE (percentage points)"
                ),
                "mean_epa": "EPA per Play",
                "season_label": "Season",
                "coach_team": (
                    "Decision-Maker-Team"
                ),
                "plays": "Plays",
                "posteam": "Team",
                "sample_games": "Sample Games",
                "sample_quality": "Sample Quality",
            },
        )

        scatter_figure.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
        )

        scatter_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        st.plotly_chart(
            scatter_figure,
            width="stretch",
        )

        st.markdown(
            (
                "#### Play-Caller-Season Results Context"
                if trends_are_callers
                else "#### Coach-Season Results Context"
            )
        )

        st.caption(
            "Records and scoring figures follow season "
            "and season-type filters, but not situation "
            "filters. Play-calling metrics use all "
            "selected filters."
        )

        trend_display_columns = [
            "season",
            "head_coach",
            "posteam",
            "plays",
            "sample_quality",
            "record_games",
            "record",
            "win_percentage",
        ]

        if not trends_are_callers:
            trend_display_columns.extend([
                "observed_prior_hc_seasons",
                "observed_prior_hc_games",
                "observed_prior_record",
                "tenure_start",
            ])

        trend_display_columns.extend([
            "pass_oe_pct",
            "mean_epa",
            "success_rate_pct",
            "points_per_game",
            "points_allowed_per_game",
            "point_differential_per_game",
        ])

        trend_display = trends[
            trend_display_columns
        ].rename(
            columns={
                "season": "Season",
                "head_coach": trend_entity_label,
                "posteam": "Team",
                "plays": "Plays",
                "sample_quality": "Sample Quality",
                "record_games": "Games",
                "record": "Record",
                "win_percentage": "PCT",
                "observed_prior_hc_seasons": (
                    "Prior HC Seasons Since 2018"
                ),
                "observed_prior_hc_games": (
                    "Prior HC Games Since 2018"
                ),
                "observed_prior_record": (
                    "Prior HC Record Since 2018"
                ),
                "tenure_start": "Tenure Start",
                "pass_oe_pct": "Pass OE",
                "mean_epa": "EPA/Play",
                "success_rate_pct": "Success %",
                "points_per_game": "Points/Game",
                "points_allowed_per_game": (
                    "Points Allowed/Game"
                ),
                "point_differential_per_game": (
                    "Point Diff/Game"
                ),
            }
        )

        st.dataframe(
            trend_display,
            width="stretch",
            hide_index=True,
        )


# Situation Lab tab
with simulator_tab:
    st.subheader("Call Sheet Situation Lab")

    st.caption(
        "Build a game situation and compare league behavior with up to "
        "two head coaches or verified offensive play callers. Results "
        "come from comparable historical plays from 2018-2025; they are "
        "descriptive tendencies, not play recommendations."
    )

    situation_presets = {
        "Custom": {
            "down": 3,
            "ydstogo": 7,
            "yardline_100": 38,
            "qtr": 2,
            "quarter_minutes": 8,
            "quarter_seconds": 0,
            "score_differential": 0,
            "goal_to_go": False,
        },
        "Opening-drive first down": {
            "down": 1,
            "ydstogo": 10,
            "yardline_100": 75,
            "qtr": 1,
            "quarter_minutes": 13,
            "quarter_seconds": 0,
            "score_differential": 0,
            "goal_to_go": False,
        },
        "Third-and-short": {
            "down": 3,
            "ydstogo": 2,
            "yardline_100": 45,
            "qtr": 2,
            "quarter_minutes": 10,
            "quarter_seconds": 0,
            "score_differential": 0,
            "goal_to_go": False,
        },
        "Third-and-seven in plus territory": {
            "down": 3,
            "ydstogo": 7,
            "yardline_100": 38,
            "qtr": 2,
            "quarter_minutes": 8,
            "quarter_seconds": 0,
            "score_differential": 0,
            "goal_to_go": False,
        },
        "Fourth-and-one decision": {
            "down": 4,
            "ydstogo": 1,
            "yardline_100": 42,
            "qtr": 4,
            "quarter_minutes": 5,
            "quarter_seconds": 0,
            "score_differential": -3,
            "goal_to_go": False,
        },
        "Red-zone second down": {
            "down": 2,
            "ydstogo": 7,
            "yardline_100": 15,
            "qtr": 2,
            "quarter_minutes": 7,
            "quarter_seconds": 0,
            "score_differential": 0,
            "goal_to_go": False,
        },
        "Two-minute comeback": {
            "down": 1,
            "ydstogo": 10,
            "yardline_100": 70,
            "qtr": 4,
            "quarter_minutes": 1,
            "quarter_seconds": 30,
            "score_differential": -4,
            "goal_to_go": False,
        },
        "Protecting a late lead": {
            "down": 2,
            "ydstogo": 7,
            "yardline_100": 60,
            "qtr": 4,
            "quarter_minutes": 6,
            "quarter_seconds": 0,
            "score_differential": 7,
            "goal_to_go": False,
        },
    }

    selected_preset_name = st.selectbox(
        "Start with a familiar situation",
        options=list(situation_presets),
        index=3,
        key="situation_lab_preset",
    )

    selected_preset = situation_presets[
        selected_preset_name
    ]

    scenario_row_one = st.columns(4)

    scenario_down = scenario_row_one[0].selectbox(
        "Down",
        options=[1, 2, 3, 4],
        index=(selected_preset["down"] - 1),
        key=(
            "situation_down_"
            + selected_preset_name
        ),
    )

    scenario_distance = scenario_row_one[1].number_input(
        "Yards to go",
        min_value=1,
        max_value=40,
        value=selected_preset["ydstogo"],
        step=1,
        key=(
            "situation_distance_"
            + selected_preset_name
        ),
    )

    scenario_yardline = scenario_row_one[2].slider(
        "Yards from opponent end zone",
        min_value=1,
        max_value=99,
        value=selected_preset["yardline_100"],
        key=(
            "situation_yardline_"
            + selected_preset_name
        ),
    )

    scenario_goal_to_go = scenario_row_one[3].checkbox(
        "Goal to go",
        value=selected_preset["goal_to_go"],
        key=(
            "situation_goal_to_go_"
            + selected_preset_name
        ),
    )

    scenario_row_two = st.columns(4)

    scenario_quarter = scenario_row_two[0].selectbox(
        "Quarter",
        options=[1, 2, 3, 4],
        index=(selected_preset["qtr"] - 1),
        key=(
            "situation_quarter_"
            + selected_preset_name
        ),
    )

    scenario_minutes = scenario_row_two[1].number_input(
        "Minutes remaining in quarter",
        min_value=0,
        max_value=15,
        value=selected_preset["quarter_minutes"],
        step=1,
        key=(
            "situation_minutes_"
            + selected_preset_name
        ),
    )

    scenario_seconds = scenario_row_two[2].number_input(
        "Additional seconds",
        min_value=0,
        max_value=59,
        value=selected_preset["quarter_seconds"],
        step=5,
        key=(
            "situation_seconds_"
            + selected_preset_name
        ),
    )

    scenario_score = scenario_row_two[3].slider(
        "Offense score differential",
        min_value=-28,
        max_value=28,
        value=selected_preset["score_differential"],
        key=(
            "situation_score_"
            + selected_preset_name
        ),
    )

    simulator_scope_columns = st.columns(2)

    simulator_seasons = simulator_scope_columns[0].multiselect(
        "Historical seasons to search",
        options=filter_options["seasons"],
        default=filter_options["seasons"],
        key="situation_lab_seasons",
    )

    simulator_season_types = simulator_scope_columns[1].multiselect(
        "Season types",
        options=["REG", "POST"],
        default=["REG", "POST"],
        format_func=lambda value: (
            "Regular season"
            if value == "REG"
            else "Postseason"
        ),
        key="situation_lab_season_types",
    )

    simulator_attribution = st.radio(
        "Compare decision-makers as",
        options=[
            "Head coach",
            "Verified offensive play caller",
        ],
        horizontal=True,
        key="situation_lab_attribution",
    )

    simulator_uses_callers = (
        simulator_attribution
        == "Verified offensive play caller"
    )

    simulator_entity_column = (
        "offensive_play_caller"
        if simulator_uses_callers
        else "head_coach"
    )

    simulator_entity_label = (
        "Offensive Play Caller"
        if simulator_uses_callers
        else "Head Coach"
    )

    simulator_entity_options = (
        filter_options["play_callers"]
        if simulator_uses_callers
        else filter_options["coaches"]
    )

    preferred_simulator_entities = [
        entity
        for entity in [
            "Andy Reid",
            "Sean McVay",
        ]
        if entity in simulator_entity_options
    ]

    simulator_entity_plural = (
        "head coaches"
        if simulator_entity_column == "head_coach"
        else "verified offensive play callers"
    )

    simulator_entities = st.multiselect(
        f"Compare up to two {simulator_entity_plural}",
        options=simulator_entity_options,
        default=preferred_simulator_entities,
        max_selections=2,
        key=(
            "situation_lab_entities_"
            + simulator_entity_column
        ),
    )

    if not simulator_seasons:
        simulator_seasons = filter_options["seasons"]
        st.info(
            "No seasons were selected, so all available seasons "
            "are being searched."
        )

    if not simulator_season_types:
        simulator_season_types = ["REG", "POST"]
        st.info(
            "No season type was selected, so regular-season and "
            "postseason plays are both included."
        )

    scenario_quarter_seconds = (
        int(scenario_minutes) * 60
        + int(scenario_seconds)
    )

    if scenario_quarter == 1:
        scenario_game_seconds = (
            2700 + scenario_quarter_seconds
        )
    elif scenario_quarter == 2:
        scenario_game_seconds = (
            1800 + scenario_quarter_seconds
        )
    elif scenario_quarter == 3:
        scenario_game_seconds = (
            900 + scenario_quarter_seconds
        )
    else:
        scenario_game_seconds = scenario_quarter_seconds

    def simulator_distance_bounds(distance):
        if distance <= 3:
            return 1, 3, "Short (1-3)"
        if distance <= 6:
            return 4, 6, "Medium (4-6)"
        if distance <= 10:
            return 7, 10, "Long (7-10)"
        return 11, 40, "Very long (11+)"

    def simulator_field_bounds(yardline):
        if yardline <= 20:
            return 1, 20, "Red zone"
        if yardline <= 50:
            return 21, 50, "Plus territory"
        if yardline <= 80:
            return 51, 80, "Own territory"
        return 81, 99, "Backed up"

    def simulator_score_bounds(score):
        if score <= -9:
            return -99, -9, "Trailing by 9+"
        if score <= -1:
            return -8, -1, "Trailing by 1-8"
        if score == 0:
            return 0, 0, "Tied"
        if score <= 8:
            return 1, 8, "Leading by 1-8"
        return 9, 99, "Leading by 9+"

    def build_simulator_filter(match_tier):
        season_placeholders = ", ".join(
            ["?"] * len(simulator_seasons)
        )
        type_placeholders = ", ".join(
            ["?"] * len(simulator_season_types)
        )

        conditions = [
            f"season IN ({season_placeholders})",
            f"season_type IN ({type_placeholders})",
            "down = ?",
            "goal_to_go = ?",
        ]

        parameters = [
            *simulator_seasons,
            *simulator_season_types,
            int(scenario_down),
            int(scenario_goal_to_go),
        ]

        (
            distance_low,
            distance_high,
            distance_label,
        ) = simulator_distance_bounds(
            int(scenario_distance)
        )

        (
            field_low,
            field_high,
            field_label,
        ) = simulator_field_bounds(
            int(scenario_yardline)
        )

        (
            score_low,
            score_high,
            score_label,
        ) = simulator_score_bounds(
            int(scenario_score)
        )

        if match_tier == "Strict":
            conditions.extend([
                "ydstogo BETWEEN ? AND ?",
                "yardline_100 BETWEEN ? AND ?",
                "qtr = ?",
                "game_seconds_remaining BETWEEN ? AND ?",
                "score_differential BETWEEN ? AND ?",
            ])
            parameters.extend([
                max(1, int(scenario_distance) - 1),
                int(scenario_distance) + 1,
                max(1, int(scenario_yardline) - 10),
                min(99, int(scenario_yardline) + 10),
                int(scenario_quarter),
                max(0, scenario_game_seconds - 180),
                min(3600, scenario_game_seconds + 180),
                int(scenario_score) - 3,
                int(scenario_score) + 3,
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
                int(scenario_quarter),
                max(0, scenario_game_seconds - 300),
                min(3600, scenario_game_seconds + 300),
                score_low,
                score_high,
            ])
        else:
            half_quarters = (
                [1, 2]
                if scenario_quarter in (1, 2)
                else [3, 4, 5]
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

        return (
            "WHERE " + " AND ".join(conditions),
            parameters,
            {
                "distance": distance_label,
                "field": field_label,
                "score": score_label,
            },
        )

    def load_situation_summary(
        match_tier,
        entity=None,
    ):
        (
            situation_where,
            situation_parameters,
            match_labels,
        ) = build_simulator_filter(
            match_tier
        )

        if entity is not None:
            situation_where += (
                f" AND {simulator_entity_column} = ?"
            )
            situation_parameters.append(entity)

        summary = connection.execute(
            f"""
            SELECT
                COUNT(*) AS plays,
                COUNT(DISTINCT game_id) AS games,
                COUNT(DISTINCT season) AS seasons,
                STRING_AGG(
                    DISTINCT posteam,
                    ', '
                    ORDER BY posteam
                ) AS teams,
                ROUND(100 * AVG(is_pass), 2)
                    AS actual_pass_rate_pct,
                ROUND(
                    100 * AVG(expected_pass_probability),
                    2
                ) AS expected_pass_rate_pct,
                ROUND(100 * AVG(model_pass_oe), 2)
                    AS pass_oe_pct,
                ROUND(AVG(epa), 4) AS mean_epa,
                ROUND(100 * AVG(success), 2)
                    AS success_rate_pct,
                ROUND(100 * AVG(shotgun), 2)
                    AS shotgun_pct,
                ROUND(100 * AVG(no_huddle), 2)
                    AS no_huddle_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Behind line'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS behind_line_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Short (0-9)'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS short_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Intermediate (10-19)'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS intermediate_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Deep (20+)'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS deep_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_direction = 'Middle'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_direction IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS middle_pass_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN run_direction IN ('Left', 'Right')
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN run_direction IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS outside_run_pct
            FROM play_style_predictions_with_callers
            {situation_where}
            """,
            situation_parameters,
        ).fetchdf().iloc[0].to_dict()

        summary["match_tier"] = match_tier
        summary["match_labels"] = match_labels
        summary["entity"] = (
            entity
            if entity is not None
            else "League"
        )

        return summary

    def choose_league_summary():
        summaries = [
            load_situation_summary(tier)
            for tier in [
                "Strict",
                "Balanced",
                "Broad",
            ]
        ]

        for summary in summaries:
            if int(summary["plays"]) >= 400:
                return summary

        return summaries[-1]

    def choose_entity_summary(entity):
        summaries = [
            load_situation_summary(
                tier,
                entity=entity,
            )
            for tier in [
                "Strict",
                "Balanced",
                "Broad",
            ]
        ]

        for summary in summaries:
            if int(summary["plays"]) >= 50:
                summary["sample_label"] = "Usable (50+)"
                return summary

        for summary in summaries:
            if int(summary["plays"]) >= 25:
                summary["sample_label"] = "Limited (25-49)"
                return summary

        summary = summaries[-1]
        summary["sample_label"] = "Insufficient (<25)"
        return summary

    league_situation = choose_league_summary()

    entity_situations = [
        choose_entity_summary(entity)
        for entity in simulator_entities
    ]

    st.divider()

    st.markdown("### Situation Snapshot")

    clock_text = (
        f"{int(scenario_minutes)}:"
        f"{int(scenario_seconds):02d}"
    )

    field_text = (
        f"opponent {int(scenario_yardline)}"
        if scenario_yardline <= 50
        else f"own {100 - int(scenario_yardline)}"
    )

    score_text = (
        "tied"
        if scenario_score == 0
        else (
            f"leading by {int(scenario_score)}"
            if scenario_score > 0
            else f"trailing by {abs(int(scenario_score))}"
        )
    )

    st.info(
        f"{int(scenario_down)} & {int(scenario_distance)} at the "
        f"{field_text}, Q{int(scenario_quarter)} {clock_text}, "
        f"offense {score_text}."
    )

    league_metrics = st.columns(5)

    league_metrics[0].metric(
        "Comparable league plays",
        f"{int(league_situation['plays']):,}",
    )
    league_metrics[1].metric(
        "Modeled pass expectation",
        (
            f"{league_situation['expected_pass_rate_pct']:.1f}%"
            if pd.notna(
                league_situation[
                    "expected_pass_rate_pct"
                ]
            )
            else "N/A"
        ),
    )
    league_metrics[2].metric(
        "Actual league pass rate",
        (
            f"{league_situation['actual_pass_rate_pct']:.1f}%"
            if pd.notna(
                league_situation[
                    "actual_pass_rate_pct"
                ]
            )
            else "N/A"
        ),
    )
    league_metrics[3].metric(
        "League Pass OE",
        (
            f"{league_situation['pass_oe_pct']:+.1f} pts"
            if pd.notna(
                league_situation["pass_oe_pct"]
            )
            else "N/A"
        ),
    )
    league_metrics[4].metric(
        "League match tier",
        league_situation["match_tier"],
    )

    st.caption(
        "The modeled expectation is the average model probability "
        "among the comparable historical plays, not a recommendation "
        "or a causal estimate for the next snap."
    )

    if not simulator_entities:
        st.warning(
            "Select one or two decision-makers to compare their "
            "historical tendencies with the league sample."
        )
    else:
        st.markdown("### Decision-Maker Comparison")

        comparison_rows = []

        comparison_rows.append({
            "Decision-Maker": "League",
            "Teams": "All teams",
            "Match Tier": league_situation["match_tier"],
            "Sample Quality": (
                "League reference"
            ),
            "Plays": int(league_situation["plays"]),
            "Games": int(league_situation["games"]),
            "Actual Pass %": league_situation[
                "actual_pass_rate_pct"
            ],
            "Expected Pass %": league_situation[
                "expected_pass_rate_pct"
            ],
            "Pass OE": league_situation["pass_oe_pct"],
            "EPA/Play": league_situation["mean_epa"],
            "Success %": league_situation[
                "success_rate_pct"
            ],
            "Shotgun %": league_situation["shotgun_pct"],
            "No-Huddle %": league_situation[
                "no_huddle_pct"
            ],
            "Behind-Line Throw % (<0 air yards)": league_situation[
                "behind_line_throw_pct"
            ],
            "Short Throw % (0-9 air yards)": league_situation[
                "short_throw_pct"
            ],
            "Intermediate Throw % (10-19 air yards)": league_situation[
                "intermediate_throw_pct"
            ],
            "Deep Throw % (20+ air yards)": league_situation[
                "deep_throw_pct"
            ],
            "Middle-Field Throw %": league_situation[
                "middle_pass_pct"
            ],
            "Outside Run %": league_situation[
                "outside_run_pct"
            ],
        })

        for entity_summary in entity_situations:
            comparison_rows.append({
                "Decision-Maker": entity_summary["entity"],
                "Teams": entity_summary["teams"],
                "Match Tier": entity_summary["match_tier"],
                "Sample Quality": entity_summary["sample_label"],
                "Plays": int(entity_summary["plays"]),
                "Games": int(entity_summary["games"]),
                "Actual Pass %": entity_summary[
                    "actual_pass_rate_pct"
                ],
                "Expected Pass %": entity_summary[
                    "expected_pass_rate_pct"
                ],
                "Pass OE": entity_summary["pass_oe_pct"],
                "EPA/Play": entity_summary["mean_epa"],
                "Success %": entity_summary[
                    "success_rate_pct"
                ],
                "Shotgun %": entity_summary["shotgun_pct"],
                "No-Huddle %": entity_summary[
                    "no_huddle_pct"
                ],
                "Behind-Line Throw % (<0 air yards)": entity_summary[
                    "behind_line_throw_pct"
                ],
                "Short Throw % (0-9 air yards)": entity_summary[
                    "short_throw_pct"
                ],
                "Intermediate Throw % (10-19 air yards)": entity_summary[
                    "intermediate_throw_pct"
                ],
                "Deep Throw % (20+ air yards)": entity_summary[
                    "deep_throw_pct"
                ],
                "Middle-Field Throw %": entity_summary[
                    "middle_pass_pct"
                ],
                "Outside Run %": entity_summary[
                    "outside_run_pct"
                ],
            })

        comparison_table = pd.DataFrame(
            comparison_rows
        )

        thin_entities = [
            summary["entity"]
            for summary in entity_situations
            if int(summary["plays"]) < 25
        ]

        limited_entities = [
            summary["entity"]
            for summary in entity_situations
            if 25 <= int(summary["plays"]) < 50
        ]

        if thin_entities:
            st.warning(
                "Insufficient comparable history for: "
                + ", ".join(thin_entities)
                + ". Their values are displayed for transparency "
                "but should not be treated as stable tendencies."
            )

        if limited_entities:
            st.info(
                "Limited 25-49 play samples for: "
                + ", ".join(limited_entities)
                + ". Interpret differences cautiously."
            )

        pass_rate_chart_data = comparison_table.melt(
            id_vars=["Decision-Maker"],
            value_vars=[
                "Actual Pass %",
                "Expected Pass %",
            ],
            var_name="Measure",
            value_name="Rate",
        )

        pass_rate_figure = px.bar(
            pass_rate_chart_data,
            x="Decision-Maker",
            y="Rate",
            color="Measure",
            barmode="group",
            title="Actual vs. Modeled Pass Rate",
            labels={
                "Rate": "Pass Rate (%)",
            },
            text_auto=".1f",
        )

        st.plotly_chart(
            pass_rate_figure,
            width="stretch",
        )

        fingerprint_data = comparison_table.melt(
            id_vars=["Decision-Maker"],
            value_vars=[
                "Shotgun %",
                "No-Huddle %",
                "Behind-Line Throw % (<0 air yards)",
                "Short Throw % (0-9 air yards)",
                "Intermediate Throw % (10-19 air yards)",
                "Deep Throw % (20+ air yards)",
                "Middle-Field Throw %",
                "Outside Run %",
            ],
            var_name="Style Dimension",
            value_name="Rate",
        )

        fingerprint_figure = px.bar(
            fingerprint_data,
            x="Style Dimension",
            y="Rate",
            color="Decision-Maker",
            barmode="group",
            title="Comparable-Situation Style Fingerprint",
            labels={
                "Rate": "Rate (%)",
            },
        )

        st.plotly_chart(
            fingerprint_figure,
            width="stretch",
        )

        st.caption(
            "Throw-depth shares use charted throws with recorded air "
            "yards: behind the line is below 0, short is 0-9, "
            "intermediate is 10-19, and deep is 20+ air yards. "
            "Middle-field throw describes target direction, not depth. "
            "Sacks and scrambles remain pass calls in the run-pass model "
            "but are excluded from throw-depth and direction shares."
        )

        st.dataframe(
            comparison_table,
            width="stretch",
            hide_index=True,
        )

    with st.expander("How adaptive matching works"):
        st.markdown(
            """
            - **Strict:** exact down and quarter; distance within one
              yard; field position within ten yards; clock within
              three minutes; score within three points.
            - **Balanced:** exact down and quarter; the same distance,
              field-position, and score-state buckets; clock within
              five minutes.
            - **Broad:** exact down; the same distance, field-position,
              score-state, and half buckets; no exact clock window.

            The league reference uses the narrowest tier with at least
            400 plays. Each decision-maker uses the narrowest tier with
            at least 50 plays, then 25 plays if necessary. If even the
            broad tier has fewer than 25 plays, the app explicitly labels
            the sample insufficient. Different rows can therefore use
            different match tiers, which are always shown in the table.
            """
        )


# Make the Call tab
with challenge_tab:
    st.subheader("You Make the Call")

    st.caption(
        "Read the situation, commit to run or pass, and then reveal "
        "how comparable NFL plays were called from 2018-2025. Your "
        "score measures whether you identified the historical majority "
        "call. It does not grade which call would be strategically best."
    )

    challenge_setup_columns = st.columns([1, 1, 1])

    challenge_difficulty = challenge_setup_columns[0].selectbox(
        "Difficulty",
        options=[
            "Rookie",
            "Pro",
            "Coordinator",
            "Mixed",
        ],
        index=1,
        help=(
            "Rookie emphasizes clearer historical tendencies. Pro uses "
            "moderately difficult situations. Coordinator emphasizes "
            "near-even historical call splits. Mixed uses every tier."
        ),
        key="make_call_difficulty_setting",
    )

    challenge_length = challenge_setup_columns[1].selectbox(
        "Challenges per session",
        options=[10, 15, 20, 25],
        index=1,
        key="make_call_length_setting",
    )

    challenge_minimum_plays = challenge_setup_columns[2].selectbox(
        "Minimum comparable plays",
        options=[200, 300, 500],
        index=1,
        help=(
            "Every challenge deck is built only from historical situation "
            "buckets meeting this league sample requirement."
        ),
        key="make_call_minimum_setting",
    )

    def ordinal_down(value):
        return {
            1: "First",
            2: "Second",
            3: "Third",
            4: "Fourth",
        }[int(value)]

    def generate_challenge_deck(
        difficulty,
        requested_length,
        minimum_plays,
    ):
        candidate_pool = connection.execute(
            """
            SELECT
                CAST(down AS INTEGER) AS down,
                CAST(goal_to_go AS INTEGER) AS goal_to_go,
                CASE
                    WHEN ydstogo <= 3 THEN 'short'
                    WHEN ydstogo <= 6 THEN 'medium'
                    WHEN ydstogo <= 10 THEN 'long'
                    ELSE 'very_long'
                END AS distance_bucket,
                CASE
                    WHEN yardline_100 <= 20 THEN 'red_zone'
                    WHEN yardline_100 <= 50 THEN 'plus_territory'
                    WHEN yardline_100 <= 80 THEN 'own_territory'
                    ELSE 'backed_up'
                END AS field_bucket,
                CASE
                    WHEN qtr IN (1, 2) THEN 1
                    ELSE 2
                END AS half_bucket,
                CASE
                    WHEN score_differential <= -9 THEN 'trail_big'
                    WHEN score_differential <= -1 THEN 'trail_close'
                    WHEN score_differential = 0 THEN 'tied'
                    WHEN score_differential <= 8 THEN 'lead_close'
                    ELSE 'lead_big'
                END AS score_bucket,
                COUNT(*) AS historical_plays,
                100 * AVG(is_pass) AS historical_pass_rate
            FROM play_style_predictions_with_callers
            WHERE
                season BETWEEN 2018 AND 2025
                AND season_type IN ('REG', 'POST')
                AND down BETWEEN 1 AND 4
                AND ydstogo BETWEEN 1 AND 20
                AND yardline_100 BETWEEN 1 AND 99
            GROUP BY ALL
            HAVING COUNT(*) >= ?
            """,
            [int(minimum_plays)],
        ).fetchdf()

        if candidate_pool.empty:
            return []

        distance_from_even = (
            candidate_pool["historical_pass_rate"] - 50
        ).abs()

        if difficulty == "Rookie":
            eligible_pool = candidate_pool[
                distance_from_even >= 20
            ]
        elif difficulty == "Pro":
            eligible_pool = candidate_pool[
                (distance_from_even >= 8)
                & (distance_from_even < 20)
            ]
        elif difficulty == "Coordinator":
            eligible_pool = candidate_pool[
                distance_from_even < 8
            ]
        else:
            eligible_pool = candidate_pool

        if len(eligible_pool) < requested_length:
            eligible_pool = candidate_pool

        random_source = random.SystemRandom()
        candidate_rows = eligible_pool.to_dict("records")
        random_source.shuffle(candidate_rows)

        distance_values = {
            "short": [1, 2, 3],
            "medium": [4, 5, 6],
            "long": [7, 8, 9, 10],
            "very_long": [11, 12, 15, 18, 20],
        }
        field_values = {
            "red_zone": [8, 12, 15, 18],
            "plus_territory": [25, 32, 38, 45],
            "own_territory": [55, 65, 75],
            "backed_up": [85, 90, 95],
        }
        field_labels = {
            "red_zone": "in the red zone",
            "plus_territory": "in plus territory",
            "own_territory": "in own territory",
            "backed_up": "backed up",
        }
        score_values = {
            "trail_big": [-17, -14, -10],
            "trail_close": [-7, -4, -3],
            "tied": [0],
            "lead_close": [3, 4, 7],
            "lead_big": [10, 14, 17],
        }
        score_labels = {
            "trail_big": "trailing by multiple scores",
            "trail_close": "trailing by one score",
            "tied": "with the score tied",
            "lead_close": "leading by one score",
            "lead_big": "leading by multiple scores",
        }

        generated_deck = []
        used_signatures = set()

        for row in candidate_rows:
            if len(generated_deck) >= requested_length:
                break

            distance_bucket = row["distance_bucket"]
            field_bucket = row["field_bucket"]
            score_bucket = row["score_bucket"]

            distance = random_source.choice(
                distance_values[distance_bucket]
            )
            yardline = random_source.choice(
                field_values[field_bucket]
            )
            score = random_source.choice(
                score_values[score_bucket]
            )

            if int(row["half_bucket"]) == 1:
                quarter = random_source.choice([1, 2])
            else:
                quarter = random_source.choice([3, 4])

            clock_options = (
                [(13, 0), (9, 30), (6, 0), (2, 0), (0, 45)]
                if quarter != 4
                else [(12, 0), (8, 0), (5, 0), (2, 0), (0, 45)]
            )
            minutes, seconds = random_source.choice(clock_options)

            goal_to_go = bool(int(row["goal_to_go"]))
            if goal_to_go:
                yardline = min(yardline, 10)
                # Goal-to-go distance is the distance to the goal line.
                distance = yardline
            elif int(row["down"]) == 1:
                # Keep synthetic first downs football-realistic. Normal
                # first downs are first-and-10; longer first downs represent
                # common penalty distances. Short first downs outside
                # goal-to-go are too unusual for the challenge deck.
                if distance_bucket in ("short", "medium"):
                    continue
                if distance_bucket == "long":
                    distance = 10
                else:
                    distance = random_source.choice([15, 20])

            if not goal_to_go:
                valid_yardlines = [
                    value
                    for value in field_values[field_bucket]
                    if value > distance
                ]
                if not valid_yardlines:
                    continue
                yardline = random_source.choice(valid_yardlines)

            signature = (
                int(row["down"]),
                distance,
                yardline,
                quarter,
                minutes,
                seconds,
                score,
                goal_to_go,
            )
            if signature in used_signatures:
                continue
            used_signatures.add(signature)

            title = (
                f"{ordinal_down(row['down'])}-and-"
                f"{int(distance)} "
                f"{field_labels[field_bucket]}, "
                f"{score_labels[score_bucket]}"
            )

            generated_deck.append({
                "name": title,
                "down": int(row["down"]),
                "ydstogo": int(distance),
                "yardline_100": int(yardline),
                "qtr": int(quarter),
                "quarter_minutes": int(minutes),
                "quarter_seconds": int(seconds),
                "score_differential": int(score),
                "goal_to_go": goal_to_go,
                "source_plays": int(row["historical_plays"]),
                "source_pass_rate": float(
                    row["historical_pass_rate"]
                ),
            })

        random_source.shuffle(generated_deck)
        return generated_deck

    start_new_session = st.button(
        "Start new randomized session",
        type="primary",
        width="stretch",
        key="make_call_new_random_session",
    )

    if (
        "make_call_deck" not in st.session_state
        or not st.session_state.make_call_deck
        or st.session_state.get("make_call_deck_version") != 2
        or start_new_session
    ):
        st.session_state.make_call_deck = generate_challenge_deck(
            challenge_difficulty,
            int(challenge_length),
            int(challenge_minimum_plays),
        )
        st.session_state.make_call_index = 0
        st.session_state.make_call_score = 0
        st.session_state.make_call_attempts = 0
        st.session_state.make_call_revealed = False
        st.session_state.make_call_choice = None
        st.session_state.make_call_session_difficulty = (
            challenge_difficulty
        )
        st.session_state.make_call_session_minimum = int(
            challenge_minimum_plays
        )
        st.session_state.make_call_deck_version = 2

    if not st.session_state.make_call_deck:
        st.error(
            "No historical situation buckets meet these session "
            "requirements. Lower the minimum comparable-play setting "
            "and start a new session."
        )
        st.stop()

    challenge_deck = st.session_state.make_call_deck
    challenge = challenge_deck[
        st.session_state.make_call_index
    ]
    challenge_name = challenge["name"]

    challenge_clock_seconds = (
        int(challenge["quarter_minutes"]) * 60
        + int(challenge["quarter_seconds"])
    )

    if int(challenge["qtr"]) == 1:
        challenge_game_seconds = 2700 + challenge_clock_seconds
    elif int(challenge["qtr"]) == 2:
        challenge_game_seconds = 1800 + challenge_clock_seconds
    elif int(challenge["qtr"]) == 3:
        challenge_game_seconds = 900 + challenge_clock_seconds
    else:
        challenge_game_seconds = challenge_clock_seconds

    challenge_distance_low, challenge_distance_high, _ = (
        simulator_distance_bounds(int(challenge["ydstogo"]))
    )
    challenge_field_low, challenge_field_high, _ = (
        simulator_field_bounds(int(challenge["yardline_100"]))
    )
    challenge_score_low, challenge_score_high, _ = (
        simulator_score_bounds(
            int(challenge["score_differential"])
        )
    )

    def load_challenge_summary(
        match_tier,
        entity_column=None,
        entity=None,
    ):
        challenge_conditions = [
            "season BETWEEN 2018 AND 2025",
            "season_type IN ('REG', 'POST')",
            "down = ?",
            "goal_to_go = ?",
            "ydstogo BETWEEN ? AND ?",
            "yardline_100 BETWEEN ? AND ?",
            "score_differential BETWEEN ? AND ?",
        ]
        challenge_parameters = [
            int(challenge["down"]),
            int(bool(challenge["goal_to_go"])),
            int(challenge_distance_low),
            int(challenge_distance_high),
            int(challenge_field_low),
            int(challenge_field_high),
            int(challenge_score_low),
            int(challenge_score_high),
        ]

        if match_tier == "Balanced":
            challenge_conditions.extend([
                "qtr = ?",
                "game_seconds_remaining BETWEEN ? AND ?",
            ])
            challenge_parameters.extend([
                int(challenge["qtr"]),
                max(0, challenge_game_seconds - 300),
                min(3600, challenge_game_seconds + 300),
            ])
        else:
            challenge_half_quarters = (
                [1, 2]
                if int(challenge["qtr"]) in (1, 2)
                else [3, 4, 5]
            )
            challenge_conditions.append(
                "qtr IN ("
                + ", ".join(
                    ["?"] * len(challenge_half_quarters)
                )
                + ")"
            )
            challenge_parameters.extend(challenge_half_quarters)

        if entity_column and entity:
            challenge_conditions.append(
                f"{entity_column} = ?"
            )
            challenge_parameters.append(entity)

        challenge_where = (
            "WHERE " + " AND ".join(challenge_conditions)
        )

        challenge_summary = connection.execute(
            f"""
            SELECT
                COUNT(*) AS plays,
                COUNT(DISTINCT game_id) AS games,
                ROUND(100 * AVG(is_pass), 2)
                    AS actual_pass_rate_pct,
                ROUND(
                    100 * AVG(expected_pass_probability),
                    2
                ) AS expected_pass_rate_pct,
                ROUND(100 * AVG(model_pass_oe), 2)
                    AS pass_oe_pct,
                ROUND(AVG(epa), 4) AS mean_epa,
                ROUND(100 * AVG(success), 2)
                    AS success_rate_pct,
                ROUND(100 * AVG(shotgun), 2)
                    AS shotgun_pct,
                ROUND(100 * AVG(no_huddle), 2)
                    AS no_huddle_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Short (0-9)'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS short_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Intermediate (10-19)'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS intermediate_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN pass_depth_bucket = 'Deep (20+)'
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS deep_throw_pct,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN run_direction IN ('Left', 'Right')
                        THEN 1 ELSE 0
                    END) / NULLIF(SUM(CASE
                        WHEN run_direction IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS outside_run_pct
            FROM play_style_predictions_with_callers
            {challenge_where}
            """,
            challenge_parameters,
        ).fetchdf().iloc[0].to_dict()

        challenge_summary["match_tier"] = match_tier
        return challenge_summary

    challenge_league = load_challenge_summary("Balanced")
    if int(challenge_league["plays"]) < 300:
        challenge_league = load_challenge_summary("Broad")

    score_columns = st.columns([2, 1, 1])
    score_columns[0].markdown(
        f"### Challenge {st.session_state.make_call_index + 1} "
        f"of {len(challenge_deck)}: "
        f"{challenge_name}"
    )
    score_columns[1].metric(
        "Session score",
        f"{st.session_state.make_call_score}/"
        f"{st.session_state.make_call_attempts}",
    )
    score_columns[2].metric(
        "Accuracy",
        (
            f"{100 * st.session_state.make_call_score / st.session_state.make_call_attempts:.0f}%"
            if st.session_state.make_call_attempts
            else "New session"
        ),
    )

    st.progress(
        st.session_state.make_call_index / len(challenge_deck),
        text=(
            f"Randomized {st.session_state.make_call_session_difficulty} "
            f"session. No situation repeats within this "
            f"{len(challenge_deck)}-challenge deck."
        ),
    )

    challenge_clock = (
        f"{int(challenge['quarter_minutes'])}:"
        f"{int(challenge['quarter_seconds']):02d}"
    )
    challenge_field_text = (
        f"opponent {int(challenge['yardline_100'])}"
        if int(challenge["yardline_100"]) <= 50
        else f"own {100 - int(challenge['yardline_100'])}"
    )
    challenge_score_text = (
        "tied"
        if int(challenge["score_differential"]) == 0
        else (
            f"leading by {int(challenge['score_differential'])}"
            if int(challenge["score_differential"]) > 0
            else "trailing by "
            + str(abs(int(challenge["score_differential"])))
        )
    )

    st.info(
        f"**{int(challenge['down'])} & "
        f"{int(challenge['ydstogo'])}** at the "
        f"**{challenge_field_text}**, Q{int(challenge['qtr'])} "
        f"{challenge_clock}, offense {challenge_score_text}."
    )

    st.markdown("#### What do you think the historical majority call was?")
    choice_columns = st.columns(2)

    run_choice = choice_columns[0].button(
        "Run",
        width="stretch",
        disabled=st.session_state.make_call_revealed,
        key="make_call_run",
    )
    pass_choice = choice_columns[1].button(
        "Pass",
        width="stretch",
        disabled=st.session_state.make_call_revealed,
        key="make_call_pass",
    )

    if run_choice or pass_choice:
        st.session_state.make_call_choice = (
            "Run" if run_choice else "Pass"
        )
        historical_majority = (
            "Pass"
            if float(
                challenge_league["actual_pass_rate_pct"]
            ) >= 50
            else "Run"
        )
        st.session_state.make_call_attempts += 1
        if st.session_state.make_call_choice == historical_majority:
            st.session_state.make_call_score += 1
        st.session_state.make_call_revealed = True
        st.rerun()

    if st.session_state.make_call_revealed:
        historical_majority = (
            "Pass"
            if float(
                challenge_league["actual_pass_rate_pct"]
            ) >= 50
            else "Run"
        )
        matched_history = (
            st.session_state.make_call_choice
            == historical_majority
        )

        if matched_history:
            st.success(
                f"Correct read. Comparable NFL plays were called as "
                f"a {historical_majority.lower()} more often."
            )
        else:
            st.warning(
                f"Contrarian read. You chose "
                f"{st.session_state.make_call_choice.lower()}, while "
                f"comparable NFL plays were called as a "
                f"{historical_majority.lower()} more often."
            )

        reveal_metrics = st.columns(5)
        reveal_metrics[0].metric(
            "Comparable plays",
            f"{int(challenge_league['plays']):,}",
        )
        reveal_metrics[1].metric(
            "Actual pass rate",
            f"{challenge_league['actual_pass_rate_pct']:.1f}%",
        )
        reveal_metrics[2].metric(
            "Modeled pass expectation",
            f"{challenge_league['expected_pass_rate_pct']:.1f}%",
        )
        reveal_metrics[3].metric(
            "League Pass OE",
            f"{challenge_league['pass_oe_pct']:+.1f} pts",
        )
        reveal_metrics[4].metric(
            "Match tier",
            challenge_league["match_tier"],
        )

        call_split = pd.DataFrame({
            "Call": ["Pass", "Run"],
            "Rate": [
                challenge_league["actual_pass_rate_pct"],
                100 - challenge_league["actual_pass_rate_pct"],
            ],
        })
        call_split_figure = px.bar(
            call_split,
            x="Call",
            y="Rate",
            color="Call",
            title="Historical Call Split",
            labels={"Rate": "Call Rate (%)"},
            text_auto=".1f",
        )
        call_split_figure.update_layout(showlegend=False)
        st.plotly_chart(call_split_figure, width="stretch")

        style_reveal = pd.DataFrame({
            "Style": [
                "Short throws (0-9)",
                "Intermediate throws (10-19)",
                "Deep throws (20+)",
                "Outside runs",
            ],
            "Rate": [
                challenge_league["short_throw_pct"],
                challenge_league["intermediate_throw_pct"],
                challenge_league["deep_throw_pct"],
                challenge_league["outside_run_pct"],
            ],
        })
        st.plotly_chart(
            px.bar(
                style_reveal,
                x="Style",
                y="Rate",
                title="What Those Comparable Plays Looked Like",
                labels={"Rate": "Share of charted plays (%)"},
                text_auto=".1f",
            ),
            width="stretch",
        )

        st.caption(
            "The score rewards matching the historical majority call, "
            "not maximizing EPA or identifying an objectively correct "
            "decision. Throw-depth shares use charted throws with "
            "recorded air yards; outside-run share uses runs with "
            "recorded direction."
        )

        is_last_challenge = (
            st.session_state.make_call_index
            == len(challenge_deck) - 1
        )

        if is_last_challenge:
            st.success(
                f"Session complete: {st.session_state.make_call_score} "
                f"correct reads out of "
                f"{st.session_state.make_call_attempts}."
            )

        navigation_columns = st.columns(2)
        if navigation_columns[0].button(
            (
                "Start another randomized session"
                if is_last_challenge
                else "Next challenge"
            ),
            type="primary",
            width="stretch",
            key="make_call_next",
        ):
            if is_last_challenge:
                st.session_state.make_call_deck = (
                    generate_challenge_deck(
                        challenge_difficulty,
                        int(challenge_length),
                        int(challenge_minimum_plays),
                    )
                )
                st.session_state.make_call_index = 0
                st.session_state.make_call_score = 0
                st.session_state.make_call_attempts = 0
                st.session_state.make_call_session_difficulty = (
                    challenge_difficulty
                )
                st.session_state.make_call_session_minimum = int(
                    challenge_minimum_plays
                )
            else:
                st.session_state.make_call_index += 1
            st.session_state.make_call_revealed = False
            st.session_state.make_call_choice = None
            st.rerun()

        if navigation_columns[1].button(
            "Replay this deck",
            width="stretch",
            key="make_call_reset",
        ):
            st.session_state.make_call_index = 0
            st.session_state.make_call_score = 0
            st.session_state.make_call_attempts = 0
            st.session_state.make_call_revealed = False
            st.session_state.make_call_choice = None
            st.rerun()


# Play Style tab
with style_tab:
    st.subheader("Play Style Explorer")

    st.caption(
        "Explore observable offensive tendencies such as pass depth, "
        "direction, run location, formation, and tempo. These labels "
        "describe recorded play characteristics; they are not inferred "
        "offensive scheme names."
    )

    style_attribution = st.radio(
        "Attribute play style to",
        options=[
            "Head coach",
            "Verified offensive play caller",
        ],
        horizontal=True,
        key="style_attribution",
    )

    style_is_caller = (
        style_attribution
        == "Verified offensive play caller"
    )

    style_entity_column = (
        "offensive_play_caller"
        if style_is_caller
        else "head_coach"
    )

    style_entity_label = (
        "Offensive Play Caller"
        if style_is_caller
        else "Head Coach"
    )

    style_options = (
        filter_options["play_callers"]
        if style_is_caller
        else filter_options["coaches"]
    )

    style_default_candidates = (
        ["Andy Reid", "Sean McVay"]
        if style_is_caller
        else ["Andy Reid", "John Harbaugh"]
    )

    style_defaults = [
        entity
        for entity in style_default_candidates
        if entity in style_options
    ]

    style_entities = st.multiselect(
        (
            "Select up to five offensive play callers"
            if style_is_caller
            else "Select up to five head coaches"
        ),
        options=style_options,
        default=style_defaults,
        max_selections=5,
        key=(
            "style_play_callers"
            if style_is_caller
            else "style_head_coaches"
        ),
    )

    if not style_entities:
        st.info(
            "Select at least one decision-maker to explore play style."
        )
    else:
        style_where, style_parameters = build_filter_query(
            coach_override=style_entities,
            coach_column=style_entity_column,
            use_sidebar_coaches=(
                not style_is_caller
            ),
        )

        fingerprint = connection.execute(
            f"""
            SELECT
                {style_entity_column} AS entity,
                posteam,
                COUNT(*) AS plays,
                SUM(CASE WHEN play_call = 'pass' THEN 1 ELSE 0 END)
                    AS pass_calls,
                SUM(CASE WHEN play_call = 'run' THEN 1 ELSE 0 END)
                    AS run_calls,

                ROUND(100 * AVG(shotgun), 2)
                    AS shotgun_pct,
                ROUND(100 * AVG(no_huddle), 2)
                    AS no_huddle_pct,

                ROUND(
                    AVG(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN air_yards
                    END),
                    2
                ) AS average_air_yards,

                ROUND(
                    100.0
                    * SUM(CASE
                        WHEN pass_depth_bucket = 'Deep (20+)'
                        THEN 1 ELSE 0
                    END)
                    / NULLIF(SUM(CASE
                        WHEN pass_depth_bucket IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS deep_pass_pct,

                ROUND(
                    100.0
                    * SUM(CASE
                        WHEN pass_direction = 'Middle'
                        THEN 1 ELSE 0
                    END)
                    / NULLIF(SUM(CASE
                        WHEN pass_direction IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS middle_pass_pct,

                ROUND(
                    100.0
                    * SUM(CASE
                        WHEN run_direction IN ('Left', 'Right')
                        THEN 1 ELSE 0
                    END)
                    / NULLIF(SUM(CASE
                        WHEN run_direction IS NOT NULL
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS outside_run_pct,

                ROUND(
                    100.0
                    * SUM(CASE
                        WHEN run_gap_style IS NOT NULL
                        THEN 1 ELSE 0
                    END)
                    / NULLIF(SUM(CASE
                        WHEN play_call = 'run'
                        THEN 1 ELSE 0
                    END), 0),
                    2
                ) AS run_gap_coverage_pct

            FROM play_style_predictions_with_callers
            {style_where}
            GROUP BY
                {style_entity_column},
                posteam
            ORDER BY
                {style_entity_column},
                posteam
            """,
            style_parameters,
        ).fetchdf()

        fingerprint["entity_team"] = (
            fingerprint["entity"]
            + " ("
            + fingerprint["posteam"]
            + ")"
        )

        overall_style = connection.execute(
            f"""
            SELECT
                COUNT(*) AS plays,
                100 * AVG(shotgun) AS shotgun_pct,
                100 * AVG(no_huddle) AS no_huddle_pct,
                100.0
                * SUM(CASE
                    WHEN pass_depth_bucket = 'Deep (20+)'
                    THEN 1 ELSE 0
                END)
                / NULLIF(SUM(CASE
                    WHEN pass_depth_bucket IS NOT NULL
                    THEN 1 ELSE 0
                END), 0) AS deep_pass_pct,
                100.0
                * SUM(CASE
                    WHEN run_direction IN ('Left', 'Right')
                    THEN 1 ELSE 0
                END)
                / NULLIF(SUM(CASE
                    WHEN run_direction IS NOT NULL
                    THEN 1 ELSE 0
                END), 0) AS outside_run_pct
            FROM play_style_predictions_with_callers
            {style_where}
            """,
            style_parameters,
        ).fetchdf().iloc[0]

        total_plays = int(overall_style["plays"])
        weighted_shotgun_rate = overall_style[
            "shotgun_pct"
        ]
        weighted_no_huddle_rate = overall_style[
            "no_huddle_pct"
        ]
        weighted_deep_rate = overall_style[
            "deep_pass_pct"
        ]
        weighted_outside_run_rate = overall_style[
            "outside_run_pct"
        ]

        style_metrics = st.columns(5)

        style_metrics[0].metric(
            "Selected plays",
            f"{total_plays:,}",
        )
        style_metrics[1].metric(
            "Shotgun rate",
            f"{weighted_shotgun_rate:.1f}%",
        )
        style_metrics[2].metric(
            "No-huddle rate",
            f"{weighted_no_huddle_rate:.1f}%",
        )
        style_metrics[3].metric(
            "Deep throw rate (20+ air yards)",
            f"{weighted_deep_rate:.1f}%",
            help=(
                "Share of charted throws traveling at least "
                "20 air yards."
            ),
        )
        style_metrics[4].metric(
            "Outside run rate",
            f"{weighted_outside_run_rate:.1f}%",
            help=(
                "Share of charted runs recorded left or right "
                "rather than middle."
            ),
        )

        st.markdown("#### Offensive Fingerprints")

        fingerprint_long = fingerprint.melt(
            id_vars=["entity_team"],
            value_vars=[
                "shotgun_pct",
                "no_huddle_pct",
                "deep_pass_pct",
                "middle_pass_pct",
                "outside_run_pct",
            ],
            var_name="style_metric",
            value_name="percentage",
        )

        fingerprint_long["style_metric"] = (
            fingerprint_long["style_metric"].replace({
                "shotgun_pct": "Shotgun",
                "no_huddle_pct": "No huddle",
                "deep_pass_pct": "Deep throws (20+ air yards)",
                "middle_pass_pct": "Middle-field throws",
                "outside_run_pct": "Outside runs",
            })
        )

        fingerprint_figure = px.bar(
            fingerprint_long,
            x="style_metric",
            y="percentage",
            color="entity_team",
            barmode="group",
            title="Observable Offensive Style Fingerprint",
            labels={
                "style_metric": "Style dimension",
                "percentage": "Rate (%)",
                "entity_team": "Decision-Maker-Team",
            },
        )

        st.plotly_chart(
            fingerprint_figure,
            width="stretch",
        )

        (
            pass_style_subtab,
            run_style_subtab,
            presentation_subtab,
            style_table_subtab,
        ) = st.tabs([
            "Passing Style",
            "Rushing Style",
            "Formation & Tempo",
            "Fingerprint Table",
        ])

        def load_style_distribution(
            field_name,
            additional_condition,
        ):
            data = connection.execute(
                f"""
                SELECT
                    {style_entity_column} AS entity,
                    posteam,
                    {field_name} AS category,
                    COUNT(*) AS plays
                FROM play_style_predictions_with_callers
                {style_where}
                    AND {additional_condition}
                    AND {field_name} IS NOT NULL
                GROUP BY
                    {style_entity_column},
                    posteam,
                    {field_name}
                ORDER BY
                    {style_entity_column},
                    posteam,
                    {field_name}
                """,
                style_parameters,
            ).fetchdf()

            if data.empty:
                return data

            data["entity_team"] = (
                data["entity"]
                + " ("
                + data["posteam"]
                + ")"
            )

            data["percentage"] = (
                100
                * data["plays"]
                / data.groupby(
                    "entity_team"
                )["plays"].transform("sum")
            )

            return data

        with pass_style_subtab:
            pass_columns = st.columns(2)

            pass_depth = load_style_distribution(
                "pass_depth_bucket",
                "play_call = 'pass'",
            )

            pass_direction = load_style_distribution(
                "pass_direction",
                "play_call = 'pass'",
            )

            with pass_columns[0]:
                st.plotly_chart(
                    px.bar(
                        pass_depth,
                        x="category",
                        y="percentage",
                        color="entity_team",
                        barmode="group",
                        title="Charted Throw Depth",
                        category_orders={
                            "category": [
                                "Behind line",
                                "Short (0-9)",
                                "Intermediate (10-19)",
                                "Deep (20+)",
                            ]
                        },
                        labels={
                            "category": "Depth",
                            "percentage": "Share of charted throws (%)",
                            "entity_team": "Decision-Maker-Team",
                        },
                    ),
                    width="stretch",
                )

            with pass_columns[1]:
                st.plotly_chart(
                    px.bar(
                        pass_direction,
                        x="category",
                        y="percentage",
                        color="entity_team",
                        barmode="group",
                        title="Charted Pass Direction",
                        category_orders={
                            "category": ["Left", "Middle", "Right"]
                        },
                        labels={
                            "category": "Direction",
                            "percentage": "Share of charted throws (%)",
                            "entity_team": "Decision-Maker-Team",
                        },
                    ),
                    width="stretch",
                )

            st.caption(
                "Pass-depth and direction charts use throws with "
                "recorded air-yard/location data. Sacks and scrambles "
                "remain pass calls in the run-pass model but are not "
                "assigned a target depth or direction."
            )

        with run_style_subtab:
            run_columns = st.columns(2)

            run_direction = load_style_distribution(
                "run_direction",
                "play_call = 'run'",
            )

            run_gap = load_style_distribution(
                "run_gap_style",
                "play_call = 'run'",
            )

            with run_columns[0]:
                st.plotly_chart(
                    px.bar(
                        run_direction,
                        x="category",
                        y="percentage",
                        color="entity_team",
                        barmode="group",
                        title="Run Direction",
                        category_orders={
                            "category": ["Left", "Middle", "Right"]
                        },
                        labels={
                            "category": "Direction",
                            "percentage": "Share of charted runs (%)",
                            "entity_team": "Decision-Maker-Team",
                        },
                    ),
                    width="stretch",
                )

            with run_columns[1]:
                st.plotly_chart(
                    px.bar(
                        run_gap,
                        x="category",
                        y="percentage",
                        color="entity_team",
                        barmode="group",
                        title="Recorded Run Gap",
                        category_orders={
                            "category": ["Guard", "Tackle", "End"]
                        },
                        labels={
                            "category": "Gap",
                            "percentage": "Share of gap-charted runs (%)",
                            "entity_team": "Decision-Maker-Team",
                        },
                    ),
                    width="stretch",
                )

            st.caption(
                "Run direction is available for nearly every run. Run "
                "gap is available for approximately 73% of runs, so gap "
                "percentages describe only the charted subset."
            )

        with presentation_subtab:
            presentation_columns = st.columns(2)

            formation = load_style_distribution(
                "formation_style",
                "formation_style != 'Unknown'",
            )

            tempo = load_style_distribution(
                "tempo_style",
                "tempo_style != 'Unknown'",
            )

            with presentation_columns[0]:
                st.plotly_chart(
                    px.bar(
                        formation,
                        x="category",
                        y="percentage",
                        color="entity_team",
                        barmode="group",
                        title="Formation Presentation",
                        labels={
                            "category": "Formation",
                            "percentage": "Play share (%)",
                            "entity_team": "Decision-Maker-Team",
                        },
                    ),
                    width="stretch",
                )

            with presentation_columns[1]:
                st.plotly_chart(
                    px.bar(
                        tempo,
                        x="category",
                        y="percentage",
                        color="entity_team",
                        barmode="group",
                        title="Tempo",
                        labels={
                            "category": "Tempo",
                            "percentage": "Play share (%)",
                            "entity_team": "Decision-Maker-Team",
                        },
                    ),
                    width="stretch",
                )

        with style_table_subtab:
            fingerprint_display = fingerprint[
                [
                    "entity",
                    "posteam",
                    "plays",
                    "pass_calls",
                    "run_calls",
                    "shotgun_pct",
                    "no_huddle_pct",
                    "average_air_yards",
                    "deep_pass_pct",
                    "middle_pass_pct",
                    "outside_run_pct",
                    "run_gap_coverage_pct",
                ]
            ].rename(
                columns={
                    "entity": style_entity_label,
                    "posteam": "Team",
                    "plays": "Plays",
                    "pass_calls": "Pass Calls",
                    "run_calls": "Run Calls",
                    "shotgun_pct": "Shotgun %",
                    "no_huddle_pct": "No-Huddle %",
                    "average_air_yards": "Average Air Yards",
                    "deep_pass_pct": "Deep Throw % (20+ Air Yards)",
                    "middle_pass_pct": "Middle-Field Throw %",
                    "outside_run_pct": "Outside Run %",
                    "run_gap_coverage_pct": "Run Gap Coverage %",
                }
            )

            st.dataframe(
                fingerprint_display,
                width="stretch",
                hide_index=True,
            )


# Attribution analysis tab
with attribution_tab:
    st.subheader("Verified Attribution Analysis")

    st.caption(
        "Compare the established head-coach attribution with "
        "the verified offensive play caller. Play-caller research "
        "is currently complete for 2018 through 2025."
    )

    verified_attribution_seasons = connection.execute(
        """
        SELECT DISTINCT season
        FROM offensive_play_caller_tenures
        WHERE verification_status = 'verified'
        ORDER BY season
        """
    ).fetchdf()["season"].tolist()

    active_attribution_seasons = [
        season
        for season in selected_seasons
        if season in verified_attribution_seasons
    ]

    attribution_mode = st.radio(
        "Attribute play-calling decisions to",
        options=[
            "Head coach",
            "Verified offensive play caller",
        ],
        horizontal=True,
        key="attribution_mode",
    )

    if not active_attribution_seasons:
        st.info(
            "Select at least one verified season in the sidebar "
            "to view the attribution comparison. Currently "
            "verified seasons: "
            + ", ".join(
                str(season)
                for season in verified_attribution_seasons
            )
            + "."
        )
    else:
        if attribution_mode == "Head coach":
            entity_column = "head_coach"
            entity_label = "Head Coach"
            attribution_note = (
                "These results use the head coach attached to each "
                "play by nflverse."
            )
        else:
            entity_column = "offensive_play_caller"
            entity_label = "Offensive Play Caller"
            attribution_note = (
                "These results use the manually verified "
                "offensive play-caller reference for the selected "
                "verified seasons."
            )

        st.caption(attribution_note)

        st.caption(
            "Active verified seasons: "
            + ", ".join(
                str(season)
                for season in active_attribution_seasons
            )
        )

        attribution_conditions = [
            (
                "attribution_type = "
                "'verified_offensive_play_caller'"
            ),
        ]
        attribution_parameters = []

        season_placeholders = ", ".join(
            ["?"] * len(active_attribution_seasons)
        )
        attribution_conditions.insert(
            0,
            f"season IN ({season_placeholders})",
        )
        attribution_parameters.extend(
            active_attribution_seasons
        )

        def add_attribution_list_filter(
            column_name,
            selected_values,
        ):
            if not selected_values:
                return

            placeholders = ", ".join(
                ["?"] * len(selected_values)
            )
            attribution_conditions.append(
                f"{column_name} IN ({placeholders})"
            )
            attribution_parameters.extend(selected_values)

        add_attribution_list_filter(
            "season_type",
            selected_season_types,
        )
        add_attribution_list_filter(
            "posteam",
            selected_teams,
        )
        add_attribution_list_filter(
            "down",
            selected_downs,
        )
        add_attribution_list_filter(
            "qtr",
            selected_quarters,
        )

        attribution_conditions.extend([
            "ydstogo BETWEEN ? AND ?",
            "yardline_100 BETWEEN ? AND ?",
            "score_differential BETWEEN ? AND ?",
        ])
        attribution_parameters.extend([
            distance_bounds[0],
            distance_bounds[1],
            yardline_bounds[0],
            yardline_bounds[1],
            score_bounds[0],
            score_bounds[1],
        ])

        attribution_where = (
            "WHERE "
            + " AND ".join(
                attribution_conditions
            )
        )

        entity_options = connection.execute(
            f"""
            SELECT DISTINCT
                {entity_column} AS entity
            FROM play_predictions_with_callers
            {attribution_where}
                AND {entity_column} IS NOT NULL
            ORDER BY entity
            """,
            attribution_parameters,
        ).fetchdf()["entity"].tolist()

        entity_selector_label = (
            "Head Coaches"
            if entity_column == "head_coach"
            else "Offensive Play Callers"
        )

        selected_entities = st.multiselect(
            entity_selector_label,
            options=entity_options,
            key=(
                "attribution_entities_"
                + entity_column
            ),
            help=(
                "Leave this empty to include every eligible "
                f"{entity_label.lower()}."
            ),
        )

        entity_where = attribution_where
        entity_parameters = list(
            attribution_parameters
        )

        if selected_entities:
            entity_placeholders = ", ".join(
                ["?"] * len(selected_entities)
            )
            entity_where += (
                f" AND {entity_column} IN "
                f"({entity_placeholders})"
            )
            entity_parameters.extend(
                selected_entities
            )

        attribution_summary = connection.execute(
            f"""
            SELECT
                {entity_column} AS entity,
                posteam,
                COUNT(*) AS plays,
                COUNT(DISTINCT game_id) AS games,
                ROUND(100 * AVG(is_pass), 2)
                    AS actual_pass_rate_pct,
                ROUND(
                    100 * AVG(
                        expected_pass_probability
                    ),
                    2
                ) AS expected_pass_rate_pct,
                ROUND(100 * AVG(model_pass_oe), 2)
                    AS pass_oe_pct,
                ROUND(AVG(epa), 4) AS mean_epa,
                ROUND(100 * AVG(success), 2)
                    AS success_rate_pct,
                ROUND(AVG(yards_gained), 2)
                    AS yards_per_play
            FROM play_predictions_with_callers
            {entity_where}
            GROUP BY
                {entity_column},
                posteam
            ORDER BY pass_oe_pct DESC
            """,
            entity_parameters,
        ).fetchdf()

        if attribution_summary.empty:
            st.warning(
                "No plays match the selected attribution and "
                "situation filters."
            )
        else:
            total_plays = int(
                attribution_summary["plays"].sum()
            )
            weighted_actual_rate = (
                (
                    attribution_summary[
                        "actual_pass_rate_pct"
                    ]
                    * attribution_summary["plays"]
                ).sum()
                / total_plays
            )
            weighted_expected_rate = (
                (
                    attribution_summary[
                        "expected_pass_rate_pct"
                    ]
                    * attribution_summary["plays"]
                ).sum()
                / total_plays
            )
            weighted_pass_oe = (
                weighted_actual_rate
                - weighted_expected_rate
            )

            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Plays",
                f"{total_plays:,}",
            )
            metric_columns[1].metric(
                "Actual pass rate",
                f"{weighted_actual_rate:.2f}%",
            )
            metric_columns[2].metric(
                "Expected pass rate",
                f"{weighted_expected_rate:.2f}%",
            )
            metric_columns[3].metric(
                "Pass rate over expected",
                f"{weighted_pass_oe:+.2f} pts",
            )

            attribution_summary["label"] = (
                attribution_summary["entity"]
                + " ("
                + attribution_summary["posteam"]
                + ")"
            )

            st.markdown(
                f"#### {entity_label} Rankings"
            )

            eligible_attribution = (
                attribution_summary[
                    attribution_summary["plays"]
                    >= minimum_plays
                ].copy()
            )

            if eligible_attribution.empty:
                st.info(
                    "No attribution groups meet the current "
                    f"{minimum_plays:,}-play minimum. Lower the "
                    "minimum-play filter to display rankings."
                )
            else:
                ranking_figure = px.bar(
                    eligible_attribution.sort_values(
                        "pass_oe_pct"
                    ),
                    x="pass_oe_pct",
                    y="label",
                    orientation="h",
                    color="pass_oe_pct",
                    color_continuous_scale=[
                        "#d73027",
                        "#f7f7f7",
                        "#1f78b4",
                    ],
                    color_continuous_midpoint=0,
                    labels={
                        "pass_oe_pct": (
                            "Pass OE (percentage points)"
                        ),
                        "label": "",
                    },
                    hover_data={
                        "plays": ":,",
                        "actual_pass_rate_pct": ":.2f",
                        "expected_pass_rate_pct": ":.2f",
                        "mean_epa": ":.4f",
                    },
                )
                ranking_figure.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="gray",
                )
                ranking_figure.update_layout(
                    coloraxis_showscale=False,
                    height=max(
                        500,
                        28 * len(
                            eligible_attribution
                        ),
                    ),
                )
                st.plotly_chart(
                    ranking_figure,
                    width="stretch",
                )

            st.markdown("#### Attribution Results")

            attribution_display = attribution_summary[
                [
                    "entity",
                    "posteam",
                    "plays",
                    "games",
                    "actual_pass_rate_pct",
                    "expected_pass_rate_pct",
                    "pass_oe_pct",
                    "mean_epa",
                    "success_rate_pct",
                    "yards_per_play",
                ]
            ].rename(
                columns={
                    "entity": entity_label,
                    "posteam": "Team",
                    "plays": "Plays",
                    "games": "Games",
                    "actual_pass_rate_pct": "Actual Pass %",
                    "expected_pass_rate_pct": "Expected Pass %",
                    "pass_oe_pct": "Pass OE",
                    "mean_epa": "EPA/Play",
                    "success_rate_pct": "Success %",
                    "yards_per_play": "Yards/Play",
                }
            )
            st.dataframe(
                attribution_display,
                width="stretch",
                hide_index=True,
            )

        if (
            attribution_mode
            == "Verified offensive play caller"
        ):
            with st.expander(
                "Verification details and sources"
            ):
                verification_details = connection.execute(
                    f"""
                    SELECT
                        season,
                        team,
                        start_week,
                        end_week,
                        head_coach,
                        offensive_play_caller,
                        caller_role,
                        source_title,
                        source_publisher,
                        source_url,
                        date_verified
                    FROM offensive_play_caller_tenures
                    WHERE
                        season IN ({season_placeholders})
                        AND verification_status = 'verified'
                    ORDER BY
                        season,
                        team,
                        start_week
                    """,
                    active_attribution_seasons,
                ).fetchdf()

                st.dataframe(
                    verification_details.rename(
                        columns={
                            "season": "Season",
                            "team": "Team",
                            "start_week": "Start Week",
                            "end_week": "End Week",
                            "head_coach": "Head Coach",
                            "offensive_play_caller": (
                                "Offensive Play Caller"
                            ),
                            "caller_role": "Caller Role",
                            "source_title": "Source",
                            "source_publisher": "Publisher",
                            "source_url": "Source URL",
                            "date_verified": "Date Verified",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Source URL": st.column_config.LinkColumn(
                            "Source URL"
                        ),
                    },
                )


# Model performance tab
with model_tab:
    st.subheader("Model Performance")

    st.markdown(
        "#### Held-Out 2025 Test Results"
    )

    final_metrics = connection.execute(
        """
        SELECT *
        FROM final_test_metrics
        ORDER BY model
        """
    ).fetchdf()

    st.dataframe(
        final_metrics,
        width="stretch",
        hide_index=True,
    )

    season_metrics = connection.execute(
        """
        SELECT *
        FROM model_metrics_by_season
        ORDER BY season, model
        """
    ).fetchdf()

    auc_figure = px.line(
        season_metrics,
        x="season",
        y="roc_auc",
        color="model",
        markers=True,
        title="ROC AUC by Season",
    )

    st.plotly_chart(
        auc_figure,
        width="stretch",
    )

    brier_figure = px.line(
        season_metrics,
        x="season",
        y="brier_score",
        color="model",
        markers=True,
        title=(
            "Brier Score by Season "
            "(Lower is Better)"
        ),
    )

    st.plotly_chart(
        brier_figure,
        width="stretch",
    )


# Methodology tab
with methodology_tab:
    st.subheader("Methodology")

    st.markdown(
        """
        This application evaluates NFL run-pass decisions against a
        common, league-wide expectation model. Its central question is:
        **given the information available before the snap, how likely was
        a pass call in this situation?** Actual decisions are then compared
        with that probability to describe play-calling tendencies.

        #### Data and unit of analysis

        The analysis covers the 2018 through 2025 NFL seasons and retains
        competitive run-pass decisions from both the regular season and
        postseason. Each row represents one eligible play. Regular-season
        and postseason results can be examined together or separately.

        - **Pass calls** include designed quarterback dropbacks, completed
          and incomplete passes, sacks, and quarterback scrambles that began
          as passing plays.
        - **Run calls** include the remaining competitive rushing attempts.
        - Kneel-downs, spikes, aborted plays, and plays lacking the required
          teams, down, or run-pass target are excluded.

        Treating sacks and scrambles as passes reflects the intended play
        design rather than the play's recorded ending.

        #### Situation-only expectation model

        A single league-wide histogram gradient boosting classifier estimates
        the probability of a pass. Every coach and play caller is therefore
        compared with the same league baseline rather than with a separately
        trained personal model.

        Model inputs are restricted to information available before the snap:

        - Season week and season type
        - Quarter, down, and yards to go
        - Field position and goal-to-go status
        - Game time remaining
        - Score differential
        - Offensive and defensive timeouts
        - Home-team and opening-kickoff context
        - Roof and playing surface
        - Pregame point spread and game total

        Coach identity, offensive team identity, defensive team identity,
        play outcome, EPA, success, yards gained, weather values, formation,
        and nflverse `xpass` are not model inputs. Excluding coach and team
        identity is essential: the baseline is intended to represent league
        situational expectations, not reproduce the tendencies being measured.
        nflverse `xpass` is retained only as an external benchmark.

        #### Training and out-of-sample evaluation

        Model selection used a chronological development design:

        - **Training:** 2018-2023
        - **Validation and model selection:** 2024
        - **Final training:** 2018-2024
        - **Held-out final test:** 2025

        The 2025 test season was not used to select or tune the final model.
        Historical 2018-2024 tendencies use leave-one-season-out predictions:
        the displayed season is excluded from the fold used to predict it.
        This produces out-of-sample residuals for descriptive historical
        comparison, but it should not be interpreted as a strictly real-time
        forecast made using only earlier seasons.

        #### Pass rate over expected

        For each play, the model residual is:

        `actual pass indicator - expected pass probability`

        Pass rate over expected, or **Pass OE**, is the average residual for
        the selected sample, expressed in percentage points. A value of
        `+4.0` means the offense passed four percentage points more often than
        the league model expected in those situations. A value of `-4.0`
        means it passed four percentage points less often.

        Pass OE describes a tendency relative to modeled context. It is not a
        measure of play quality and does not by itself establish that passing
        or running more often improved results.

        #### Head-coach and offensive play-caller attribution

        Head-coach attribution uses the home and away head coaches supplied
        with nflverse play-by-play. Because a head coach does not necessarily
        call the offense, the project also contains a manually curated and
        sourced offensive play-caller reference covering 2018-2025.

        The reference is organized into team-season-week segments so that
        midseason changes are preserved. Each verified play is matched by
        season, offensive team, head coach, and week. The Attribution Analysis,
        Coach Comparison, and Historical Trends interfaces allow users to
        distinguish the established head-coach view from verified offensive
        play-caller attribution.

        Attribution identifies whose published role included calling the
        offense; it does not claim that one person independently determined
        every play. Game plans and decisions may also reflect coordinators,
        quarterbacks, other assistants, ownership, personnel, and in-game
        collaboration.

        #### Play Style Explorer

        The Play Style tab is a descriptive companion to the run-pass model.
        It uses recorded play-by-play characteristics rather than adding
        post-snap information to the expectation model. Available dimensions
        include shotgun presentation, no-huddle tempo, air-yard depth, pass
        direction, run direction, and recorded run gap.

        Pass-depth and direction summaries use throws with recorded charting;
        sacks and scrambles remain pass calls but are not assigned a target
        depth. Run direction is nearly complete, while run-gap labels cover
        approximately 73% of runs and are presented as a charted subset.
        Play-action and named scheme classifications are not displayed because
        they are not reliably available in the underlying data. Terms such as
        "deep throw rate (20+ air yards)" or "outside run rate" describe
        observable tendencies, not inferred systems such as wide zone,
        West Coast, or Air Coryell.

        #### Situation Lab and Make the Call

        The Situation Lab searches for comparable historical plays using
        progressively broader matching tiers based on down, distance, field
        position, quarter or half, clock, score state, and goal-to-go status.
        The narrowest tier meeting the displayed sample requirement is used.

        Make the Call uses the same descriptive historical framework. Each
        session is generated from historically populated combinations of
        down, distance, field-position bucket, half, score state, and
        goal-to-go status. The deck is shuffled and a situation cannot repeat
        within the same session. Difficulty controls how close the underlying
        historical call split is to 50-50. A point is awarded when the user's
        run-pass choice matches the majority call among the comparable league
        plays. The score does not evaluate play quality, expected points, win
        probability, or strategic optimality. It is a game about recognizing
        historical league behavior, not a play recommendation engine.

        #### Confidence intervals and minimum samples

        Uncertainty is estimated separately for head-coach-team-seasons and
        verified caller-team-seasons. For each group, the application performs
        2,000 bootstrap resamples of complete games. Sampling whole games,
        rather than individual plays, preserves within-game dependence while
        estimating the uncertainty around average Pass OE.

        - The displayed interval is the 2.5th to 97.5th percentile of the
          bootstrap distribution.
        - A group is labeled **pass-heavy** when the entire interval is above
          zero.
        - A group is labeled **run-heavy** when the entire interval is below
          zero.
        - Otherwise, its tendency is labeled **uncertain**.
        - Rankings and uncertainty displays require at least 500 eligible
          plays by default.

        Confidence intervals use complete group-seasons and therefore do not
        change with down, distance, field-position, quarter, or score filters.
        This prevents a full-season interval from being misrepresented as an
        interval for a much smaller filtered sample.

        #### Outcomes and team records

        EPA per play, success rate, yards per play, scoring, point
        differential, and win-loss records are shown as context. Play-level
        outcomes follow all active play filters. Team records and scoring
        summaries follow season and season-type filters but not situation
        filters.

        These outcomes are descriptive associations, not causal estimates.
        Differences may reflect quarterback performance, opponent strength,
        injuries, roster quality, game state, execution, and many other factors
        beyond play-calling tendency.

        #### Limitations

        - The expectation model cannot observe every pre-snap consideration,
          including the full personnel package, defensive alignment, injury
          information, audibles, and private game-plan information.
        - Historical cross-fitting is designed for out-of-sample description,
          not a simulation of what was knowable at every historical date.
        - Published play-caller responsibilities can be collaborative or
          described imperfectly even after manual verification.
        - A tendency relative to expectation should not be interpreted as a
          recommendation to pass or run more frequently.
        - Rankings become less stable as filters reduce the number of plays;
          sample sizes should always be considered alongside point estimates.
        """
    )