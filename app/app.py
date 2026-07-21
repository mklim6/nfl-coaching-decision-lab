from io import BytesIO
from pathlib import Path
import html
import random
import re
import zipfile

# NFL Coaching Decision Lab with realistic randomized Make the Call sessions.

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="NFL Coaching Decision Lab",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
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


TEAM_VISUALS = {
    "ARI": {"name": "Arizona Cardinals", "primary": "#97233F", "secondary": "#FFB612"},
    "ATL": {"name": "Atlanta Falcons", "primary": "#A71930", "secondary": "#A5ACAF"},
    "BAL": {"name": "Baltimore Ravens", "primary": "#241773", "secondary": "#9E7C0C"},
    "BUF": {"name": "Buffalo Bills", "primary": "#00338D", "secondary": "#C60C30"},
    "CAR": {"name": "Carolina Panthers", "primary": "#0085CA", "secondary": "#BFC0BF"},
    "CHI": {"name": "Chicago Bears", "primary": "#C83803", "secondary": "#0B162A"},
    "CIN": {"name": "Cincinnati Bengals", "primary": "#FB4F14", "secondary": "#A5ACAF"},
    "CLE": {"name": "Cleveland Browns", "primary": "#FF3C00", "secondary": "#311D00"},
    "DAL": {"name": "Dallas Cowboys", "primary": "#003594", "secondary": "#869397"},
    "DEN": {"name": "Denver Broncos", "primary": "#FB4F14", "secondary": "#002244"},
    "DET": {"name": "Detroit Lions", "primary": "#0076B6", "secondary": "#B0B7BC"},
    "GB": {"name": "Green Bay Packers", "primary": "#203731", "secondary": "#FFB612"},
    "HOU": {"name": "Houston Texans", "primary": "#03202F", "secondary": "#A71930"},
    "IND": {"name": "Indianapolis Colts", "primary": "#002C5F", "secondary": "#A2AAAD"},
    "JAX": {"name": "Jacksonville Jaguars", "primary": "#006778", "secondary": "#D7A22A"},
    "KC": {"name": "Kansas City Chiefs", "primary": "#E31837", "secondary": "#FFB81C"},
    "LA": {"name": "Los Angeles Rams", "primary": "#003594", "secondary": "#FFA300"},
    "LAC": {"name": "Los Angeles Chargers", "primary": "#0080C6", "secondary": "#FFC20E"},
    "LV": {"name": "Las Vegas Raiders", "primary": "#A5ACAF", "secondary": "#6B7280"},
    "MIA": {"name": "Miami Dolphins", "primary": "#008E97", "secondary": "#FC4C02"},
    "MIN": {"name": "Minnesota Vikings", "primary": "#4F2683", "secondary": "#FFC62F"},
    "NE": {"name": "New England Patriots", "primary": "#002244", "secondary": "#C60C30"},
    "NO": {"name": "New Orleans Saints", "primary": "#D3BC8D", "secondary": "#8A7448"},
    "NYG": {"name": "New York Giants", "primary": "#0B2265", "secondary": "#A71930"},
    "NYJ": {"name": "New York Jets", "primary": "#125740", "secondary": "#6B8F7B"},
    "PHI": {"name": "Philadelphia Eagles", "primary": "#004C54", "secondary": "#A5ACAF"},
    "PIT": {"name": "Pittsburgh Steelers", "primary": "#FFB612", "secondary": "#8A7A48"},
    "SEA": {"name": "Seattle Seahawks", "primary": "#69BE28", "secondary": "#005C5C"},
    "SF": {"name": "San Francisco 49ers", "primary": "#AA0000", "secondary": "#B3995D"},
    "TB": {"name": "Tampa Bay Buccaneers", "primary": "#D50A0A", "secondary": "#FF7900"},
    "TEN": {"name": "Tennessee Titans", "primary": "#4B92DB", "secondary": "#C8102E"},
    "WAS": {"name": "Washington Commanders", "primary": "#5A1414", "secondary": "#FFB612"},
}


def hex_to_rgba(hex_color, alpha):
    """Convert a hex color into an rgba CSS value."""

    normalized = hex_color.lstrip("#")
    red = int(normalized[0:2], 16)
    green = int(normalized[2:4], 16)
    blue = int(normalized[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def readable_text_color(hex_color):
    """Choose dark or light text for a solid background color."""

    normalized = hex_color.lstrip("#")
    red = int(normalized[0:2], 16) / 255
    green = int(normalized[2:4], 16) / 255
    blue = int(normalized[4:6], 16) / 255

    def linearize(channel):
        return (
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )

    luminance = (
        0.2126 * linearize(red)
        + 0.7152 * linearize(green)
        + 0.0722 * linearize(blue)
    )
    return "#0B0F16" if luminance > 0.48 else "#FFFFFF"


active_team = (
    selected_teams[0]
    if len(selected_teams) == 1
    else None
)
active_team_visual = TEAM_VISUALS.get(
    active_team,
    {
        "name": "NFL Coaching Decision Lab",
        "primary": "#FF4B4B",
        "secondary": "#4DA3FF",
    },
)
active_primary = active_team_visual["primary"]
active_secondary = active_team_visual["secondary"]
active_primary_text = readable_text_color(active_primary)
active_secondary_text = readable_text_color(active_secondary)
active_colorway = [
    active_primary,
    active_secondary,
    "#4DA3FF",
    "#58C17D",
    "#C084FC",
    "#F2A65A",
    "#F472B6",
    "#8FA4B8",
]


st.markdown(
    f"""
    <style>
    :root {{
        --team-primary: {active_primary};
        --team-secondary: {active_secondary};
        --team-primary-soft: {hex_to_rgba(active_primary, 0.15)};
        --team-primary-faint: {hex_to_rgba(active_primary, 0.065)};
        --team-secondary-soft: {hex_to_rgba(active_secondary, 0.14)};
        --team-primary-text: {active_primary_text};
        --team-secondary-text: {active_secondary_text};
        --page-bg: #0B0F16;
        --panel-bg: rgba(18, 24, 33, 0.80);
        --panel-bg-strong: rgba(14, 19, 27, 0.96);
        --panel-border: rgba(148, 163, 184, 0.18);
        --panel-border-hover: rgba(148, 163, 184, 0.30);
        --muted-text: #AEB8C6;
        --soft-text: #D5DEE9;
    }}

    [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(
                circle at 10% -8%,
                var(--team-primary-soft),
                transparent 34rem
            ),
            radial-gradient(
                circle at 92% 4%,
                var(--team-secondary-soft),
                transparent 29rem
            ),
            linear-gradient(
                180deg,
                #0D121A 0%,
                var(--page-bg) 48%,
                #090D13 100%
            );
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(11, 15, 22, 0.72);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }}

    .block-container {{
        max-width: 1480px;
        padding-top: 2.35rem;
        padding-bottom: 4.5rem;
    }}

    h1 {{
        letter-spacing: -0.042em;
        line-height: 1.02;
        margin-bottom: 0.38rem;
        color: #F8FAFC;
        text-shadow: 0 10px 34px rgba(0, 0, 0, 0.30);
    }}

    h1::after {{
        content: "";
        display: block;
        width: 58px;
        height: 3px;
        margin-top: 0.68rem;
        border-radius: 999px;
        background: linear-gradient(
            90deg,
            var(--team-primary),
            var(--team-secondary)
        );
        box-shadow: 0 0 14px var(--team-primary-soft);
    }}

    h2, h3 {{
        letter-spacing: -0.024em;
        color: #F4F7FB;
    }}

    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        text-wrap: balance;
    }}

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {{
        color: var(--soft-text);
        line-height: 1.62;
    }}

    [data-testid="stCaptionContainer"] {{
        color: var(--muted-text);
    }}

    a {{
        color: var(--team-secondary);
    }}

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                var(--team-primary-soft) 0,
                rgba(32, 34, 44, 0.98) 10.5rem,
                rgba(31, 33, 43, 0.99) 100%
            );
        border-right: 1px solid var(--panel-border);
    }}

    [data-testid="stSidebar"] h2 {{
        letter-spacing: -0.015em;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        line-height: 1.5;
    }}

    [data-baseweb="tag"] {{
        background-color: var(--team-primary) !important;
        color: var(--team-primary-text) !important;
        border: 1px solid {hex_to_rgba(active_secondary, 0.42)} !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.14);
    }}

    [data-testid="stMetric"] {{
        min-height: 112px;
        padding: 1rem 1.08rem;
        border: 1px solid var(--panel-border);
        border-top: 3px solid var(--team-primary);
        border-radius: 17px;
        background:
            radial-gradient(
                circle at 100% 0%,
                var(--team-primary-faint),
                transparent 12rem
            ),
            linear-gradient(
                145deg,
                rgba(24, 31, 42, 0.93),
                rgba(16, 22, 31, 0.91)
            );
        box-shadow:
            0 14px 34px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
        transition:
            transform 140ms ease,
            border-color 140ms ease,
            box-shadow 140ms ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        border-color: var(--panel-border-hover);
        box-shadow:
            0 18px 38px rgba(0, 0, 0, 0.20),
            0 0 0 1px var(--team-primary-faint);
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--muted-text);
        font-weight: 700;
        letter-spacing: 0.015em;
    }}

    [data-testid="stMetricValue"] {{
        letter-spacing: -0.04em;
        line-height: 1.06;
        overflow: visible !important;
    }}

    [data-testid="stMetricValue"] > div {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
        font-size: clamp(1.42rem, 2.15vw, 2.15rem) !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid var(--panel-border) !important;
        border-radius: 17px !important;
        background:
            radial-gradient(
                circle at 100% 0%,
                var(--team-primary-faint),
                transparent 18rem
            ),
            linear-gradient(
                145deg,
                rgba(20, 27, 37, 0.88),
                rgba(14, 20, 29, 0.84)
            );
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.13),
            inset 0 1px 0 rgba(255, 255, 255, 0.02);
        transition:
            border-color 140ms ease,
            transform 140ms ease,
            box-shadow 140ms ease;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: var(--panel-border-hover) !important;
        box-shadow:
            0 16px 34px rgba(0, 0, 0, 0.17),
            0 0 0 1px var(--team-primary-faint);
    }}

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] h4 {{
        color: #F8FAFC;
        margin-bottom: 0.55rem;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 14px !important;
        overflow: hidden;
        background: rgba(15, 21, 30, 0.66);
        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.018);
    }}

    [data-testid="stExpander"] details > summary {{
        min-height: 2.8rem;
        padding-top: 0.18rem;
        padding-bottom: 0.18rem;
        color: #E8EDF4;
        font-weight: 690;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: rgba(148, 163, 184, 0.15) !important;
        background: rgba(13, 19, 28, 0.52);
        box-shadow: none;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        overflow: hidden;
        box-shadow:
            0 7px 20px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.018);
    }}

    [data-testid="stCodeBlock"] {{
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        overflow: hidden;
        background:
            linear-gradient(
                145deg,
                rgba(22, 28, 38, 0.92),
                rgba(14, 19, 28, 0.94)
            );
        box-shadow:
            0 7px 20px rgba(0, 0, 0, 0.09),
            inset 0 1px 0 rgba(255, 255, 255, 0.018);
    }}

    [data-testid="stCodeBlock"] pre {{
        padding: 1.08rem 1.15rem !important;
        color: #EAF0F7 !important;
        font-size: 0.90rem !important;
        line-height: 1.66 !important;
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        flex-wrap: nowrap;
        gap: 0.18rem;
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-width: none;
        -ms-overflow-style: none;
        scroll-behavior: smooth;
        scroll-snap-type: x proximity;
        border-bottom: 1px solid var(--panel-border);
        padding: 0.15rem 0.12rem 0;
        scroll-padding-inline: 0.75rem;
    }}

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
        display: none;
        width: 0;
        height: 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        flex: 0 0 auto;
        min-width: max-content;
        height: 2.72rem;
        padding-left: 0.78rem;
        padding-right: 0.78rem;
        border-radius: 9px 9px 0 0;
        color: #D8E0EA;
        font-weight: 680;
        white-space: nowrap;
        scroll-snap-align: start;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: #FFFFFF;
        background: var(--team-primary-faint);
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--team-secondary) !important;
        background:
            linear-gradient(
                180deg,
                var(--team-primary-soft),
                var(--team-primary-faint)
            ) !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background:
            linear-gradient(
                90deg,
                var(--team-primary),
                var(--team-secondary)
            ) !important;
        height: 3px;
        border-radius: 999px 999px 0 0;
    }}

    .stTabs [data-baseweb="tab"]:focus-visible,
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    input:focus-visible,
    textarea:focus-visible {{
        outline: 2px solid var(--team-secondary) !important;
        outline-offset: 2px;
        box-shadow: 0 0 0 4px var(--team-secondary-soft) !important;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        min-height: 2.58rem;
        border-radius: 11px;
        border-color: {hex_to_rgba(active_primary, 0.62)};
        background: rgba(22, 28, 38, 0.84);
        transition:
            transform 120ms ease,
            box-shadow 120ms ease,
            border-color 120ms ease,
            background 120ms ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        border-color: var(--team-secondary);
        background: var(--team-primary-faint);
        box-shadow: 0 8px 22px var(--team-primary-soft);
    }}

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {{
        background:
            linear-gradient(
                135deg,
                var(--team-primary),
                {hex_to_rgba(active_primary, 0.80)}
            );
        color: var(--team-primary-text);
        box-shadow: 0 8px 22px var(--team-primary-soft);
    }}

    [data-testid="stAlert"] {{
        border-radius: 13px;
        border: 1px solid var(--panel-border);
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.10);
    }}

    hr {{
        border-color: var(--panel-border) !important;
        margin-top: 1.45rem !important;
        margin-bottom: 1.45rem !important;
    }}

    .team-theme-banner {{
        display: inline-flex;
        align-items: center;
        gap: 0.58rem;
        margin: 0.45rem 0 0.8rem;
        padding: 0.46rem 0.78rem;
        border: 1px solid {hex_to_rgba(active_primary, 0.52)};
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                var(--team-primary-soft),
                var(--team-secondary-soft)
            );
        color: #EEF3F8;
        font-size: 0.83rem;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.12);
    }}

    .team-theme-dot {{
        width: 0.67rem;
        height: 0.67rem;
        border-radius: 999px;
        background: var(--team-primary);
        box-shadow: 0 0 0 4px var(--team-primary-soft);
    }}

    .section-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.38rem;
        color: var(--team-secondary);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }}

    .section-eyebrow::before {{
        content: "";
        width: 1.6rem;
        height: 2px;
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                var(--team-primary),
                var(--team-secondary)
            );
    }}

    .share-summary-shell,
    .model-summary-shell {{
        margin: 0.45rem 0 1.1rem;
        padding: 1rem;
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        background:
            radial-gradient(
                circle at 100% 0%,
                var(--team-primary-soft),
                transparent 24rem
            ),
            linear-gradient(
                145deg,
                rgba(20, 27, 37, 0.92),
                rgba(12, 18, 27, 0.92)
            );
        box-shadow:
            0 16px 38px rgba(0, 0, 0, 0.16),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }}

    .selection-ribbon {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.48rem;
        margin-bottom: 0.9rem;
    }}

    .selection-ribbon-label {{
        color: #F8FAFC;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }}

    .selection-ribbon-item {{
        padding: 0.33rem 0.58rem;
        border: 1px solid var(--panel-border);
        border-radius: 999px;
        background: rgba(9, 14, 22, 0.62);
        color: #C7D1DE;
        font-size: 0.79rem;
        font-weight: 650;
        line-height: 1.2;
    }}

    .share-summary-grid,
    .model-summary-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.78rem;
    }}

    .share-summary-card,
    .model-summary-card {{
        min-width: 0;
        min-height: 112px;
        padding: 0.9rem 0.95rem 0.85rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-top: 3px solid var(--team-primary);
        border-radius: 14px;
        background:
            linear-gradient(
                150deg,
                rgba(29, 37, 49, 0.92),
                rgba(16, 22, 31, 0.94)
            );
        box-shadow:
            0 9px 24px rgba(0, 0, 0, 0.13),
            inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }}

    .share-summary-label,
    .model-summary-label {{
        color: #AEB9C7;
        font-size: 0.76rem;
        font-weight: 720;
        letter-spacing: 0.025em;
    }}

    .share-summary-value,
    .model-summary-value {{
        margin-top: 0.32rem;
        color: #F8FAFC;
        font-size: clamp(1.4rem, 2.05vw, 2rem);
        font-weight: 760;
        letter-spacing: -0.035em;
        line-height: 1.04;
        overflow-wrap: anywhere;
    }}

    .share-summary-detail,
    .model-summary-detail {{
        margin-top: 0.45rem;
        color: #8FA0B3;
        font-size: 0.76rem;
        font-weight: 600;
        line-height: 1.32;
    }}

    .share-secondary-line {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.15rem;
        margin-top: 0.8rem;
        padding: 0.72rem 0.78rem 0.1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        color: #AEB9C7;
        font-size: 0.8rem;
    }}

    .share-secondary-line strong {{
        color: #EDF2F7;
        font-weight: 760;
    }}

    .model-takeaway {{
        margin-top: 0.85rem;
        padding: 0.82rem 0.92rem;
        border-left: 4px solid var(--team-secondary);
        border-radius: 10px;
        background: rgba(9, 14, 22, 0.58);
        color: #CFD8E4;
        font-size: 0.86rem;
        line-height: 1.55;
    }}

    .model-takeaway strong {{
        color: #FFFFFF;
    }}

    .metric-delta-positive {{
        color: #74D69A;
    }}

    .metric-delta-neutral {{
        color: #AEB9C7;
    }}

    .content-card-title {{
        margin-bottom: 0.1rem;
        color: #F8FAFC;
        font-size: 1.08rem;
        font-weight: 760;
        letter-spacing: -0.018em;
    }}

    .content-card-subtitle {{
        margin-bottom: 0.75rem;
        color: #95A5B8;
        font-size: 0.82rem;
        line-height: 1.45;
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        max-width: 100% !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }}

    @media (max-width: 1120px) {{
        .share-summary-grid,
        .model-summary-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    @media (max-width: 900px) {{
        .block-container {{
            padding-left: 0.9rem;
            padding-right: 0.9rem;
            padding-top: 1.4rem;
        }}

        [data-testid="stMetric"] {{
            min-height: 98px;
        }}

        [data-testid="stMetricValue"] > div {{
            font-size: clamp(1.3rem, 5vw, 1.85rem) !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 2.55rem;
            padding-left: 0.66rem;
            padding-right: 0.66rem;
        }}

        .share-summary-shell,
        .model-summary-shell {{
            padding: 0.78rem;
            border-radius: 15px;
        }}
    }}

    ::selection {{
        background: var(--team-primary-soft);
        color: #FFFFFF;
    }}

    * {{
        scrollbar-width: thin;
        scrollbar-color:
            rgba(148, 163, 184, 0.36)
            rgba(15, 20, 28, 0.34);
    }}

    *::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    *::-webkit-scrollbar-track {{
        background: rgba(15, 20, 28, 0.34);
    }}

    *::-webkit-scrollbar-thumb {{
        border: 2px solid rgba(15, 20, 28, 0.34);
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.36);
    }}

    *::-webkit-scrollbar-thumb:hover {{
        background: rgba(148, 163, 184, 0.56);
    }}

    @media (max-width: 620px) {{
        .share-summary-grid,
        .model-summary-grid {{
            grid-template-columns: 1fr;
        }}

        .share-summary-card,
        .model-summary-card {{
            min-height: 96px;
        }}

        [data-testid="stCodeBlock"] pre {{
            padding: 0.92rem 0.95rem !important;
            font-size: 0.84rem !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


if active_team:
    st.sidebar.caption(
        f"Visual theme: {active_team_visual['name']}"
    )
    st.markdown(
        (
            '<div class="team-theme-banner">'
            '<span class="team-theme-dot"></span>'
            f"{active_team_visual['name']} visual theme active"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


def polish_plotly_figure(figure):
    """Apply one consistent visual language to every Plotly chart."""

    existing_title = getattr(
        getattr(figure.layout, "title", None),
        "text",
        None,
    )
    top_margin = 68 if existing_title else 28

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=active_colorway,
        font={
            "family": (
                "Inter, ui-sans-serif, system-ui, -apple-system, "
                "BlinkMacSystemFont, Segoe UI, sans-serif"
            ),
            "color": "#E8EDF4",
            "size": 13,
        },
        hoverlabel={
            "bgcolor": "#171C26",
            "bordercolor": active_primary,
            "font": {
                "color": "#F8FAFC",
                "size": 13,
            },
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12},
            "title": {"font": {"size": 12}},
        },
        margin={
            "l": 35,
            "r": 24,
            "t": top_margin,
            "b": 42,
        },
    )

    if existing_title:
        figure.update_layout(
            title_font={
                "size": 19,
                "color": "#F8FAFC",
            }
        )
    else:
        # Leave untitled figures completely untouched. In some
        # Plotly/Streamlit combinations, update_layout(title=None)
        # serializes an empty title object that renders as "undefined".
        figure.layout.title = None

    figure.update_xaxes(
        showgrid=False,
        zerolinecolor="rgba(255,255,255,0.16)",
        linecolor="rgba(255,255,255,0.12)",
        tickfont={"color": "#D7DEE8"},
        title_font={"color": "#D7DEE8"},
    )
    figure.update_yaxes(
        gridcolor="rgba(255,255,255,0.10)",
        zerolinecolor="rgba(255,255,255,0.18)",
        linecolor="rgba(255,255,255,0.12)",
        tickfont={"color": "#D7DEE8"},
        title_font={"color": "#D7DEE8"},
    )

    try:
        figure.update_traces(
            marker_cornerradius=5,
            selector={"type": "bar"},
        )
    except (TypeError, ValueError):
        pass

    return figure


def render_plotly_chart(figure, *args, **kwargs):
    """Render a polished Plotly figure with a quieter mode bar."""

    if isinstance(figure, go.Figure):
        figure = polish_plotly_figure(figure)

    supplied_config = kwargs.pop("config", None) or {}
    chart_config = {
        **PLOTLY_CONFIG,
        **supplied_config,
    }
    return st.plotly_chart(
        figure,
        *args,
        config=chart_config,
        **kwargs,
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



def describe_selected_values(
    values,
    empty_label,
    item_label,
):
    """Summarize a sidebar selection without creating a long label."""

    values = list(values)

    if not values:
        return empty_label

    if len(values) == 1:
        return str(values[0])

    if len(values) <= 3:
        return ", ".join(str(value) for value in values)

    return f"{len(values)} selected {item_label}"


def describe_season_window(seasons):
    """Format selected seasons for short narrative text."""

    seasons = sorted(int(season) for season in seasons)

    if not seasons:
        return "all available seasons"

    if len(seasons) == 1:
        return str(seasons[0])

    if seasons == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]} to {seasons[-1]}"

    if len(seasons) <= 4:
        return ", ".join(str(season) for season in seasons)

    return (
        f"{seasons[0]} to {seasons[-1]} "
        f"({len(seasons)} selected seasons)"
    )


def describe_season_types(season_types):
    """Translate season-type codes into readable text."""

    selected = set(season_types)

    if not selected or selected == {"REG", "POST"}:
        return "regular season and postseason"

    if selected == {"REG"}:
        return "regular season"

    if selected == {"POST"}:
        return "postseason"

    return "selected season types"


def describe_insight_subject(
    selected_coaches,
    selected_teams,
):
    """Create a natural subject for automatic insight sentences."""

    if len(selected_coaches) == 1 and len(selected_teams) == 1:
        return f"{selected_coaches[0]} with {selected_teams[0]}"

    if len(selected_coaches) == 1:
        return selected_coaches[0]

    if len(selected_teams) == 1:
        return selected_teams[0]

    return "The selected sample"


def render_insight_card(title, body):
    """Render one compact narrative card."""

    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(body)


def load_overview_style_summary(
    overview_where_clause,
    overview_parameters,
):
    """Load filtered, directly observable style rates."""

    return connection.execute(
        f"""
        SELECT
            COUNT(*) AS plays,
            SUM(CASE
                WHEN pass_depth_bucket IS NOT NULL
                THEN 1 ELSE 0
            END) AS charted_throw_depth_plays,
            SUM(CASE
                WHEN pass_direction IS NOT NULL
                THEN 1 ELSE 0
            END) AS charted_pass_direction_plays,
            SUM(CASE
                WHEN run_direction IS NOT NULL
                THEN 1 ELSE 0
            END) AS charted_run_direction_plays,
            100.0 * AVG(shotgun) AS shotgun_pct,
            100.0 * AVG(no_huddle) AS no_huddle_pct,
            100.0
                * SUM(CASE
                    WHEN pass_depth_bucket = 'Deep (20+)'
                    THEN 1 ELSE 0
                END)
                / NULLIF(SUM(CASE
                    WHEN pass_depth_bucket IS NOT NULL
                    THEN 1 ELSE 0
                END), 0) AS deep_throw_pct,
            100.0
                * SUM(CASE
                    WHEN pass_direction = 'Middle'
                    THEN 1 ELSE 0
                END)
                / NULLIF(SUM(CASE
                    WHEN pass_direction IS NOT NULL
                    THEN 1 ELSE 0
                END), 0) AS middle_throw_pct,
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
        {overview_where_clause}
        """,
        overview_parameters,
    ).fetchdf().iloc[0]


def build_style_insight(style_summary, subject):
    """Identify the largest well-supported displayed style share."""

    total_plays = int(style_summary["plays"])
    charted_throw_depth = int(
        style_summary["charted_throw_depth_plays"]
    )
    charted_pass_direction = int(
        style_summary["charted_pass_direction_plays"]
    )
    charted_run_direction = int(
        style_summary["charted_run_direction_plays"]
    )

    candidates = [
        {
            "rate": style_summary["shotgun_pct"],
            "sample": total_plays,
            "text": (
                "shotgun presentation was recorded on "
                "{rate:.1f}% of {sample:,} filtered plays"
            ),
        },
        {
            "rate": style_summary["no_huddle_pct"],
            "sample": total_plays,
            "text": (
                "no-huddle tempo was recorded on "
                "{rate:.1f}% of {sample:,} filtered plays"
            ),
        },
        {
            "rate": style_summary["deep_throw_pct"],
            "sample": charted_throw_depth,
            "text": (
                "deep throws accounted for {rate:.1f}% of "
                "{sample:,} throws with recorded depth"
            ),
        },
        {
            "rate": style_summary["middle_throw_pct"],
            "sample": charted_pass_direction,
            "text": (
                "middle-field throws accounted for {rate:.1f}% of "
                "{sample:,} throws with recorded direction"
            ),
        },
        {
            "rate": style_summary["outside_run_pct"],
            "sample": charted_run_direction,
            "text": (
                "outside runs accounted for {rate:.1f}% of "
                "{sample:,} runs with recorded direction"
            ),
        },
    ]

    supported_candidates = [
        candidate
        for candidate in candidates
        if candidate["sample"] >= 25
        and pd.notna(candidate["rate"])
    ]

    if not supported_candidates:
        return (
            f"{subject} does not have enough charted plays in the current "
            "selection to summarize a stable style share."
        )

    largest = max(
        supported_candidates,
        key=lambda candidate: candidate["rate"],
    )

    detail = largest["text"].format(
        rate=largest["rate"],
        sample=largest["sample"],
    )

    return (
        f"For {subject.lower() if subject.startswith('The ') else subject}, "
        f"{detail}. This is the largest displayed observable style share "
        "in the current filtered sample."
    )


def load_matching_coach_uncertainty():
    """Load full-season intervals matching selected coaches, teams, seasons."""

    if not selected_coaches:
        return pd.DataFrame()

    conditions = ["meets_minimum_sample = true"]
    parameters = []

    def add_uncertainty_filter(column_name, values):
        if not values:
            return

        placeholders = ", ".join(["?"] * len(values))
        conditions.append(
            f"{column_name} IN ({placeholders})"
        )
        parameters.extend(values)

    add_uncertainty_filter("season", selected_seasons)
    add_uncertainty_filter("head_coach", selected_coaches)
    add_uncertainty_filter("posteam", selected_teams)

    uncertainty_where = (
        "WHERE " + " AND ".join(conditions)
    )

    return connection.execute(
        f"""
        SELECT
            season,
            head_coach,
            posteam,
            plays,
            model_pass_oe_pct,
            ci_95_lower_pct,
            ci_95_upper_pct,
            tendency_label
        FROM coach_uncertainty
        {uncertainty_where}
        ORDER BY
            season,
            ABS(model_pass_oe_pct) DESC
        """,
        parameters,
    ).fetchdf()


def build_confidence_insight(uncertainty_data):
    """Explain whether matching full-season intervals exclude zero."""

    if not selected_coaches:
        return (
            "Select at least one head coach to add a full-season confidence "
            "check. These intervals do not change with down, distance, field "
            "position, quarter, or score filters."
        )

    if uncertainty_data.empty:
        return (
            "No matching coach-team-season meets the app's minimum sample "
            "requirement for a full-season confidence interval."
        )

    data = uncertainty_data.copy()
    data["excludes_zero"] = (
        (data["ci_95_lower_pct"] > 0)
        | (data["ci_95_upper_pct"] < 0)
    )

    strongest = data.loc[
        data["model_pass_oe_pct"].abs().idxmax()
    ]

    lower = strongest["ci_95_lower_pct"]
    upper = strongest["ci_95_upper_pct"]

    if lower > 0:
        interval_message = (
            "The interval stays above zero, supporting a descriptively "
            "pass-heavy classification for that complete season."
        )
    elif upper < 0:
        interval_message = (
            "The interval stays below zero, supporting a descriptively "
            "run-heavy classification for that complete season."
        )
    else:
        interval_message = (
            "The interval crosses zero, so the full-season tendency remains "
            "uncertain."
        )

    group_count = len(data)
    excluding_count = int(data["excludes_zero"].sum())

    if group_count == 1:
        prefix = "The matching full-season sample"
    else:
        prefix = (
            f"Among {group_count} matching full-season coach-team samples, "
            f"{excluding_count} confidence intervals exclude zero. The "
            "largest absolute Pass OE point estimate"
        )

    if group_count == 1:
        sentence = (
            f"{prefix} is {strongest['head_coach']} "
            f"({strongest['posteam']}, {int(strongest['season'])}): "
            f"{strongest['model_pass_oe_pct']:+.2f} points with a 95% "
            f"confidence interval from {lower:+.2f} to {upper:+.2f}."
        )
    else:
        sentence = (
            f"{prefix} is {strongest['head_coach']} "
            f"({strongest['posteam']}, {int(strongest['season'])}): "
            f"{strongest['model_pass_oe_pct']:+.2f} points with a 95% "
            f"confidence interval from {lower:+.2f} to {upper:+.2f}."
        )

    return (
        f"{sentence} {interval_message} The interval applies to the complete "
        "coach-team season, not the smaller situational subset currently "
        "shown elsewhere on this page."
    )


def load_third_down_distance_split(
    overview_where_clause,
    overview_parameters,
):
    """Compare short and long third-down pass rates when both are supported."""

    if selected_downs and 3 not in selected_downs:
        return pd.DataFrame()

    return connection.execute(
        f"""
        SELECT
            CASE
                WHEN ydstogo >= 7 THEN '7+ yards'
                ELSE '1-6 yards'
            END AS distance_group,
            COUNT(*) AS plays,
            100.0 * AVG(is_pass) AS pass_rate_pct
        FROM play_predictions
        {overview_where_clause}
            AND down = 3
            AND ydstogo >= 1
        GROUP BY distance_group
        ORDER BY distance_group
        """,
        overview_parameters,
    ).fetchdf()


def build_third_down_insight(third_down_data):
    """Create a descriptive third-down distance comparison."""

    if third_down_data.empty:
        return None

    lookup = third_down_data.set_index("distance_group")

    required_groups = {"1-6 yards", "7+ yards"}
    if not required_groups.issubset(set(lookup.index)):
        return None

    short_plays = int(lookup.loc["1-6 yards", "plays"])
    long_plays = int(lookup.loc["7+ yards", "plays"])

    if min(short_plays, long_plays) < 50:
        return None

    short_rate = float(
        lookup.loc["1-6 yards", "pass_rate_pct"]
    )
    long_rate = float(
        lookup.loc["7+ yards", "pass_rate_pct"]
    )
    difference = long_rate - short_rate

    if abs(difference) < 3:
        comparison = (
            "was similar between the two distance groups"
        )
    elif difference > 0:
        comparison = (
            f"was {difference:.1f} percentage points higher when "
            "the offense needed at least seven yards"
        )
    else:
        comparison = (
            f"was {abs(difference):.1f} percentage points lower when "
            "the offense needed at least seven yards"
        )

    return (
        f"On third down, the observed pass rate {comparison}: "
        f"{long_rate:.1f}% on 7+ yards versus {short_rate:.1f}% on 1 to 6 "
        f"yards. The comparison uses {long_plays:,} and {short_plays:,} "
        "plays respectively and is descriptive rather than causal."
    )



def format_profile_period(first_season, last_season):
    """Format the season range represented by one profile sample."""

    first_season = int(first_season)
    last_season = int(last_season)

    if first_season == last_season:
        return str(first_season)

    return f"{first_season} to {last_season}"


def profile_sample_label(plays):
    """Classify profile sample size using the app's existing guidance."""

    plays = int(plays)

    if plays >= 500:
        return "Stronger sample"

    if plays >= 200:
        return "Limited sample"

    return "Exploratory sample"


def build_profile_tendency_statement(row):
    """Describe a filtered profile's run-pass tendency."""

    pass_oe = float(row["pass_oe_pct"])
    actual_rate = float(row["actual_pass_rate_pct"])
    expected_rate = float(row["expected_pass_rate_pct"])
    plays = int(row["plays"])
    period = format_profile_period(
        row["first_season"],
        row["last_season"],
    )

    if abs(pass_oe) < 1:
        direction_text = (
            "called passes at a rate close to the league model's "
            "expectation"
        )
    elif pass_oe > 0:
        direction_text = (
            f"passed {pass_oe:.2f} percentage points above the "
            "league model's expectation"
        )
    else:
        direction_text = (
            f"passed {abs(pass_oe):.2f} percentage points below the "
            "league model's expectation"
        )

    return (
        f"Across {plays:,} filtered plays from {period}, this profile "
        f"{direction_text}. The observed pass rate was {actual_rate:.2f}% "
        f"compared with an expected rate of {expected_rate:.2f}%."
    )


def build_profile_outcome_statement(row):
    """Describe outcomes without assigning a causal effect."""

    return (
        f"The same filtered plays averaged {row['mean_epa']:.3f} EPA per "
        f"play, a {row['success_rate_pct']:.2f}% success rate, and "
        f"{row['yards_per_play']:.2f} yards per play. These are descriptive "
        "outcomes associated with the sample and do not isolate the effect "
        "of the decision-maker."
    )


def build_profile_style_statement(row, subject):
    """Describe the largest supported style share for a profile row."""

    style_summary = pd.Series({
        "plays": row.get("style_plays", 0),
        "charted_throw_depth_plays": row.get(
            "charted_throw_depth_plays",
            0,
        ),
        "charted_pass_direction_plays": row.get(
            "charted_pass_direction_plays",
            0,
        ),
        "charted_run_direction_plays": row.get(
            "charted_run_direction_plays",
            0,
        ),
        "shotgun_pct": row.get("shotgun_pct"),
        "no_huddle_pct": row.get("no_huddle_pct"),
        "deep_throw_pct": row.get("deep_throw_pct"),
        "middle_throw_pct": row.get("middle_throw_pct"),
        "outside_run_pct": row.get("outside_run_pct"),
    })

    if pd.isna(style_summary["plays"]):
        return (
            "No matching play-style rows are available for the current "
            "profile filters."
        )

    return build_style_insight(style_summary, subject)


def build_profile_confidence_statement(uncertainty_data):
    """Summarize complete-season confidence evidence for one team."""

    if uncertainty_data.empty:
        return (
            "No matching complete-season sample meets the app's minimum "
            "requirement for a confidence interval."
        )

    data = uncertainty_data.copy()
    data["excludes_zero"] = (
        (data["ci_95_lower_pct"] > 0)
        | (data["ci_95_upper_pct"] < 0)
    )

    strongest = data.loc[
        data["model_pass_oe_pct"].abs().idxmax()
    ]
    excluding_count = int(data["excludes_zero"].sum())
    group_count = len(data)
    lower = float(strongest["ci_95_lower_pct"])
    upper = float(strongest["ci_95_upper_pct"])

    if lower > 0:
        classification = (
            "The interval remains above zero, supporting a descriptively "
            "pass-heavy classification for that complete season."
        )
    elif upper < 0:
        classification = (
            "The interval remains below zero, supporting a descriptively "
            "run-heavy classification for that complete season."
        )
    else:
        classification = (
            "The interval crosses zero, so that complete-season tendency "
            "remains uncertain."
        )

    return (
        f"{excluding_count} of {group_count} matching complete-season "
        f"confidence intervals exclude zero. The largest absolute point "
        f"estimate is {int(strongest['season'])}: "
        f"{strongest['model_pass_oe_pct']:+.2f} points with a 95% interval "
        f"from {lower:+.2f} to {upper:+.2f}. {classification} These "
        "intervals do not change with situation filters."
    )


def load_profile_uncertainty(
    entity,
    entity_column,
    source_table,
):
    """Load complete-season uncertainty rows for a profile selection."""

    conditions = [
        "meets_minimum_sample = true",
        f"{entity_column} = ?",
    ]
    parameters = [entity]

    def add_profile_uncertainty_filter(column_name, values):
        if not values:
            return

        placeholders = ", ".join(["?"] * len(values))
        conditions.append(
            f"{column_name} IN ({placeholders})"
        )
        parameters.extend(values)

    add_profile_uncertainty_filter(
        "season",
        selected_seasons,
    )
    add_profile_uncertainty_filter(
        "posteam",
        selected_teams,
    )

    uncertainty_where = (
        "WHERE " + " AND ".join(conditions)
    )

    return connection.execute(
        f"""
        SELECT
            season,
            {entity_column} AS entity,
            posteam,
            plays,
            model_pass_oe_pct,
            ci_95_lower_pct,
            ci_95_upper_pct,
            tendency_label
        FROM {source_table}
        {uncertainty_where}
        ORDER BY
            posteam,
            season
        """,
        parameters,
    ).fetchdf()


def describe_export_values(values, empty_label):
    """Format complete selected values for exported reports."""

    if not values:
        return empty_label

    return ", ".join(str(value) for value in values)


def export_sample_quality(plays):
    """Return the app's descriptive sample-quality label."""

    plays = int(plays)

    if plays >= 500:
        return "Stronger sample (500+)"

    if plays >= 200:
        return "Limited sample (200-499)"

    return "Exploratory sample (<200)"


def _sample_quality_parts(plays):
    """Return a short title and detail for sample-quality cards."""

    plays = int(plays)

    if plays >= 500:
        return "Stronger sample", "500+ filtered plays"

    if plays >= 200:
        return "Limited sample", "200-499 filtered plays"

    return "Exploratory sample", "Fewer than 200 plays"


def render_share_summary_panel(summary, selection_details):
    """Render a responsive Share Findings overview without clipped text."""

    pass_oe_points = 100 * float(summary["pass_oe"])
    quality_title, quality_detail = _sample_quality_parts(
        summary["plays"]
    )

    if pass_oe_points > 0.5:
        pass_detail = "More passing than modeled"
    elif pass_oe_points < -0.5:
        pass_detail = "More rushing than modeled"
    else:
        pass_detail = "Close to modeled expectation"

    ribbon_values = [
        selection_details["Seasons"],
        selection_details["Season type"],
        selection_details["Head coaches"],
        selection_details["Offensive teams"],
    ]
    ribbon_html = "".join(
        (
            "<span class='selection-ribbon-item'>"
            f"{html.escape(str(value))}"
            "</span>"
        )
        for value in ribbon_values
    )

    cards = [
        (
            "Filtered plays",
            f"{int(summary['plays']):,}",
            "Active sidebar selection",
        ),
        (
            "Games",
            f"{int(summary['games']):,}",
            "Distinct games represented",
        ),
        (
            "Pass OE",
            f"{pass_oe_points:+.2f} pts",
            pass_detail,
        ),
        (
            "Sample quality",
            quality_title,
            quality_detail,
        ),
    ]
    cards_html = "".join(
        (
            "<article class='share-summary-card'>"
            f"<div class='share-summary-label'>{html.escape(label)}</div>"
            f"<div class='share-summary-value'>{html.escape(value)}</div>"
            f"<div class='share-summary-detail'>{html.escape(detail)}</div>"
            "</article>"
        )
        for label, value, detail in cards
    )

    secondary_items = [
        (
            "Actual pass rate",
            f"{100 * float(summary['actual_pass_rate']):.2f}%",
        ),
        (
            "Expected pass rate",
            f"{100 * float(summary['expected_pass_rate']):.2f}%",
        ),
        (
            "EPA/play",
            f"{float(summary['mean_epa']):.3f}",
        ),
        (
            "Success rate",
            f"{100 * float(summary['success_rate']):.2f}%",
        ),
        (
            "Yards/play",
            f"{float(summary['yards_per_play']):.2f}",
        ),
    ]
    secondary_html = "".join(
        (
            "<span>"
            f"{html.escape(label)} "
            f"<strong>{html.escape(value)}</strong>"
            "</span>"
        )
        for label, value in secondary_items
    )

    st.markdown(
        (
            "<section class='share-summary-shell'>"
            "<div class='selection-ribbon'>"
            "<span class='selection-ribbon-label'>Active selection</span>"
            f"{ribbon_html}"
            "</div>"
            "<div class='share-summary-grid'>"
            f"{cards_html}"
            "</div>"
            "<div class='share-secondary-line'>"
            f"{secondary_html}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _find_metric_row(metrics, model_name):
    """Return one model row when present, otherwise an empty Series."""

    matching = metrics.loc[metrics["model"] == model_name]
    if matching.empty:
        return pd.Series(dtype="object")
    return matching.iloc[0]


def render_model_performance_summary(final_metrics):
    """Render polished held-out model cards and a benchmark takeaway."""

    lab_row = _find_metric_row(
        final_metrics,
        "final_hist_gradient_boosting",
    )
    benchmark_row = _find_metric_row(
        final_metrics,
        "nflverse_xpass",
    )

    if lab_row.empty:
        return

    cards = [
        (
            "Held-out plays",
            f"{int(lab_row['plays']):,}",
            "2025 final test sample",
        ),
        (
            "ROC AUC",
            f"{float(lab_row['roc_auc']):.4f}",
            "Higher is better",
        ),
        (
            "Brier score",
            f"{float(lab_row['brier_score']):.4f}",
            "Lower is better",
        ),
        (
            "Calibration error",
            f"{float(lab_row['expected_calibration_error']):.4f}",
            "Lower is better",
        ),
    ]

    cards_html = "".join(
        (
            "<article class='model-summary-card'>"
            f"<div class='model-summary-label'>{html.escape(label)}</div>"
            f"<div class='model-summary-value'>{html.escape(value)}</div>"
            f"<div class='model-summary-detail'>{html.escape(detail)}</div>"
            "</article>"
        )
        for label, value, detail in cards
    )

    takeaway = (
        "The final locked model is shown without a benchmark comparison "
        "because the nflverse xpass row was not available."
    )

    if not benchmark_row.empty:
        auc_delta = (
            float(lab_row["roc_auc"])
            - float(benchmark_row["roc_auc"])
        )
        brier_delta = (
            float(benchmark_row["brier_score"])
            - float(lab_row["brier_score"])
        )
        calibration_delta = (
            float(benchmark_row["expected_calibration_error"])
            - float(lab_row["expected_calibration_error"])
        )
        log_loss_delta = (
            float(benchmark_row["log_loss"])
            - float(lab_row["log_loss"])
        )

        takeaway = (
            "On the held-out 2025 sample, the locked Coaching Lab model "
            f"improved ROC AUC by {auc_delta:+.4f}, reduced Brier score by "
            f"{brier_delta:.4f}, reduced log loss by {log_loss_delta:.4f}, "
            f"and reduced expected calibration error by "
            f"{calibration_delta:.4f} relative to nflverse xpass."
        )

    st.markdown(
        (
            "<section class='model-summary-shell'>"
            "<div class='model-summary-grid'>"
            f"{cards_html}"
            "</div>"
            "<div class='model-takeaway'>"
            "<strong>Held-out takeaway.</strong> "
            f"{html.escape(takeaway)}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def build_share_selection_details():
    """Capture every active sidebar filter in readable form."""

    down_text = (
        "All downs"
        if set(selected_downs) == {1, 2, 3, 4}
        else describe_export_values(
            [f"Down {down}" for down in selected_downs],
            "All downs",
        )
    )

    quarter_text = (
        "All quarters and overtime"
        if set(selected_quarters) == {1, 2, 3, 4, 5}
        else describe_export_values(
            [
                "Overtime" if quarter == 5 else f"Quarter {quarter}"
                for quarter in selected_quarters
            ],
            "All quarters and overtime",
        )
    )

    return {
        "Seasons": describe_season_window(selected_seasons),
        "Season type": describe_season_types(selected_season_types),
        "Head coaches": describe_export_values(
            selected_coaches,
            "All head coaches",
        ),
        "Offensive teams": describe_export_values(
            selected_teams,
            "All offensive teams",
        ),
        "Down": down_text,
        "Quarter": quarter_text,
        "Yards to go": (
            f"{distance_bounds[0]} to {distance_bounds[1]}"
        ),
        "Field position": (
            f"{yardline_bounds[0]} to {yardline_bounds[1]} yards "
            "from the opponent end zone"
        ),
        "Score differential": (
            f"{score_bounds[0]:+d} to {score_bounds[1]:+d}"
        ),
    }


def build_share_insights(
    summary,
    style_summary,
    uncertainty_data,
    third_down_data,
):
    """Build export-ready versions of the automatic insight cards."""

    subject = describe_insight_subject(
        selected_coaches,
        selected_teams,
    )
    season_window = describe_season_window(selected_seasons)
    coaches = describe_selected_values(
        selected_coaches,
        "all head coaches",
        "head coaches",
    )
    teams = describe_selected_values(
        selected_teams,
        "all offensive teams",
        "teams",
    )
    season_types = describe_season_types(
        selected_season_types
    )

    insights = [
        {
            "title": "Selection Context",
            "body": (
                f"{coaches}; {teams}; {season_window}; {season_types}. "
                f"The current selection contains {int(summary['plays']):,} "
                f"plays across {int(summary['games']):,} games."
            ),
        }
    ]

    pass_oe_points = 100 * float(summary["pass_oe"])

    if pass_oe_points >= 0.5:
        tendency_text = (
            f"{subject} passed {pass_oe_points:.2f} percentage points "
            "above the league model's expectation in the selected "
            "situations."
        )
    elif pass_oe_points <= -0.5:
        tendency_text = (
            f"{subject} passed {abs(pass_oe_points):.2f} percentage "
            "points below the league model's expectation, indicating a "
            "more run-leaning call mix in the selected situations."
        )
    else:
        tendency_text = (
            f"{subject} stayed close to modeled expectation, with Pass "
            f"OE of {pass_oe_points:+.2f} percentage points in the "
            "selected situations."
        )

    insights.append({
        "title": "Run-Pass Tendency",
        "body": (
            f"{tendency_text} This describes call selection relative to "
            "context, not whether passing or rushing more often caused "
            "better results."
        ),
    })

    insights.append({
        "title": "Largest Style Share",
        "body": build_style_insight(
            style_summary,
            subject,
        ),
    })

    insights.append({
        "title": "Confidence Check",
        "body": build_confidence_insight(
            uncertainty_data
        ),
    })

    play_count = int(summary["plays"])
    if play_count < 200:
        sample_message = (
            f"This filtered sample contains only {play_count:,} plays, "
            "so treat the result as exploratory and sensitive to a small "
            "number of games."
        )
    elif play_count < 500:
        sample_message = (
            f"This filtered sample contains {play_count:,} plays. It is "
            "large enough to inspect, but remains below the app's "
            "stronger 500-play descriptive threshold."
        )
    else:
        sample_message = (
            f"This filtered sample contains {play_count:,} plays and "
            "meets the app's stronger descriptive threshold. Sample size "
            "does not remove confounding or make the result causal."
        )

    insights.append({
        "title": "Sample Quality",
        "body": sample_message,
    })

    third_down_insight = build_third_down_insight(
        third_down_data
    )
    if third_down_insight is not None:
        insights.append({
            "title": "Third-Down Distance Split",
            "body": third_down_insight,
        })

    return insights


def strongest_style_share_for_export(style_summary):
    """Return one concise supported style observation for sharing."""

    candidates = [
        {
            "rate": style_summary["shotgun_pct"],
            "sample": int(style_summary["plays"]),
            "text": (
                "shotgun was recorded on {rate:.1f}% of {sample:,} plays"
            ),
        },
        {
            "rate": style_summary["no_huddle_pct"],
            "sample": int(style_summary["plays"]),
            "text": (
                "no-huddle was recorded on {rate:.1f}% of {sample:,} plays"
            ),
        },
        {
            "rate": style_summary["deep_throw_pct"],
            "sample": int(
                style_summary["charted_throw_depth_plays"]
            ),
            "text": (
                "deep throws represented {rate:.1f}% of {sample:,} "
                "throws with recorded depth"
            ),
        },
        {
            "rate": style_summary["middle_throw_pct"],
            "sample": int(
                style_summary["charted_pass_direction_plays"]
            ),
            "text": (
                "middle-field throws represented {rate:.1f}% of "
                "{sample:,} throws with recorded direction"
            ),
        },
        {
            "rate": style_summary["outside_run_pct"],
            "sample": int(
                style_summary["charted_run_direction_plays"]
            ),
            "text": (
                "outside runs represented {rate:.1f}% of {sample:,} "
                "runs with recorded direction"
            ),
        },
    ]

    supported = [
        candidate
        for candidate in candidates
        if candidate["sample"] >= 25
        and pd.notna(candidate["rate"])
    ]

    if not supported:
        return None

    strongest = max(
        supported,
        key=lambda candidate: candidate["rate"],
    )

    return strongest["text"].format(
        rate=float(strongest["rate"]),
        sample=int(strongest["sample"]),
    )


def build_copy_ready_findings(
    summary,
    style_summary,
    uncertainty_data,
    selection_details,
    app_link="",
):
    """Create concise and detailed text that can be copied directly."""

    subject = describe_insight_subject(
        selected_coaches,
        selected_teams,
    )
    period = describe_season_window(selected_seasons)
    season_types = describe_season_types(selected_season_types)

    plays = int(summary["plays"])
    games = int(summary["games"])
    actual_pass = 100 * float(summary["actual_pass_rate"])
    expected_pass = 100 * float(summary["expected_pass_rate"])
    pass_oe = 100 * float(summary["pass_oe"])

    if pass_oe >= 0.5:
        tendency = (
            f"passed {pass_oe:.2f} percentage points above the league "
            "model's situation-based expectation"
        )
    elif pass_oe <= -0.5:
        tendency = (
            f"passed {abs(pass_oe):.2f} percentage points below the "
            "league model's situation-based expectation"
        )
    else:
        tendency = (
            f"was close to modeled expectation at {pass_oe:+.2f} Pass "
            "OE points"
        )

    style_phrase = strongest_style_share_for_export(
        style_summary
    )

    quick_parts = [
        (
            f"NFL Coaching Decision Lab finding for {subject}, {period} "
            f"({season_types}): {plays:,} plays across {games:,} games. "
            f"The selected offense {tendency} ({actual_pass:.2f}% actual "
            f"versus {expected_pass:.2f}% expected)."
        )
    ]

    if style_phrase:
        quick_parts.append(
            f"The largest displayed observable style share was that "
            f"{style_phrase}."
        )

    quick_parts.append(
        "These findings are descriptive and do not establish that one "
        "play-calling tendency caused better or worse outcomes."
    )

    quick_text = " ".join(quick_parts)

    detailed_lines = [
        "NFL Coaching Decision Lab | Filtered Finding",
        "",
        f"Selection: {subject}; {period}; {season_types}",
        f"Sample: {plays:,} plays across {games:,} games",
        f"Actual pass rate: {actual_pass:.2f}%",
        f"Modeled pass expectation: {expected_pass:.2f}%",
        f"Pass rate over expected: {pass_oe:+.2f} percentage points",
        f"EPA per play: {float(summary['mean_epa']):.3f}",
        f"Success rate: {100 * float(summary['success_rate']):.2f}%",
        f"Yards per play: {float(summary['yards_per_play']):.2f}",
    ]

    if style_phrase:
        detailed_lines.append(
            f"Largest observable style share: {style_phrase}."
        )

    if not uncertainty_data.empty:
        strongest = uncertainty_data.loc[
            uncertainty_data["model_pass_oe_pct"].abs().idxmax()
        ]
        lower = float(strongest["ci_95_lower_pct"])
        upper = float(strongest["ci_95_upper_pct"])
        detailed_lines.append(
            "Strongest matching full-season interval: "
            f"{strongest['head_coach']} ({strongest['posteam']}, "
            f"{int(strongest['season'])}) at "
            f"{float(strongest['model_pass_oe_pct']):+.2f} Pass OE "
            f"points, 95% CI [{lower:+.2f}, {upper:+.2f}]."
        )

    detailed_lines.extend([
        f"Sample quality: {export_sample_quality(plays)}",
        "",
        "Interpretation note: Pass OE describes call selection relative "
        "to modeled pre-snap context. Outcomes and style rates are "
        "descriptive associations, not causal estimates.",
    ])

    normalized_link = app_link.strip()
    if normalized_link:
        quick_text += f" Explore the interactive app: {normalized_link}"
        detailed_lines.extend([
            "",
            f"Interactive app: {normalized_link}",
        ])

    return quick_text, "\n".join(detailed_lines)


def dataframe_to_markdown_table(dataframe):
    """Render a dataframe as basic Markdown without extra dependencies."""

    if dataframe.empty:
        return "_No matching rows._"

    columns = [str(column) for column in dataframe.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []

    for _, row in dataframe.iterrows():
        values = []
        for column in dataframe.columns:
            value = row[column]
            if pd.isna(value):
                value = ""
            value = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(value)
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def build_markdown_findings_report(
    report_title,
    selection_details,
    summary_metrics,
    insights,
    season_display,
    entity_display,
    records_display,
    uncertainty_display,
    app_link="",
):
    """Create a portable Markdown report for the current filters."""

    lines = [
        f"# {report_title}",
        "",
        "This report was generated from the active filters in the NFL "
        "Coaching Decision Lab.",
        "",
        "## Selection",
        "",
    ]

    for label, value in selection_details.items():
        lines.append(f"- **{label}:** {value}")

    lines.extend([
        "",
        "## Key Metrics",
        "",
        dataframe_to_markdown_table(summary_metrics),
        "",
        "## Automatic Observations",
        "",
    ])

    for insight in insights:
        lines.extend([
            f"### {insight['title']}",
            "",
            insight["body"],
            "",
        ])

    lines.extend([
        "## Season Breakdown",
        "",
        dataframe_to_markdown_table(season_display),
        "",
        "## Decision-Maker and Team Breakdown",
        "",
        "The table below shows the largest 25 filtered samples. The "
        "complete table is included as CSV in the download package.",
        "",
        dataframe_to_markdown_table(entity_display.head(25)),
        "",
        "## Team Results Context",
        "",
        "Records follow the selected seasons and season types, but do not "
        "change with situation filters such as down, distance, field "
        "position, quarter, or score.",
        "",
        dataframe_to_markdown_table(records_display),
        "",
        "## Full-Season Confidence Intervals",
        "",
        "These intervals use complete coach-team seasons and do not change "
        "with the situation filters.",
        "",
        dataframe_to_markdown_table(uncertainty_display),
        "",
        "## Interpretation",
        "",
        "Pass rate over expected is the average difference between the "
        "actual pass indicator and the league model's expected pass "
        "probability. It describes a tendency relative to modeled pre-snap "
        "context. EPA, success rate, yardage, team records, and style rates "
        "are descriptive associations rather than causal estimates.",
    ])

    if app_link.strip():
        lines.extend([
            "",
            f"Interactive app: {app_link.strip()}",
        ])

    return "\n".join(lines)


def build_html_findings_report(
    report_title,
    selection_details,
    summary_metrics,
    insights,
    season_display,
    entity_display,
    records_display,
    uncertainty_display,
    app_link="",
):
    """Create a polished, standalone HTML findings report."""

    escaped_title = html.escape(report_title)

    selection_html = "".join(
        (
            "<div class='selection-item'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            "</div>"
        )
        for label, value in selection_details.items()
    )

    metric_cards = "".join(
        (
            "<div class='metric-card'>"
            f"<span>{html.escape(str(row['Metric']))}</span>"
            f"<strong>{html.escape(str(row['Value']))}</strong>"
            "</div>"
        )
        for _, row in summary_metrics.iterrows()
    )

    insight_cards = "".join(
        (
            "<article class='insight-card'>"
            f"<h3>{html.escape(insight['title'])}</h3>"
            f"<p>{html.escape(insight['body'])}</p>"
            "</article>"
        )
        for insight in insights
    )

    def report_table(dataframe):
        if dataframe.empty:
            return "<p class='muted'>No matching rows.</p>"
        return dataframe.to_html(
            index=False,
            border=0,
            classes="data-table",
            justify="left",
            escape=True,
        )

    link_html = ""
    normalized_link = app_link.strip()
    if normalized_link:
        safe_link = html.escape(normalized_link, quote=True)
        link_html = (
            "<p class='app-link'><a href='"
            f"{safe_link}' target='_blank' rel='noopener noreferrer'>"
            "Open the interactive NFL Coaching Decision Lab</a></p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
:root {{
  color-scheme: light dark;
  --background: #0e1117;
  --panel: #151a23;
  --panel-soft: #1d2430;
  --text: #f5f7fa;
  --muted: #aeb8c6;
  --border: #343c49;
  --accent: #ff4b4b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--background);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
}}
main {{
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 42px 0 64px;
}}
h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.2rem); }}
h2 {{ margin-top: 42px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
h3 {{ margin: 0 0 8px; }}
p {{ margin: 0 0 14px; }}
.subtitle, .muted {{ color: var(--muted); }}
.selection-grid, .metric-grid, .insight-grid {{
  display: grid;
  gap: 14px;
}}
.selection-grid {{ grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }}
.metric-grid {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
.insight-grid {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
.selection-item, .metric-card, .insight-card {{
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--panel);
  padding: 16px;
}}
.selection-item span, .metric-card span {{
  display: block;
  color: var(--muted);
  font-size: 0.82rem;
  margin-bottom: 5px;
}}
.selection-item strong {{ font-size: 0.98rem; }}
.metric-card strong {{ font-size: 1.55rem; }}
.insight-card p {{ color: #dce2ea; }}
.table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
}}
.data-table {{
  border-collapse: collapse;
  width: 100%;
  min-width: 760px;
  background: var(--panel);
}}
.data-table th, .data-table td {{
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
.data-table th {{ background: var(--panel-soft); }}
.note {{
  border-left: 4px solid var(--accent);
  background: var(--panel);
  padding: 16px 18px;
  border-radius: 8px;
}}
.app-link a {{ color: #7ebcff; font-weight: 700; }}
@media (max-width: 640px) {{
  main {{ width: min(100% - 20px, 1180px); padding-top: 24px; }}
  .insight-grid {{ grid-template-columns: 1fr; }}
}}
@media print {{
  :root {{
    --background: #ffffff;
    --panel: #ffffff;
    --panel-soft: #f4f6f8;
    --text: #111827;
    --muted: #4b5563;
    --border: #d1d5db;
  }}
  main {{ width: 100%; padding: 0; }}
}}
</style>
</head>
<body>
<main>
  <header>
    <h1>{escaped_title}</h1>
    <p class="subtitle">A filter-specific export from the NFL Coaching Decision Lab.</p>
    {link_html}
  </header>

  <h2>Selection</h2>
  <section class="selection-grid">{selection_html}</section>

  <h2>Key Metrics</h2>
  <section class="metric-grid">{metric_cards}</section>

  <h2>Automatic Observations</h2>
  <section class="insight-grid">{insight_cards}</section>

  <h2>Season Breakdown</h2>
  <div class="table-wrap">{report_table(season_display)}</div>

  <h2>Decision-Maker and Team Breakdown</h2>
  <p class="muted">The largest 25 filtered samples are shown. The complete table is included in the CSV download package.</p>
  <div class="table-wrap">{report_table(entity_display.head(25))}</div>

  <h2>Team Results Context</h2>
  <p class="muted">Records follow season and season-type filters, but not situation filters.</p>
  <div class="table-wrap">{report_table(records_display)}</div>

  <h2>Full-Season Confidence Intervals</h2>
  <p class="muted">Intervals use complete coach-team seasons and do not change with situation filters.</p>
  <div class="table-wrap">{report_table(uncertainty_display)}</div>

  <h2>Interpretation</h2>
  <div class="note">
    Pass rate over expected describes call selection relative to modeled
    pre-snap context. EPA, success rate, yardage, records, and style rates
    are descriptive associations. They do not establish that one tendency
    caused better or worse outcomes.
  </div>
</main>
</body>
</html>"""


def slugify_export_filename(value):
    """Create a safe, compact filename component."""

    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        str(value).strip().lower(),
    ).strip("-")

    return slug or "all"


def build_findings_file_stem():
    """Create a recognizable file stem from the active selection."""

    parts = ["nfl-coaching-findings"]

    if len(selected_coaches) == 1:
        parts.append(selected_coaches[0])
    elif len(selected_teams) == 1:
        parts.append(selected_teams[0])

    parts.append(describe_season_window(selected_seasons))

    return "_".join(
        slugify_export_filename(part)
        for part in parts
    )


def build_findings_zip(
    file_stem,
    html_report,
    markdown_report,
    quick_text,
    detailed_text,
    summary_export,
    season_export,
    entity_export,
    records_export,
    uncertainty_export,
):
    """Bundle the report and all aggregate tables into one ZIP file."""

    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            f"{file_stem}_report.html",
            html_report,
        )
        archive.writestr(
            f"{file_stem}_report.md",
            markdown_report,
        )
        archive.writestr(
            f"{file_stem}_copy_ready.txt",
            quick_text + "\n\n" + detailed_text,
        )
        archive.writestr(
            f"{file_stem}_summary.csv",
            summary_export.to_csv(index=False),
        )
        archive.writestr(
            f"{file_stem}_by_season.csv",
            season_export.to_csv(index=False),
        )
        archive.writestr(
            f"{file_stem}_by_decision_maker.csv",
            entity_export.to_csv(index=False),
        )
        archive.writestr(
            f"{file_stem}_team_records.csv",
            records_export.to_csv(index=False),
        )
        archive.writestr(
            f"{file_stem}_confidence_intervals.csv",
            uncertainty_export.to_csv(index=False),
        )
        archive.writestr(
            "README.txt",
            (
                "This package was generated by the NFL Coaching Decision "
                "Lab using the active sidebar filters. The HTML file is a "
                "standalone shareable report. The Markdown file is easy to "
                "edit. CSV files contain the complete aggregate tables. "
                "Confidence intervals use complete coach-team seasons and "
                "do not change with situation filters. Findings are "
                "descriptive and should not be interpreted causally."
            ),
        )

    return buffer.getvalue()


where_clause, query_parameters = (
    build_filter_query()
)


(
    overview_tab,
    rankings_tab,
    profiles_tab,
    comparison_tab,
    trends_tab,
    simulator_tab,
    challenge_tab,
    style_tab,
    share_tab,
    attribution_tab,
    model_tab,
    methodology_tab,
) = st.tabs([
    "Overview",
    "Coach Rankings",
    "Decision-Maker Profiles",
    "Coach Comparison",
    "Historical Trends",
    "Situation Lab",
    "Make the Call",
    "Play Style",
    "Share Findings",
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

    route_columns = st.columns(4)
    with route_columns[0]:
        with st.container(border=True):
            st.markdown("#### Open a profile")
            st.caption(
                "Turn one coach or verified offensive play caller into a "
                "compact tendency, style, confidence, and results summary."
            )
            st.markdown("Use **Decision-Maker Profiles** above.")

    with route_columns[1]:
        with st.container(border=True):
            st.markdown("#### Compare decision-makers")
            st.caption(
                "Put coaches or verified offensive play callers side by "
                "side, with context-adjusted tendencies and outcomes."
            )
            st.markdown("Use the **Coach Comparison** tab above.")

    with route_columns[2]:
        with st.container(border=True):
            st.markdown("#### Build a game situation")
            st.caption(
                "Choose down, distance, field position, clock, and score "
                "to find comparable historical calls."
            )
            st.markdown("Open the **Situation Lab** tab above.")

    with route_columns[3]:
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

        st.markdown("### Automatic Insights")
        st.caption(
            "These observations update with the sidebar filters. They "
            "describe recorded tendencies and associations; they do not "
            "claim that a play-calling choice caused an outcome."
        )

        insight_subject = describe_insight_subject(
            selected_coaches,
            selected_teams,
        )
        insight_seasons = describe_season_window(
            selected_seasons
        )
        insight_coaches = describe_selected_values(
            selected_coaches,
            "all head coaches",
            "head coaches",
        )
        insight_teams = describe_selected_values(
            selected_teams,
            "all offensive teams",
            "teams",
        )
        insight_season_types = describe_season_types(
            selected_season_types
        )

        overview_style = load_overview_style_summary(
            where_clause,
            query_parameters,
        )
        overview_uncertainty = (
            load_matching_coach_uncertainty()
        )
        third_down_split = load_third_down_distance_split(
            where_clause,
            query_parameters,
        )

        insight_cards = []

        insight_cards.append({
            "title": "Selection Context",
            "body": (
                f"{insight_coaches}; {insight_teams}; {insight_seasons}; "
                f"{insight_season_types}. The current selection contains "
                f"{int(overview['plays']):,} plays across "
                f"{int(overview['games']):,} games."
            ),
        })

        pass_oe_points = 100 * overview["pass_oe"]

        if pass_oe_points >= 0.5:
            tendency_text = (
                f"{insight_subject} passed {pass_oe_points:.2f} "
                "percentage points above the league model's expectation "
                "in the selected situations."
            )
        elif pass_oe_points <= -0.5:
            tendency_text = (
                f"{insight_subject} passed {abs(pass_oe_points):.2f} "
                "percentage points below the league model's expectation, "
                "indicating a more run-leaning call mix in the selected "
                "situations."
            )
        else:
            tendency_text = (
                f"{insight_subject} stayed close to modeled expectation, "
                f"with Pass OE of {pass_oe_points:+.2f} percentage points "
                "in the selected situations."
            )

        insight_cards.append({
            "title": "Run-Pass Tendency",
            "body": (
                f"{tendency_text} This describes call selection relative "
                "to context, not whether passing or rushing more often "
                "caused better results."
            ),
        })

        insight_cards.append({
            "title": "Largest Style Share",
            "body": build_style_insight(
                overview_style,
                insight_subject,
            ),
        })

        insight_cards.append({
            "title": "Confidence Check",
            "body": build_confidence_insight(
                overview_uncertainty
            ),
        })

        play_count = int(overview["plays"])
        if play_count < 200:
            sample_message = (
                f"This filtered sample contains only {play_count:,} plays, "
                "so treat the result as exploratory and sensitive to a "
                "small number of games."
            )
        elif play_count < 500:
            sample_message = (
                f"This filtered sample contains {play_count:,} plays. It is "
                "large enough to inspect, but remains below the app's "
                "stronger 500-play descriptive threshold."
            )
        else:
            sample_message = (
                f"This filtered sample contains {play_count:,} plays and "
                "meets the app's stronger descriptive threshold. Sample "
                "size does not remove confounding or make the result causal."
            )

        insight_cards.append({
            "title": "Sample Quality",
            "body": sample_message,
        })

        third_down_insight = build_third_down_insight(
            third_down_split
        )
        if third_down_insight is not None:
            insight_cards.append({
                "title": "Third-Down Distance Split",
                "body": third_down_insight,
            })

        insight_columns = st.columns(2)
        for index, insight in enumerate(insight_cards):
            with insight_columns[index % 2]:
                render_insight_card(
                    insight["title"],
                    insight["body"],
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

        render_plotly_chart(
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

        st.markdown(
            "#### Pass Rate Over Expected by Coach-Team"
        )

        ranked_coaches = rankings.sort_values(
            "pass_oe_pct"
        ).copy()

        largest_absolute_pass_oe = max(
            1.0,
            float(
                ranked_coaches[
                    "pass_oe_pct"
                ].abs().max()
            ),
        )
        axis_padding = max(
            1.0,
            0.18 * largest_absolute_pass_oe,
        )
        symmetric_limit = (
            largest_absolute_pass_oe
            + axis_padding
        )

        if len(ranked_coaches) == 1:
            single_row = ranked_coaches.iloc[0]
            pass_oe_value = float(
                single_row["pass_oe_pct"]
            )
            point_color = (
                "#D73027"
                if pass_oe_value < 0
                else "#1F78B4"
            )

            ranking_figure = go.Figure()

            ranking_figure.add_shape(
                type="line",
                x0=0,
                x1=pass_oe_value,
                y0=0,
                y1=0,
                line={
                    "color": point_color,
                    "width": 8,
                },
            )

            ranking_figure.add_trace(
                go.Scatter(
                    x=[pass_oe_value],
                    y=[0],
                    mode="markers+text",
                    marker={
                        "size": 18,
                        "color": point_color,
                        "line": {
                            "color": "#F8FAFC",
                            "width": 1.5,
                        },
                    },
                    text=[
                        f"{pass_oe_value:+.2f} pts"
                    ],
                    textposition=(
                        "middle left"
                        if pass_oe_value < 0
                        else "middle right"
                    ),
                    textfont={
                        "size": 15,
                        "color": "#F8FAFC",
                    },
                    customdata=[[
                        int(single_row["plays"]),
                        int(single_row["games"]),
                        float(
                            single_row[
                                "actual_pass_rate_pct"
                            ]
                        ),
                        float(
                            single_row[
                                "expected_pass_rate_pct"
                            ]
                        ),
                        float(single_row["mean_epa"]),
                    ]],
                    hovertemplate=(
                        f"<b>{single_row['label']}</b><br>"
                        "Pass OE: %{x:+.2f} pts<br>"
                        "Plays: %{customdata[0]:,}<br>"
                        "Games: %{customdata[1]:,}<br>"
                        "Actual pass rate: "
                        "%{customdata[2]:.2f}%<br>"
                        "Expected pass rate: "
                        "%{customdata[3]:.2f}%<br>"
                        "EPA/play: %{customdata[4]:.4f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

            ranking_figure.add_vline(
                x=0,
                line_dash="dash",
                line_color="rgba(255,255,255,0.48)",
                line_width=2,
            )

            ranking_figure.add_annotation(
                x=0,
                y=0.72,
                text="League-model expectation",
                showarrow=False,
                font={
                    "size": 12,
                    "color": "#AEB8C6",
                },
            )
            ranking_figure.add_annotation(
                x=-symmetric_limit,
                y=-0.72,
                text="More run-heavy",
                showarrow=False,
                xanchor="left",
                font={
                    "size": 12,
                    "color": "#E58A84",
                },
            )
            ranking_figure.add_annotation(
                x=symmetric_limit,
                y=-0.72,
                text="More pass-heavy",
                showarrow=False,
                xanchor="right",
                font={
                    "size": 12,
                    "color": "#82B7DE",
                },
            )

            ranking_figure.update_xaxes(
                range=[
                    -symmetric_limit,
                    symmetric_limit,
                ],
                autorange=False,
                title=(
                    "Pass OE "
                    "(percentage points)"
                ),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False,
            )
            ranking_figure.update_yaxes(
                range=[-1, 1],
                visible=False,
                fixedrange=True,
            )
            ranking_figure.update_layout(
                height=285,
                margin={
                    "l": 48,
                    "r": 48,
                    "t": 30,
                    "b": 55,
                },
                hovermode="closest",
            )
            ranking_figure.layout.title = None

            st.caption(
                f"{single_row['label']} is the only "
                "coach-team sample meeting the current "
                f"{minimum_plays:,}-play minimum."
            )
        else:
            ranking_figure = px.bar(
                ranked_coaches,
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
                labels={
                    "pass_oe_pct": (
                        "Pass OE "
                        "(percentage points)"
                    ),
                    "label": "",
                },
                hover_data={
                    "plays": ":,",
                    "games": ":,",
                    "actual_pass_rate_pct": ":.2f",
                    "expected_pass_rate_pct": ":.2f",
                    "mean_epa": ":.4f",
                },
            )
            ranking_figure.update_traces(
                texttemplate="%{x:+.2f}",
                textposition="outside",
                cliponaxis=False,
            )
            ranking_figure.update_xaxes(
                range=[
                    -symmetric_limit,
                    symmetric_limit,
                ],
                autorange=False,
                zeroline=False,
            )
            ranking_figure.update_layout(
                yaxis={
                    "categoryorder": "total ascending"
                },
                coloraxis_showscale=False,
                height=max(
                    420,
                    34 * len(ranked_coaches),
                ),
            )
            ranking_figure.add_vline(
                x=0,
                line_dash="dash",
                line_color="gray",
            )
            ranking_figure.layout.title = None

        render_plotly_chart(
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


# Decision-maker profiles tab
with profiles_tab:
    st.subheader("Decision-Maker Profiles")

    st.caption(
        "Build a readable profile for one head coach or verified offensive "
        "play caller. Point estimates and observable style shares follow all "
        "sidebar filters. Team records and complete-season confidence "
        "intervals follow only the compatible period and team filters."
    )

    profile_attribution = st.radio(
        "Build a profile for",
        options=[
            "Head coach",
            "Verified offensive play caller",
        ],
        horizontal=True,
        key="profile_attribution",
    )

    profile_is_caller = (
        profile_attribution
        == "Verified offensive play caller"
    )

    profile_options = (
        filter_options["play_callers"]
        if profile_is_caller
        else filter_options["coaches"]
    )

    profile_entity_column = (
        "offensive_play_caller"
        if profile_is_caller
        else "head_coach"
    )

    profile_source = (
        "play_predictions_with_callers"
        if profile_is_caller
        else "play_predictions"
    )

    profile_uncertainty_source = (
        "play_caller_uncertainty"
        if profile_is_caller
        else "coach_uncertainty"
    )

    profile_entity_label = (
        "Offensive Play Caller"
        if profile_is_caller
        else "Head Coach"
    )

    preferred_profile_entity = None

    if (
        not profile_is_caller
        and len(selected_coaches) == 1
        and selected_coaches[0] in profile_options
    ):
        preferred_profile_entity = selected_coaches[0]
    elif "Andy Reid" in profile_options:
        preferred_profile_entity = "Andy Reid"
    elif profile_options:
        preferred_profile_entity = profile_options[0]

    if not profile_options:
        st.warning(
            "No decision-makers are available for this attribution view."
        )
    else:
        profile_entity = st.selectbox(
            (
                "Select an offensive play caller"
                if profile_is_caller
                else "Select a head coach"
            ),
            options=profile_options,
            index=profile_options.index(
                preferred_profile_entity
            ),
            key=(
                "profile_play_caller"
                if profile_is_caller
                else "profile_head_coach"
            ),
        )

        profile_where, profile_parameters = build_filter_query(
            coach_override=[profile_entity],
            coach_column=profile_entity_column,
            use_sidebar_coaches=False,
        )

        profile_summary = connection.execute(
            f"""
            SELECT
                {profile_entity_column} AS entity,
                posteam,
                MIN(season) AS first_season,
                MAX(season) AS last_season,
                COUNT(DISTINCT season) AS seasons_represented,
                COUNT(DISTINCT game_id) AS games,
                COUNT(*) AS plays,
                ROUND(100.0 * AVG(is_pass), 2)
                    AS actual_pass_rate_pct,
                ROUND(
                    100.0 * AVG(expected_pass_probability),
                    2
                ) AS expected_pass_rate_pct,
                ROUND(100.0 * AVG(model_pass_oe), 2)
                    AS pass_oe_pct,
                ROUND(AVG(epa), 4) AS mean_epa,
                ROUND(100.0 * AVG(success), 2)
                    AS success_rate_pct,
                ROUND(AVG(yards_gained), 2)
                    AS yards_per_play
            FROM {profile_source}
            {profile_where}
            GROUP BY
                {profile_entity_column},
                posteam
            ORDER BY
                plays DESC,
                posteam
            """,
            profile_parameters,
        ).fetchdf()

        profile_style = connection.execute(
            f"""
            SELECT
                {profile_entity_column} AS entity,
                posteam,
                COUNT(*) AS style_plays,
                SUM(CASE
                    WHEN pass_depth_bucket IS NOT NULL
                    THEN 1 ELSE 0
                END) AS charted_throw_depth_plays,
                SUM(CASE
                    WHEN pass_direction IS NOT NULL
                    THEN 1 ELSE 0
                END) AS charted_pass_direction_plays,
                SUM(CASE
                    WHEN run_direction IS NOT NULL
                    THEN 1 ELSE 0
                END) AS charted_run_direction_plays,
                ROUND(100.0 * AVG(shotgun), 2)
                    AS shotgun_pct,
                ROUND(100.0 * AVG(no_huddle), 2)
                    AS no_huddle_pct,
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
                ) AS deep_throw_pct,
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
                ) AS middle_throw_pct,
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
                ) AS outside_run_pct
            FROM play_style_predictions_with_callers
            {profile_where}
            GROUP BY
                {profile_entity_column},
                posteam
            ORDER BY
                {profile_entity_column},
                posteam
            """,
            profile_parameters,
        ).fetchdf()

        profile_season_data = connection.execute(
            f"""
            SELECT
                season,
                {profile_entity_column} AS entity,
                posteam,
                COUNT(*) AS plays,
                ROUND(100.0 * AVG(model_pass_oe), 2)
                    AS pass_oe_pct,
                ROUND(AVG(epa), 4) AS mean_epa,
                ROUND(100.0 * AVG(success), 2)
                    AS success_rate_pct
            FROM {profile_source}
            {profile_where}
            GROUP BY
                season,
                {profile_entity_column},
                posteam
            ORDER BY
                season,
                posteam
            """,
            profile_parameters,
        ).fetchdf()

        profile_uncertainty = load_profile_uncertainty(
            profile_entity,
            profile_entity_column,
            profile_uncertainty_source,
        )

        if profile_summary.empty:
            if selected_teams:
                team_text = ", ".join(selected_teams)
                decision_maker_type = (
                    "offensive play caller"
                    if profile_is_caller
                    else "head coach"
                )
                st.warning(
                    f"{profile_entity} has no matching plays for "
                    f"{team_text} under the active filters. The selected "
                    f"{decision_maker_type} and team may not align for the "
                    "chosen seasons, or the remaining situation filters may "
                    "remove every play."
                )
                st.caption(
                    "The visual theme follows the offensive-team sidebar "
                    "selection. It does not imply that the selected "
                    "decision-maker coached or called plays for that team."
                )
            else:
                st.warning(
                    "No plays match this decision-maker and the active "
                    "sidebar filters. Try expanding the seasons or "
                    "situation ranges."
                )
        else:
            profile_summary = profile_summary.merge(
                profile_style,
                on=["entity", "posteam"],
                how="left",
            )

            if profile_is_caller:
                profile_records = load_team_records()
                profile_summary = profile_summary.merge(
                    profile_records,
                    on="posteam",
                    how="left",
                )
            else:
                profile_records = load_coach_team_records(
                    coach_override=[profile_entity]
                ).rename(columns={"head_coach": "entity"})
                profile_summary = profile_summary.merge(
                    profile_records,
                    on=["entity", "posteam"],
                    how="left",
                    suffixes=("", "_record"),
                )

                profile_experience = load_latest_tenure_context(
                    coach_override=[profile_entity]
                ).rename(columns={"head_coach": "entity"})

                profile_summary = profile_summary.merge(
                    profile_experience,
                    on=["entity", "posteam"],
                    how="left",
                )

            st.markdown(
                f"### {profile_entity_label}: {profile_entity}"
            )

            if len(profile_summary) > 1:
                st.caption(
                    "More than one team matches the current period and team "
                    "filters, so the app displays a separate profile card "
                    "for each team."
                )

            for _, profile_row in profile_summary.iterrows():
                team = profile_row["posteam"]
                period = format_profile_period(
                    profile_row["first_season"],
                    profile_row["last_season"],
                )
                team_uncertainty = profile_uncertainty[
                    profile_uncertainty["posteam"] == team
                ]
                subject = f"{profile_entity} with {team}"

                with st.container(border=True):
                    header_columns = st.columns([3, 1])

                    with header_columns[0]:
                        st.markdown(
                            f"### {profile_entity} · {team}"
                        )
                        st.caption(
                            f"{period} | "
                            f"{int(profile_row['seasons_represented'])} "
                            f"season(s) | {int(profile_row['games']):,} "
                            f"games | {int(profile_row['plays']):,} plays"
                        )

                    with header_columns[1]:
                        st.markdown(
                            f"**{profile_sample_label(profile_row['plays'])}**"
                        )
                        st.caption(
                            "Based on the current filtered play sample."
                        )

                    profile_metrics = st.columns(4)
                    profile_metrics[0].metric(
                        "Pass OE",
                        f"{profile_row['pass_oe_pct']:+.2f} pts",
                    )
                    profile_metrics[1].metric(
                        "Actual / expected pass",
                        (
                            f"{profile_row['actual_pass_rate_pct']:.1f}% / "
                            f"{profile_row['expected_pass_rate_pct']:.1f}%"
                        ),
                    )
                    profile_metrics[2].metric(
                        "EPA per play",
                        f"{profile_row['mean_epa']:.3f}",
                    )
                    profile_metrics[3].metric(
                        "Success rate",
                        f"{profile_row['success_rate_pct']:.1f}%",
                    )

                    detail_columns = st.columns(2)

                    with detail_columns[0]:
                        st.markdown("**Tendency summary**")
                        st.write(
                            build_profile_tendency_statement(
                                profile_row
                            )
                        )

                        st.markdown("**Results context**")
                        st.write(
                            build_profile_outcome_statement(
                                profile_row
                            )
                        )

                    with detail_columns[1]:
                        st.markdown("**Observable style**")
                        st.write(
                            build_profile_style_statement(
                                profile_row,
                                subject,
                            )
                        )

                        st.markdown("**Confidence evidence**")
                        st.write(
                            build_profile_confidence_statement(
                                team_uncertainty
                            )
                        )

                    context_parts = []

                    if pd.notna(profile_row.get("wins")):
                        record = format_record(profile_row)
                        context_parts.append(
                            f"Team record: {record}"
                        )

                    if pd.notna(
                        profile_row.get(
                            "point_differential_per_game"
                        )
                    ):
                        context_parts.append(
                            "point differential per game: "
                            f"{profile_row['point_differential_per_game']:+.2f}"
                        )

                    if (
                        not profile_is_caller
                        and pd.notna(
                            profile_row.get("experience_season")
                        )
                    ):
                        prior_record = format_prior_record(
                            profile_row
                        )
                        context_parts.append(
                            "observed NFL head-coaching experience entering "
                            f"{int(profile_row['experience_season'])}: "
                            f"{int(profile_row['observed_prior_hc_seasons'])} "
                            f"prior seasons, "
                            f"{int(profile_row['observed_prior_hc_games']):,} "
                            f"games, {prior_record} record"
                        )

                    if context_parts:
                        st.caption(
                            " | ".join(context_parts)
                        )

                    if profile_is_caller:
                        st.caption(
                            "Team results provide context and are not "
                            "attributed solely to the offensive play caller."
                        )

            if not profile_season_data.empty:
                st.markdown("### Season-by-Season Tendency")
                st.caption(
                    "These point estimates follow every active sidebar "
                    "filter. Smaller season samples may move substantially "
                    "with only a few additional games."
                )

                profile_season_data["entity_team"] = (
                    profile_season_data["entity"]
                    + " ("
                    + profile_season_data["posteam"]
                    + ")"
                )

                profile_trend_figure = px.line(
                    profile_season_data,
                    x="season",
                    y="pass_oe_pct",
                    color="entity_team",
                    markers=True,
                    hover_data={
                        "plays": ":,",
                        "mean_epa": ":.3f",
                        "success_rate_pct": ":.1f",
                        "entity_team": False,
                    },
                    title="Filtered Pass Rate Over Expected by Season",
                    labels={
                        "season": "Season",
                        "pass_oe_pct": (
                            "Pass OE (percentage points)"
                        ),
                        "entity_team": "Profile",
                        "plays": "Plays",
                        "mean_epa": "EPA per Play",
                        "success_rate_pct": "Success Rate (%)",
                    },
                )

                profile_trend_figure.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                )

                render_plotly_chart(
                    profile_trend_figure,
                    width="stretch",
                )

            style_chart_columns = [
                "shotgun_pct",
                "no_huddle_pct",
                "deep_throw_pct",
                "middle_throw_pct",
                "outside_run_pct",
            ]

            available_style_columns = [
                column
                for column in style_chart_columns
                if column in profile_summary.columns
                and profile_summary[column].notna().any()
            ]

            if available_style_columns:
                st.markdown("### Observable Style Fingerprint")
                st.caption(
                    "Throw and run direction rates use only plays with the "
                    "required charting. The bars describe recorded behavior "
                    "rather than named offensive schemes."
                )

                style_chart_data = profile_summary[
                    ["entity", "posteam"]
                    + available_style_columns
                ].copy()

                style_chart_data["entity_team"] = (
                    style_chart_data["entity"]
                    + " ("
                    + style_chart_data["posteam"]
                    + ")"
                )

                style_chart_data = style_chart_data.melt(
                    id_vars=["entity_team"],
                    value_vars=available_style_columns,
                    var_name="style_metric",
                    value_name="percentage",
                )

                style_chart_data["style_metric"] = (
                    style_chart_data["style_metric"].replace({
                        "shotgun_pct": "Shotgun",
                        "no_huddle_pct": "No huddle",
                        "deep_throw_pct": (
                            "Deep throws (20+ air yards)"
                        ),
                        "middle_throw_pct": (
                            "Middle-field throws"
                        ),
                        "outside_run_pct": "Outside runs",
                    })
                )

                profile_style_figure = px.bar(
                    style_chart_data,
                    x="style_metric",
                    y="percentage",
                    color="entity_team",
                    barmode="group",
                    text_auto=".1f",
                    title="Profile Style Shares",
                    labels={
                        "style_metric": "Style dimension",
                        "percentage": "Rate (%)",
                        "entity_team": "Profile",
                    },
                )

                render_plotly_chart(
                    profile_style_figure,
                    width="stretch",
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

            render_plotly_chart(
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

            render_plotly_chart(
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

                render_plotly_chart(
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

        render_plotly_chart(
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

        render_plotly_chart(
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

        render_plotly_chart(
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

        render_plotly_chart(
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
        or st.session_state.get("make_call_deck_version") != 3
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
        st.session_state.make_call_history = []
        st.session_state.make_call_session_difficulty = (
            challenge_difficulty
        )
        st.session_state.make_call_session_minimum = int(
            challenge_minimum_plays
        )
        st.session_state.make_call_deck_version = 3

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
        matched_history = (
            st.session_state.make_call_choice
            == historical_majority
        )

        st.session_state.make_call_attempts += 1
        if matched_history:
            st.session_state.make_call_score += 1

        result_number = (
            int(st.session_state.make_call_index) + 1
        )
        recorded_numbers = {
            int(result["challenge_number"])
            for result in st.session_state.make_call_history
        }

        if result_number not in recorded_numbers:
            historical_pass_rate = float(
                challenge_league["actual_pass_rate_pct"]
            )
            st.session_state.make_call_history.append({
                "challenge_number": result_number,
                "situation": challenge_name,
                "down": int(challenge["down"]),
                "distance": int(challenge["ydstogo"]),
                "field_position": challenge_field_text,
                "quarter": int(challenge["qtr"]),
                "clock": challenge_clock,
                "score_state": challenge_score_text,
                "choice": st.session_state.make_call_choice,
                "historical_majority": historical_majority,
                "correct": bool(matched_history),
                "historical_pass_rate_pct": (
                    historical_pass_rate
                ),
                "historical_run_rate_pct": (
                    100 - historical_pass_rate
                ),
                "majority_margin_pct": abs(
                    historical_pass_rate - 50
                ),
                "comparable_plays": int(
                    challenge_league["plays"]
                ),
                "match_tier": challenge_league[
                    "match_tier"
                ],
                "pass_oe_pct": float(
                    challenge_league["pass_oe_pct"]
                ),
            })

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

        majority_margin = abs(
            float(challenge_league["actual_pass_rate_pct"]) - 50
        )
        if majority_margin >= 20:
            difficulty_description = (
                "a clear historical tendency"
            )
        elif majority_margin >= 10:
            difficulty_description = (
                "a noticeable historical lean"
            )
        elif majority_margin >= 5:
            difficulty_description = (
                "a modest historical lean"
            )
        else:
            difficulty_description = (
                "a near-even historical split"
            )

        st.caption(
            f"This challenge had {difficulty_description}: the "
            f"historical pass rate was "
            f"{float(challenge_league['actual_pass_rate_pct']):.1f}%, "
            f"or {majority_margin:.1f} points from an even split."
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

        def render_challenge_reveal_details():
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
            render_plotly_chart(call_split_figure, width="stretch")

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
            render_plotly_chart(
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
            with st.expander(
                "Review the final challenge details",
                expanded=False,
            ):
                render_challenge_reveal_details()
        else:
            render_challenge_reveal_details()

        if is_last_challenge:
            history = pd.DataFrame(
                st.session_state.make_call_history
            ).sort_values("challenge_number")

            st.divider()
            st.markdown("## Session Results")

            session_accuracy = (
                100 * history["correct"].mean()
                if not history.empty
                else 0.0
            )

            if session_accuracy >= 85:
                st.success(
                    "Excellent session. You consistently recognized "
                    "the league's historical situational tendencies."
                )
            elif session_accuracy >= 70:
                st.success(
                    "Strong session. Your reads matched the historical "
                    "majority on most situations."
                )
            elif session_accuracy >= 55:
                st.info(
                    "Competitive session. You identified several useful "
                    "patterns, with room to sharpen the closest calls."
                )
            else:
                st.warning(
                    "Developing session. Review the down-and-distance "
                    "patterns below to see where the historical calls "
                    "differed from your instincts."
                )

            pass_reads = history[
                history["historical_majority"] == "Pass"
            ]
            run_reads = history[
                history["historical_majority"] == "Run"
            ]

            def result_fraction(frame):
                if frame.empty:
                    return "No calls"
                return (
                    f"{int(frame['correct'].sum())}/"
                    f"{len(frame)}"
                )

            result_metrics = st.columns(4)
            result_metrics[0].metric(
                "Final score",
                f"{int(history['correct'].sum())}/{len(history)}",
            )
            result_metrics[1].metric(
                "Accuracy",
                f"{session_accuracy:.0f}%",
            )
            result_metrics[2].metric(
                "Pass-majority reads",
                result_fraction(pass_reads),
            )
            result_metrics[3].metric(
                "Run-majority reads",
                result_fraction(run_reads),
            )

            user_pass_choice_rate = (
                100 * (history["choice"] == "Pass").mean()
            )
            majority_pass_rate = (
                100
                * (
                    history["historical_majority"] == "Pass"
                ).mean()
            )
            average_margin = history[
                "majority_margin_pct"
            ].mean()

            down_summary = (
                history.groupby("down", as_index=False)
                .agg(
                    attempts=("correct", "size"),
                    correct_reads=("correct", "sum"),
                    accuracy_pct=("correct", "mean"),
                )
            )
            down_summary["accuracy_pct"] *= 100
            down_summary["Down"] = down_summary["down"].map({
                1: "First down",
                2: "Second down",
                3: "Third down",
                4: "Fourth down",
            })

            repeated_downs = down_summary[
                down_summary["attempts"] >= 2
            ]
            if repeated_downs.empty:
                best_down_text = (
                    "No down appeared at least twice, so there is not "
                    "enough repetition for a meaningful down-specific "
                    "strength."
                )
            else:
                best_down = repeated_downs.sort_values(
                    ["accuracy_pct", "attempts"],
                    ascending=[False, False],
                ).iloc[0]
                best_down_text = (
                    f"Your strongest repeated situation was "
                    f"{best_down['Down'].lower()}: "
                    f"{best_down['accuracy_pct']:.0f}% accuracy "
                    f"across {int(best_down['attempts'])} calls."
                )

            closest_call = history.sort_values(
                "majority_margin_pct"
            ).iloc[0]
            closest_result = (
                "matched"
                if bool(closest_call["correct"])
                else "did not match"
            )

            insight_columns = st.columns(3)
            with insight_columns[0]:
                with st.container(border=True):
                    st.markdown("#### Your Call Profile")
                    st.write(
                        f"You selected pass on "
                        f"{user_pass_choice_rate:.0f}% of challenges. "
                        f"The historical majority was pass on "
                        f"{majority_pass_rate:.0f}% of them."
                    )

            with insight_columns[1]:
                with st.container(border=True):
                    st.markdown("#### Best Repeated Situation")
                    st.write(best_down_text)

            with insight_columns[2]:
                with st.container(border=True):
                    st.markdown("#### Closest Historical Split")
                    st.write(
                        f"Challenge "
                        f"{int(closest_call['challenge_number'])} "
                        f"was only "
                        f"{closest_call['majority_margin_pct']:.1f} "
                        f"points from 50-50. Your read "
                        f"{closest_result} the historical majority."
                    )

            st.caption(
                f"Across the deck, the average historical majority "
                f"margin was {average_margin:.1f} percentage points. "
                "Larger margins indicate clearer historical call "
                "tendencies, not better football decisions."
            )

            history["Result"] = history["correct"].map({
                True: "Correct",
                False: "Missed",
            })
            history["Majority Call"] = history[
                "historical_majority"
            ]

            result_chart_column, down_chart_column = st.columns(2)

            with result_chart_column:
                review_figure = px.scatter(
                    history,
                    x="challenge_number",
                    y="historical_pass_rate_pct",
                    color="Result",
                    symbol="Majority Call",
                    color_discrete_map={
                        "Correct": "#2E8B57",
                        "Missed": "#C0392B",
                    },
                    title="Challenge-by-Challenge Reads",
                    labels={
                        "challenge_number": "Challenge",
                        "historical_pass_rate_pct": (
                            "Historical Pass Rate (%)"
                        ),
                    },
                    hover_data={
                        "situation": True,
                        "choice": True,
                        "comparable_plays": ":,",
                        "match_tier": True,
                        "challenge_number": False,
                    },
                )
                review_figure.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Even split",
                )
                review_figure.update_traces(
                    marker={"size": 13}
                )
                review_figure.update_yaxes(range=[0, 100])
                render_plotly_chart(
                    review_figure,
                    width="stretch",
                )

            with down_chart_column:
                down_figure = px.bar(
                    down_summary,
                    x="Down",
                    y="accuracy_pct",
                    text="accuracy_pct",
                    title="Accuracy by Down",
                    labels={
                        "accuracy_pct": "Accuracy (%)",
                    },
                    category_orders={
                        "Down": [
                            "First down",
                            "Second down",
                            "Third down",
                            "Fourth down",
                        ]
                    },
                    hover_data={
                        "attempts": True,
                        "correct_reads": True,
                        "down": False,
                    },
                )
                down_figure.update_traces(
                    texttemplate="%{text:.0f}%",
                    textposition="outside",
                )
                down_figure.update_yaxes(range=[0, 105])
                render_plotly_chart(
                    down_figure,
                    width="stretch",
                )

            with st.expander(
                "Review every call",
                expanded=True,
            ):
                review_table = history[[
                    "challenge_number",
                    "situation",
                    "choice",
                    "historical_majority",
                    "historical_pass_rate_pct",
                    "comparable_plays",
                    "match_tier",
                    "Result",
                ]].rename(columns={
                    "challenge_number": "Challenge",
                    "situation": "Situation",
                    "choice": "Your Call",
                    "historical_majority": (
                        "Historical Majority"
                    ),
                    "historical_pass_rate_pct": (
                        "Historical Pass %"
                    ),
                    "comparable_plays": "Comparable Plays",
                    "match_tier": "Match Tier",
                })
                st.dataframe(
                    review_table,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Historical Pass %": (
                            st.column_config.NumberColumn(
                                format="%.1f%%"
                            )
                        ),
                        "Comparable Plays": (
                            st.column_config.NumberColumn(
                                format="%d"
                            )
                        ),
                    },
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
                st.session_state.make_call_history = []
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
            st.session_state.make_call_history = []
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

        if fingerprint.empty:
            selected_entity_text = ", ".join(style_entities)
            if selected_teams:
                selected_team_text = ", ".join(selected_teams)
                st.warning(
                    f"No play-style data matches {selected_entity_text} "
                    f"with {selected_team_text} under the active filters. "
                    "The selected decision-maker and team may not align for "
                    "the chosen seasons."
                )
                st.caption(
                    "Choose a team associated with the selected "
                    "decision-maker, clear the offensive-team filter, or "
                    "expand the selected seasons. The team-colored visual "
                    "theme is controlled separately by the team filter."
                )
            else:
                st.warning(
                    "No play-style data matches the selected "
                    "decision-maker and active filters. Try expanding the "
                    "seasons or situation ranges."
                )
        else:
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

            render_plotly_chart(
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
                    render_plotly_chart(
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
                    render_plotly_chart(
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
                    render_plotly_chart(
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
                    render_plotly_chart(
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
                    render_plotly_chart(
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
                    render_plotly_chart(
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


# Download and share tab
with share_tab:
    st.subheader("Download & Share Filtered Findings")

    st.caption(
        "Turn the active sidebar filters into a copy-ready finding, a "
        "standalone HTML report, or a package of analysis-ready CSV files. "
        "Every export updates when the filters change."
    )

    share_summary = connection.execute(
        f"""
        SELECT
            COUNT(*) AS plays,
            COUNT(DISTINCT game_id) AS games,
            AVG(is_pass) AS actual_pass_rate,
            AVG(expected_pass_probability) AS expected_pass_rate,
            AVG(model_pass_oe) AS pass_oe,
            AVG(epa) AS mean_epa,
            AVG(success) AS success_rate,
            AVG(yards_gained) AS yards_per_play
        FROM play_predictions
        {where_clause}
        """,
        query_parameters,
    ).fetchdf().iloc[0]

    if int(share_summary["plays"]) == 0:
        st.warning(
            "No plays match the current filters, so there is nothing to "
            "export. Broaden the sidebar selection and try again."
        )
    else:
        share_style = load_overview_style_summary(
            where_clause,
            query_parameters,
        )
        share_uncertainty = load_matching_coach_uncertainty()
        share_third_down = load_third_down_distance_split(
            where_clause,
            query_parameters,
        )
        share_insights = build_share_insights(
            share_summary,
            share_style,
            share_uncertainty,
            share_third_down,
        )
        share_selection = build_share_selection_details()

        share_season_data = connection.execute(
            f"""
            SELECT
                season,
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
                ROUND(AVG(yards_gained), 2)
                    AS yards_per_play
            FROM play_predictions
            {where_clause}
            GROUP BY season
            ORDER BY season
            """,
            query_parameters,
        ).fetchdf()

        share_entity_data = connection.execute(
            f"""
            SELECT
                head_coach,
                posteam,
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
                ROUND(AVG(yards_gained), 2)
                    AS yards_per_play
            FROM play_predictions
            {where_clause}
            GROUP BY
                head_coach,
                posteam
            ORDER BY
                plays DESC,
                head_coach,
                posteam
            """,
            query_parameters,
        ).fetchdf()

        share_entity_data["sample_quality"] = (
            share_entity_data["plays"].apply(
                export_sample_quality
            )
        )

        share_records = load_coach_team_records()
        if not share_records.empty:
            share_records = share_records.copy()
            share_records["record"] = share_records.apply(
                format_record,
                axis=1,
            )

        summary_export = pd.DataFrame([{
            "seasons": share_selection["Seasons"],
            "season_type": share_selection["Season type"],
            "head_coaches": share_selection["Head coaches"],
            "offensive_teams": share_selection["Offensive teams"],
            "down": share_selection["Down"],
            "quarter": share_selection["Quarter"],
            "yards_to_go": share_selection["Yards to go"],
            "field_position": share_selection["Field position"],
            "score_differential": share_selection[
                "Score differential"
            ],
            "plays": int(share_summary["plays"]),
            "games": int(share_summary["games"]),
            "actual_pass_rate_pct": round(
                100 * float(share_summary["actual_pass_rate"]),
                2,
            ),
            "expected_pass_rate_pct": round(
                100 * float(share_summary["expected_pass_rate"]),
                2,
            ),
            "pass_oe_pct": round(
                100 * float(share_summary["pass_oe"]),
                2,
            ),
            "mean_epa": round(
                float(share_summary["mean_epa"]),
                4,
            ),
            "success_rate_pct": round(
                100 * float(share_summary["success_rate"]),
                2,
            ),
            "yards_per_play": round(
                float(share_summary["yards_per_play"]),
                2,
            ),
            "shotgun_pct": round(
                float(share_style["shotgun_pct"]),
                2,
            ),
            "no_huddle_pct": round(
                float(share_style["no_huddle_pct"]),
                2,
            ),
            "deep_throw_pct": (
                round(float(share_style["deep_throw_pct"]), 2)
                if pd.notna(share_style["deep_throw_pct"])
                else None
            ),
            "middle_throw_pct": (
                round(float(share_style["middle_throw_pct"]), 2)
                if pd.notna(share_style["middle_throw_pct"])
                else None
            ),
            "outside_run_pct": (
                round(float(share_style["outside_run_pct"]), 2)
                if pd.notna(share_style["outside_run_pct"])
                else None
            ),
            "sample_quality": export_sample_quality(
                share_summary["plays"]
            ),
        }])

        summary_metrics = pd.DataFrame({
            "Metric": [
                "Plays",
                "Games",
                "Actual Pass Rate",
                "Expected Pass Rate",
                "Pass OE",
                "EPA per Play",
                "Success Rate",
                "Yards per Play",
            ],
            "Value": [
                f"{int(share_summary['plays']):,}",
                f"{int(share_summary['games']):,}",
                f"{100 * share_summary['actual_pass_rate']:.2f}%",
                f"{100 * share_summary['expected_pass_rate']:.2f}%",
                f"{100 * share_summary['pass_oe']:+.2f} pts",
                f"{share_summary['mean_epa']:.3f}",
                f"{100 * share_summary['success_rate']:.2f}%",
                f"{share_summary['yards_per_play']:.2f}",
            ],
        })

        season_display = share_season_data.rename(
            columns={
                "season": "Season",
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

        entity_display = share_entity_data.rename(
            columns={
                "head_coach": "Head Coach",
                "posteam": "Team",
                "plays": "Plays",
                "games": "Games",
                "actual_pass_rate_pct": "Actual Pass %",
                "expected_pass_rate_pct": "Expected Pass %",
                "pass_oe_pct": "Pass OE",
                "mean_epa": "EPA/Play",
                "success_rate_pct": "Success %",
                "yards_per_play": "Yards/Play",
                "sample_quality": "Sample Quality",
            }
        )

        if share_records.empty:
            records_display = pd.DataFrame(
                columns=[
                    "Head Coach",
                    "Team",
                    "Games",
                    "Record",
                    "PCT",
                    "Points/Game",
                    "Points Allowed/Game",
                    "Point Diff/Game",
                ]
            )
        else:
            records_display = share_records[
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
                    "head_coach": "Head Coach",
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

        if share_uncertainty.empty:
            uncertainty_display = pd.DataFrame(
                columns=[
                    "Season",
                    "Head Coach",
                    "Team",
                    "Plays",
                    "Pass OE",
                    "95% CI Lower",
                    "95% CI Upper",
                    "Classification",
                ]
            )
        else:
            uncertainty_display = share_uncertainty.rename(
                columns={
                    "season": "Season",
                    "head_coach": "Head Coach",
                    "posteam": "Team",
                    "plays": "Plays",
                    "model_pass_oe_pct": "Pass OE",
                    "ci_95_lower_pct": "95% CI Lower",
                    "ci_95_upper_pct": "95% CI Upper",
                    "tendency_label": "Classification",
                }
            )

        render_share_summary_panel(
            share_summary,
            share_selection,
        )

        copy_tab, download_tab, data_tab = st.tabs([
            "Copy-Ready Summary",
            "Report Downloads",
            "Data Exports",
        ])

        with copy_tab:
            app_link = st.text_input(
                "Optional public app link",
                placeholder=(
                    "https://your-streamlit-app.streamlit.app"
                ),
                key="share_app_link",
                help=(
                    "When supplied, the link is appended to the copy-ready "
                    "text and included in the downloaded reports."
                ),
            )

            quick_text, detailed_text = (
                build_copy_ready_findings(
                    share_summary,
                    share_style,
                    share_uncertainty,
                    share_selection,
                    app_link=app_link,
                )
            )

            st.markdown(
                "<div class='section-eyebrow'>Copy-ready output</div>",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown(
                    "<div class='content-card-title'>Quick Share</div>"
                    "<div class='content-card-subtitle'>"
                    "Best for a LinkedIn post, message, portfolio update, "
                    "or short project description. Use the copy button in "
                    "the upper-right corner."
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.code(
                    quick_text,
                    language=None,
                    wrap_lines=True,
                )

            with st.container(border=True):
                st.markdown(
                    "<div class='content-card-title'>Detailed Share</div>"
                    "<div class='content-card-subtitle'>"
                    "A fuller summary with selection context, model "
                    "expectation, outcomes, uncertainty, and sample guidance."
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.code(
                    detailed_text,
                    language=None,
                    wrap_lines=True,
                )

            with st.expander("Preview the automatic observations"):
                preview_columns = st.columns(2)
                for index, insight in enumerate(share_insights):
                    with preview_columns[index % 2]:
                        render_insight_card(
                            insight["title"],
                            insight["body"],
                        )

        with download_tab:
            app_link_for_download = st.session_state.get(
                "share_app_link",
                "",
            )
            quick_download_text, detailed_download_text = (
                build_copy_ready_findings(
                    share_summary,
                    share_style,
                    share_uncertainty,
                    share_selection,
                    app_link=app_link_for_download,
                )
            )

            report_title = (
                "NFL Coaching Decision Lab: Filtered Findings"
            )
            file_stem = build_findings_file_stem()

            markdown_report = build_markdown_findings_report(
                report_title,
                share_selection,
                summary_metrics,
                share_insights,
                season_display,
                entity_display,
                records_display,
                uncertainty_display,
                app_link=app_link_for_download,
            )
            html_report = build_html_findings_report(
                report_title,
                share_selection,
                summary_metrics,
                share_insights,
                season_display,
                entity_display,
                records_display,
                uncertainty_display,
                app_link=app_link_for_download,
            )
            findings_zip = build_findings_zip(
                file_stem,
                html_report,
                markdown_report,
                quick_download_text,
                detailed_download_text,
                summary_export,
                share_season_data,
                share_entity_data,
                share_records,
                share_uncertainty,
            )

            st.markdown(
                "<div class='section-eyebrow'>Export formats</div>",
                unsafe_allow_html=True,
            )

            report_columns = st.columns(3, gap="large")

            with report_columns[0]:
                with st.container(border=True):
                    st.markdown("#### Complete Package")
                    st.caption(
                        "HTML, Markdown, copy-ready text, summary tables, "
                        "team records, and confidence intervals in one ZIP."
                    )
                    st.download_button(
                        "Download ZIP package",
                        data=findings_zip,
                        file_name=f"{file_stem}.zip",
                        mime="application/zip",
                        width="stretch",
                        key="download_findings_zip",
                    )

            with report_columns[1]:
                with st.container(border=True):
                    st.markdown("#### Shareable Report")
                    st.caption(
                        "A standalone responsive webpage designed for "
                        "sharing, mobile viewing, and printing."
                    )
                    st.download_button(
                        "Download HTML report",
                        data=html_report,
                        file_name=f"{file_stem}_report.html",
                        mime="text/html",
                        width="stretch",
                        key="download_findings_html",
                    )

            with report_columns[2]:
                with st.container(border=True):
                    st.markdown("#### Editable Report")
                    st.caption(
                        "A portable Markdown version that is easy to revise "
                        "for GitHub, notes, or a written analysis."
                    )
                    st.download_button(
                        "Download Markdown report",
                        data=markdown_report,
                        file_name=f"{file_stem}_report.md",
                        mime="text/markdown",
                        width="stretch",
                        key="download_findings_markdown",
                    )

            st.markdown("#### Report Preview")
            st.caption(
                "The HTML download uses the same selection, metrics, and "
                "automatic observations shown here and is formatted for "
                "desktop, mobile, and printing."
            )

            preview_metrics = st.columns(4)
            preview_metrics[0].metric(
                "Actual pass rate",
                f"{100 * share_summary['actual_pass_rate']:.2f}%",
            )
            preview_metrics[1].metric(
                "Expected pass rate",
                f"{100 * share_summary['expected_pass_rate']:.2f}%",
            )
            preview_metrics[2].metric(
                "EPA per play",
                f"{share_summary['mean_epa']:.3f}",
            )
            preview_metrics[3].metric(
                "Success rate",
                f"{100 * share_summary['success_rate']:.2f}%",
            )

        with data_tab:
            st.markdown("#### Aggregate CSV Files")
            st.caption(
                "These tables are designed for spreadsheet analysis and "
                "reproduce the active filters. Team records follow season "
                "and season-type filters, but not situation filters."
            )

            file_stem = build_findings_file_stem()
            csv_columns = st.columns(2)

            with csv_columns[0]:
                st.download_button(
                    "Download summary metrics CSV",
                    data=summary_export.to_csv(index=False),
                    file_name=f"{file_stem}_summary.csv",
                    mime="text/csv",
                    width="stretch",
                    key="download_summary_csv",
                )
                st.download_button(
                    "Download season breakdown CSV",
                    data=share_season_data.to_csv(index=False),
                    file_name=f"{file_stem}_by_season.csv",
                    mime="text/csv",
                    width="stretch",
                    key="download_season_csv",
                )

            with csv_columns[1]:
                st.download_button(
                    "Download decision-maker breakdown CSV",
                    data=share_entity_data.to_csv(index=False),
                    file_name=(
                        f"{file_stem}_by_decision_maker.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                    key="download_entity_csv",
                )
                st.download_button(
                    "Download confidence intervals CSV",
                    data=share_uncertainty.to_csv(index=False),
                    file_name=(
                        f"{file_stem}_confidence_intervals.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                    key="download_uncertainty_csv",
                    disabled=share_uncertainty.empty,
                    help=(
                        "Select a head coach with an eligible full-season "
                        "sample to enable this export."
                        if share_uncertainty.empty
                        else None
                    ),
                )

            with st.expander(
                "Advanced: prepare a compressed play-level export"
            ):
                st.caption(
                    "This export can be large when many seasons are "
                    "selected, so the app only prepares it after you check "
                    "the box below."
                )

                prepare_play_export = st.checkbox(
                    "Prepare filtered play-level CSV",
                    key="prepare_share_play_export",
                )

                if prepare_play_export:
                    if int(share_summary["plays"]) > 100000:
                        st.warning(
                            "The current selection contains more than "
                            "100,000 plays. Preparing the download may take "
                            "several seconds and produce a large file."
                        )

                    filtered_play_export = connection.execute(
                        f"""
                        SELECT
                            season,
                            season_type,
                            week,
                            game_id,
                            posteam,
                            defteam,
                            head_coach,
                            qtr,
                            down,
                            ydstogo,
                            yardline_100,
                            game_seconds_remaining,
                            score_differential,
                            play_call,
                            is_pass,
                            expected_pass_probability,
                            model_pass_oe,
                            epa,
                            success,
                            yards_gained
                        FROM play_predictions
                        {where_clause}
                        ORDER BY
                            season,
                            week,
                            game_id
                        """,
                        query_parameters,
                    ).fetchdf()

                    play_buffer = BytesIO()
                    filtered_play_export.to_csv(
                        play_buffer,
                        index=False,
                        compression="gzip",
                    )

                    st.download_button(
                        "Download compressed play-level CSV (.csv.gz)",
                        data=play_buffer.getvalue(),
                        file_name=(
                            f"{file_stem}_play_level.csv.gz"
                        ),
                        mime="application/gzip",
                        width="stretch",
                        key="download_play_level_csv",
                    )

                    st.caption(
                        f"Prepared {len(filtered_play_export):,} rows with "
                        "pre-snap context, model expectation, call type, "
                        "and descriptive play outcomes."
                    )

            st.markdown("#### Preview")
            st.dataframe(
                entity_display.head(50),
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
                ranked_attribution = (
                    eligible_attribution.sort_values(
                        "pass_oe_pct"
                    ).copy()
                )

                largest_absolute_pass_oe = max(
                    1.0,
                    float(
                        ranked_attribution[
                            "pass_oe_pct"
                        ].abs().max()
                    ),
                )
                axis_padding = max(
                    1.0,
                    0.18 * largest_absolute_pass_oe,
                )
                symmetric_limit = (
                    largest_absolute_pass_oe
                    + axis_padding
                )

                if len(ranked_attribution) == 1:
                    # A full ranking bar is visually oversized when only
                    # one attribution group qualifies. Use a compact
                    # zero-centered lollipop indicator instead.
                    single_row = ranked_attribution.iloc[0]
                    pass_oe_value = float(
                        single_row["pass_oe_pct"]
                    )
                    point_color = (
                        "#D73027"
                        if pass_oe_value < 0
                        else "#1F78B4"
                    )

                    ranking_figure = go.Figure()

                    ranking_figure.add_shape(
                        type="line",
                        x0=0,
                        x1=pass_oe_value,
                        y0=0,
                        y1=0,
                        line={
                            "color": point_color,
                            "width": 8,
                        },
                    )

                    ranking_figure.add_trace(
                        go.Scatter(
                            x=[pass_oe_value],
                            y=[0],
                            mode="markers+text",
                            marker={
                                "size": 18,
                                "color": point_color,
                                "line": {
                                    "color": "#F8FAFC",
                                    "width": 1.5,
                                },
                            },
                            text=[
                                f"{pass_oe_value:+.2f} pts"
                            ],
                            textposition=(
                                "middle left"
                                if pass_oe_value < 0
                                else "middle right"
                            ),
                            textfont={
                                "size": 15,
                                "color": "#F8FAFC",
                            },
                            customdata=[[
                                int(single_row["plays"]),
                                float(
                                    single_row[
                                        "actual_pass_rate_pct"
                                    ]
                                ),
                                float(
                                    single_row[
                                        "expected_pass_rate_pct"
                                    ]
                                ),
                                float(single_row["mean_epa"]),
                            ]],
                            hovertemplate=(
                                f"<b>{single_row['label']}</b><br>"
                                "Pass OE: %{x:+.2f} pts<br>"
                                "Plays: %{customdata[0]:,}<br>"
                                "Actual pass rate: "
                                "%{customdata[1]:.2f}%<br>"
                                "Expected pass rate: "
                                "%{customdata[2]:.2f}%<br>"
                                "EPA/play: %{customdata[3]:.4f}"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                    ranking_figure.add_vline(
                        x=0,
                        line_dash="dash",
                        line_color="rgba(255,255,255,0.48)",
                        line_width=2,
                    )

                    ranking_figure.add_annotation(
                        x=0,
                        y=0.72,
                        text="League-model expectation",
                        showarrow=False,
                        font={
                            "size": 12,
                            "color": "#AEB8C6",
                        },
                    )

                    ranking_figure.add_annotation(
                        x=-symmetric_limit,
                        y=-0.72,
                        text="More run-heavy",
                        showarrow=False,
                        xanchor="left",
                        font={
                            "size": 12,
                            "color": "#E58A84",
                        },
                    )

                    ranking_figure.add_annotation(
                        x=symmetric_limit,
                        y=-0.72,
                        text="More pass-heavy",
                        showarrow=False,
                        xanchor="right",
                        font={
                            "size": 12,
                            "color": "#82B7DE",
                        },
                    )

                    ranking_figure.update_xaxes(
                        range=[
                            -symmetric_limit,
                            symmetric_limit,
                        ],
                        autorange=False,
                        fixedrange=False,
                        title=(
                            "Pass OE "
                            "(percentage points)"
                        ),
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.08)",
                        zeroline=False,
                    )
                    ranking_figure.update_yaxes(
                        range=[-1, 1],
                        visible=False,
                        fixedrange=True,
                    )
                    ranking_figure.update_layout(
                        height=285,
                        margin={
                            "l": 48,
                            "r": 48,
                            "t": 30,
                            "b": 55,
                        },
                        hovermode="closest",
                    )
                    ranking_figure.layout.title = None

                    st.caption(
                        f"{single_row['label']} is the only "
                        "attribution group meeting the current "
                        f"{minimum_plays:,}-play minimum."
                    )
                else:
                    ranking_figure = px.bar(
                        ranked_attribution,
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
                                "Pass OE "
                                "(percentage points)"
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
                    ranking_figure.update_traces(
                        texttemplate="%{x:+.2f}",
                        textposition="outside",
                        cliponaxis=False,
                    )
                    ranking_figure.update_xaxes(
                        range=[
                            -symmetric_limit,
                            symmetric_limit,
                        ],
                        autorange=False,
                        zeroline=False,
                        fixedrange=False,
                    )
                    ranking_figure.update_layout(
                        coloraxis_showscale=False,
                        height=max(
                            420,
                            34 * len(
                                ranked_attribution
                            ),
                        ),
                    )
                    ranking_figure.layout.title = None

                render_plotly_chart(
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

    st.caption(
        "Review held-out 2025 performance and the stability of the "
        "out-of-sample expectation model across seasons. ROC AUC measures "
        "ranking discrimination, while Brier score and calibration error "
        "evaluate probability quality."
    )

    final_metrics = connection.execute(
        """
        SELECT *
        FROM final_test_metrics
        ORDER BY model
        """
    ).fetchdf()

    render_model_performance_summary(final_metrics)

    st.markdown(
        "<div class='section-eyebrow'>Held-out comparison</div>",
        unsafe_allow_html=True,
    )

    test_display = final_metrics.copy()
    test_display["model"] = test_display["model"].replace({
        "final_hist_gradient_boosting": "Coaching Lab locked model",
        "nflverse_xpass": "nflverse xpass benchmark",
    })
    test_display = test_display.rename(
        columns={
            "model": "Model",
            "plays": "Plays",
            "accuracy": "Accuracy",
            "roc_auc": "ROC AUC",
            "log_loss": "Log Loss",
            "brier_score": "Brier Score",
            "expected_calibration_error": "Calibration Error",
            "actual_pass_rate": "Actual Pass Rate",
            "expected_pass_rate": "Expected Pass Rate",
            "mean_pass_oe": "Mean Pass OE",
        }
    )

    primary_test_columns = [
        "Model",
        "Plays",
        "Accuracy",
        "ROC AUC",
        "Log Loss",
        "Brier Score",
        "Calibration Error",
    ]

    with st.container(border=True):
        st.markdown(
            "<div class='content-card-title'>Held-Out 2025 Test Results</div>"
            "<div class='content-card-subtitle'>"
            "The locked model was selected before the 2025 test season. "
            "Lower log loss, Brier score, and calibration error indicate "
            "better probability estimates."
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            test_display[primary_test_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "Plays": st.column_config.NumberColumn(
                    "Plays",
                    format="%d",
                ),
                "Accuracy": st.column_config.NumberColumn(
                    "Accuracy",
                    format="%.4f",
                ),
                "ROC AUC": st.column_config.NumberColumn(
                    "ROC AUC",
                    format="%.4f",
                ),
                "Log Loss": st.column_config.NumberColumn(
                    "Log Loss",
                    format="%.4f",
                ),
                "Brier Score": st.column_config.NumberColumn(
                    "Brier Score",
                    format="%.4f",
                ),
                "Calibration Error": st.column_config.NumberColumn(
                    "Calibration Error",
                    format="%.4f",
                ),
            },
        )

        with st.expander("View complete held-out diagnostics"):
            st.dataframe(
                test_display,
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

    season_chart_data = season_metrics.copy()
    season_chart_data["Series"] = (
        season_chart_data["model"].replace({
            "cross_fitted_locked_model": "Coaching Lab model",
            "final_locked_model": "Coaching Lab model",
            "nflverse_xpass": "nflverse xpass",
        })
    )
    season_chart_data["Evaluation"] = (
        season_chart_data["model"].replace({
            "cross_fitted_locked_model": (
                "Leave-one-season-out cross-fitted"
            ),
            "final_locked_model": "Held-out final test",
            "nflverse_xpass": "External benchmark",
        })
    )

    st.markdown(
        "<div class='section-eyebrow'>Historical stability</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "For visual continuity, the Coaching Lab series combines "
        "cross-fitted 2018-2024 results with the separately held-out 2025 "
        "final-model result. Hover text identifies the evaluation design."
    )

    model_colors = {
        "Coaching Lab model": active_primary,
        "nflverse xpass": "#4DA3FF",
    }

    trend_columns = st.columns(2, gap="large")

    with trend_columns[0]:
        with st.container(border=True):
            st.markdown(
                "<div class='content-card-title'>ROC AUC by Season</div>"
                "<div class='content-card-subtitle'>"
                "Higher values indicate better ranking of pass versus run "
                "calls across the held-out plays."
                "</div>",
                unsafe_allow_html=True,
            )

            auc_figure = px.line(
                season_chart_data,
                x="season",
                y="roc_auc",
                color="Series",
                markers=True,
                color_discrete_map=model_colors,
                custom_data=["Evaluation", "model"],
                labels={
                    "season": "Season",
                    "roc_auc": "ROC AUC",
                },
            )
            auc_figure.update_traces(
                line={"width": 3},
                marker={"size": 8},
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Season: %{x}<br>"
                    "ROC AUC: %{y:.4f}<br>"
                    "Evaluation: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
            auc_min = float(season_chart_data["roc_auc"].min())
            auc_max = float(season_chart_data["roc_auc"].max())
            auc_pad = max(0.002, 0.18 * (auc_max - auc_min))
            auc_figure.update_yaxes(
                range=[auc_min - auc_pad, auc_max + auc_pad],
                tickformat=".3f",
            )
            auc_figure.update_layout(
                height=390,
                hovermode="x unified",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "left",
                    "x": 0,
                    "title": None,
                },
            )
            render_plotly_chart(
                auc_figure,
                width="stretch",
            )

    with trend_columns[1]:
        with st.container(border=True):
            st.markdown(
                "<div class='content-card-title'>Brier Score by Season</div>"
                "<div class='content-card-subtitle'>"
                "Lower values indicate more accurate probability forecasts "
                "and penalize confident errors more heavily."
                "</div>",
                unsafe_allow_html=True,
            )

            brier_figure = px.line(
                season_chart_data,
                x="season",
                y="brier_score",
                color="Series",
                markers=True,
                color_discrete_map=model_colors,
                custom_data=["Evaluation", "model"],
                labels={
                    "season": "Season",
                    "brier_score": "Brier Score",
                },
            )
            brier_figure.update_traces(
                line={"width": 3},
                marker={"size": 8},
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Season: %{x}<br>"
                    "Brier score: %{y:.4f}<br>"
                    "Evaluation: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
            brier_min = float(
                season_chart_data["brier_score"].min()
            )
            brier_max = float(
                season_chart_data["brier_score"].max()
            )
            brier_pad = max(
                0.001,
                0.18 * (brier_max - brier_min),
            )
            brier_figure.update_yaxes(
                range=[
                    brier_min - brier_pad,
                    brier_max + brier_pad,
                ],
                tickformat=".3f",
            )
            brier_figure.update_layout(
                height=390,
                hovermode="x unified",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "left",
                    "x": 0,
                    "title": None,
                },
            )
            render_plotly_chart(
                brier_figure,
                width="stretch",
            )

    with st.expander("How to read the model metrics"):
        guide_columns = st.columns(4)
        with guide_columns[0]:
            st.markdown("**ROC AUC**")
            st.caption(
                "How well the model ranks pass calls above run calls. "
                "Higher is better."
            )
        with guide_columns[1]:
            st.markdown("**Log loss**")
            st.caption(
                "Probability error with a strong penalty for confident "
                "mistakes. Lower is better."
            )
        with guide_columns[2]:
            st.markdown("**Brier score**")
            st.caption(
                "Mean squared error of the predicted pass probabilities. "
                "Lower is better."
            )
        with guide_columns[3]:
            st.markdown("**Calibration error**")
            st.caption(
                "How closely predicted probabilities match observed rates. "
                "Lower is better."
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