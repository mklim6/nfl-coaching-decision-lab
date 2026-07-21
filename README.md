# NFL Coaching Decision Lab

The NFL Coaching Decision Lab is an interactive Streamlit application for
exploring NFL run-pass decisions relative to a shared league-wide expectation
model.

The app compares head coaches and verified offensive play callers across
seasons, teams, game situations, and observable offensive styles. It covers
the 2018 through 2025 NFL seasons and includes 275,610 competitive run-pass
decisions.

## Live application

[Open the NFL Coaching Decision Lab](https://nfl-coaching-decision-lab-ufleodcg7vfpvip6pdyqkv.streamlit.app/)

## Project overview

The central question behind the project is:

> How often does a coach or play caller choose to pass relative to what the
> game situation would normally suggest?

A league-wide machine-learning model estimates the probability of a pass
using only information available before the snap. Each decision-maker is then
compared against that same situation-based baseline.

The result is Pass Rate Over Expected, or Pass OE, which describes whether a
team passed more or less often than expected in comparable situations.

## Main features

- Interactive filters for season, season type, coach, team, down, quarter,
  distance, field position, and score differential
- Context-adjusted coach and team rankings using Pass OE
- Decision-maker profiles with tendency, outcome, style, and sample-quality
  summaries
- Side-by-side coach comparison tools
- Historical trends built from out-of-sample season predictions
- Verified offensive play-caller attribution by team, season, and week
- Regular-season and postseason analysis
- Team records, scoring context, EPA, success rate, and yards per play
- Game-clustered confidence intervals with minimum-sample safeguards
- Play Style Explorer for shotgun, no huddle, pass depth, pass direction,
  run direction, and charted run gap
- Situation Lab for finding comparable historical play-calling decisions
- Make the Call game with randomized situations and detailed results
- Share Findings tools for copy-ready summaries, HTML reports, Markdown
  reports, ZIP packages, and CSV exports
- Model Performance section with held-out evaluation and seasonal diagnostics
- Team-responsive visual themes and mobile-friendly layouts

## Model design

A single league-wide histogram gradient boosting classifier estimates the
probability that an eligible play is a pass.

Every coach and play caller is compared against the same model, allowing the
app to separate decision-making tendency from basic game context.

Model inputs are limited to pre-snap information, including:

- Week and season type
- Quarter, down, and distance
- Field position and goal-to-go status
- Game time remaining
- Score differential
- Offensive and defensive timeouts
- Home and opening-kickoff context
- Roof and playing surface
- Pregame point spread and game total

Coach identity, team identity, play outcomes, EPA, success, yards gained,
formation, and nflverse xpass are not model inputs.

## Evaluation design

- Training: 2018 through 2023
- Validation and model selection: 2024
- Final training: 2018 through 2024
- Held-out final test: 2025

Historical displays for 2018 through 2024 use leave-one-season-out
cross-fitted predictions. The 2025 season remains a separate held-out final
test.

### Held-out 2025 results

| Metric | Coaching Lab model |
|---|---:|
| Accuracy | 0.7114 |
| ROC AUC | 0.7930 |
| Log loss | 0.5265 |
| Brier score | 0.1791 |
| Expected calibration error | 0.0068 |

The final model slightly outperformed nflverse xpass on ROC AUC, log loss,
Brier score, and calibration error in the held-out 2025 sample.

## Pass Rate Over Expected

For each play:

```text
actual pass indicator - expected pass probability
```

Pass OE is the average residual expressed in percentage points.

- Positive Pass OE indicates more passing than expected
- Negative Pass OE indicates more rushing than expected
- Values near zero indicate a call mix close to the modeled expectation

Pass OE describes play-calling tendency relative to modeled context. It is not
a causal estimate and does not claim that passing or rushing more often would
improve results.

## Data and attribution

Play-by-play and schedule data come from nflverse.

Pass calls include designed dropbacks, sacks, and quarterback scrambles that
began as pass plays. Kneel-downs, spikes, aborted plays, and plays without the
required fields are excluded.

Head-coach attribution comes from the coach attached to each play by nflverse.

Offensive play-caller attribution is manually curated by team, season, and
week so midseason responsibility changes can be preserved. Published
play-calling responsibility may still reflect collaborative decision-making.

## Technology

- Python 3.12
- Streamlit
- DuckDB
- pandas
- Plotly
- scikit-learn histogram gradient boosting
- nflverse data

## Local setup

Python 3.12 is recommended.

```powershell
conda activate nfl-coaching-lab
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The application expects the deployment database at:

```text
database/coaching_lab.duckdb
```

## Public deployment

The app is designed for Streamlit Community Cloud.

Use the following deployment settings:

- Repository branch: `main`
- App file path: `app/app.py`
- Python version: `3.12`

All files required by the app, including
`database/coaching_lab.duckdb`, must be committed to the repository.

## Repository structure

```text
NFL Coaching Decision Lab/
|-- app/
|   `-- app.py
|-- data/
|   |-- processed/
|   |-- raw/
|   `-- reference/
|-- database/
|   `-- coaching_lab.duckdb
|-- models/
|-- outputs/
|   |-- figures/
|   `-- tables/
|-- src/
|-- tests/
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Important limitations

- The model cannot observe every pre-snap consideration, including complete
  personnel packages, defensive alignment, injuries, audibles, and private
  game-plan information.
- Play-style fields describe recorded characteristics and should not be
  treated as definitive offensive scheme classifications.
- Outcomes are descriptive context rather than causal estimates.
- Small filtered samples are less stable and should be interpreted alongside
  the displayed sample warnings.
- Verified play-caller research is based on publicly available reporting and
  may not capture every collaborative or situational responsibility.
- Make the Call rewards matching the historical majority decision. It does
  not identify the strategically optimal play.

## License

The original code in this repository is available under the MIT License. See
the `LICENSE` file for details.

Third-party data and derived fields remain subject to the terms and licenses
of their original sources, including nflverse.

## Author

Matthew Klima

Statistics graduate from the University of Illinois Urbana-Champaign with an
interest in football analytics, scouting, and accessible sports data tools.