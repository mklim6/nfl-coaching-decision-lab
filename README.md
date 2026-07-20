# NFL Coaching Decision Lab

The NFL Coaching Decision Lab is an interactive Streamlit application for
exploring NFL run-pass decisions relative to a common league-wide expectation
model. It compares head coaches and verified offensive play callers across
game situations, seasons, teams, and observable offensive styles.

The project covers the 2018 through 2025 NFL seasons and includes 275,610
competitive run-pass decisions.

## Live application

Add the public Streamlit URL here after deployment.

## Main features

- League-wide run-pass expectation model using pre-snap information
- Head-coach and manually verified offensive play-caller attribution
- Coach and play-caller rankings using Pass Rate Over Expected
- Historical trends with out-of-sample season predictions
- Coach and decision-maker comparison tools
- Regular-season and postseason filters
- Team records, scoring context, EPA, success rate, and yardage
- Game-clustered confidence intervals with minimum-sample safeguards
- Play Style Explorer for formation, tempo, pass depth, pass direction,
  run direction, and charted run gap
- Call Sheet Situation Lab with adaptive historical matching
- Randomized Make the Call game with difficulty and sample controls

## Model design

A single league-wide histogram gradient boosting classifier estimates the
probability that an eligible play is a pass. All decision-makers are compared
against the same situation-only baseline.

Model inputs are limited to information available before the snap, including:

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

Historical 2018 through 2024 displays use leave-one-season-out predictions.
The final 2025 test produced approximately:

- Accuracy: 0.7114
- ROC AUC: 0.7930
- Log loss: 0.5265
- Brier score: 0.1791
- Expected calibration error: 0.0068

## Pass Rate Over Expected

For each play:

```text
actual pass indicator - expected pass probability
```

Pass Rate Over Expected, or Pass OE, is the average residual expressed in
percentage points. Positive values indicate more passing than the model
expected in the selected situations. Negative values indicate less passing.

Pass OE describes a tendency relative to modeled context. It is not a causal
estimate or a claim that passing or running more often would improve results.

## Data and attribution

Play-by-play and schedule data come from nflverse. Pass calls include designed
dropbacks, sacks, and quarterback scrambles that began as pass plays.
Kneel-downs, spikes, aborted plays, and plays without the required fields are
excluded.

Offensive play-caller attribution is manually curated by team, season, and
week so midseason responsibility changes can be preserved. Published calling
responsibility may still reflect collaborative decision-making.

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

The app is designed for Streamlit Community Cloud. Keep `requirements.txt`
at the repository root and deploy with `app/app.py` as the entrypoint.

During deployment:

1. Connect Streamlit Community Cloud to the GitHub repository.
2. Choose the repository and its main branch.
3. Enter `app/app.py` as the app file path.
4. Open **Advanced settings** and select Python 3.12.
5. Deploy the app and review its startup logs.

All files read by the app, including `database/coaching_lab.duckdb`, must be
committed to the repository and preserve the paths shown below.

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
- Make the Call rewards matching the historical majority decision. It does
  not identify the strategically optimal play.

## Author

Matthew Klima

Statistics graduate from the University of Illinois Urbana-Champaign with an
interest in football analytics, scouting, and accessible sports data tools.