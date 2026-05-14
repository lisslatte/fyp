# Flight Delay Classifier Demo

Interactive Streamlit prototype for the saved flight delay classifier in `../fly/flight_delay_model_artifacts`.

## Run

```powershell
cd D:\FYP\flight_delay_classifier_demo
streamlit run app.py
```

The demo includes:

- Simple single-flight inputs for route, airline, schedule, weather, airport traffic, and recent delays.
- A clear delay probability with a passenger, crew, or airport staff action message.
- Optional CSV batch scoring for staff.

## Model

The app loads:

```text
..\fly\flight_delay_model_artifacts\best_delay_model_pipeline.joblib
```

The current saved artifact is the Random Forest classifier with the trained feature builder bundled inside the joblib file.
