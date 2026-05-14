from __future__ import annotations

import json
import math
import sys
from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FLY_DIR = PROJECT_ROOT / "fly"
MODEL_PATH = FLY_DIR / "flight_delay_model_artifacts" / "best_delay_model_pipeline.joblib"
METADATA_PATH = FLY_DIR / "flight_delay_model_artifacts" / "best_delay_model_metadata.json"
AIRPORTS_PATH = PROJECT_ROOT / "airports.csv"

if str(FLY_DIR) not in sys.path:
    sys.path.insert(0, str(FLY_DIR))


POPULAR_AIRPORTS = [
    "ATL",
    "BOS",
    "BWI",
    "CLT",
    "DCA",
    "DEN",
    "DFW",
    "DTW",
    "EWR",
    "FLL",
    "IAD",
    "IAH",
    "JFK",
    "LAS",
    "LAX",
    "LGA",
    "MCO",
    "MDW",
    "MIA",
    "MSP",
    "ORD",
    "PHL",
    "PHX",
    "SAN",
    "SEA",
    "SFO",
]

AIRLINES = {
    "American Airlines": "AA",
    "Alaska Airlines": "AS",
    "JetBlue": "B6",
    "Delta Air Lines": "DL",
    "Frontier Airlines": "F9",
    "Allegiant Air": "G4",
    "Hawaiian Airlines": "HA",
    "Spirit Airlines": "NK",
    "United Airlines": "UA",
    "Southwest Airlines": "WN",
    "SkyWest Airlines": "OO",
    "Republic Airways": "YX",
}

AIRPORT_FALLBACK = {
    "ATL": ("Atlanta Hartsfield-Jackson", 33.6367, -84.4281),
    "BOS": ("Boston Logan", 42.3656, -71.0096),
    "CLT": ("Charlotte Douglas", 35.2140, -80.9431),
    "DEN": ("Denver", 39.8561, -104.6737),
    "DFW": ("Dallas/Fort Worth", 32.8998, -97.0403),
    "JFK": ("New York JFK", 40.6413, -73.7781),
    "LAS": ("Las Vegas Harry Reid", 36.0840, -115.1537),
    "LAX": ("Los Angeles", 33.9416, -118.4085),
    "MIA": ("Miami", 25.7959, -80.2870),
    "ORD": ("Chicago O'Hare", 41.9742, -87.9073),
    "PHX": ("Phoenix Sky Harbor", 33.4352, -112.0101),
    "SEA": ("Seattle-Tacoma", 47.4502, -122.3088),
    "SFO": ("San Francisco", 37.6213, -122.3790),
}

WEATHER_PRESETS = {
    "Clear": {
        "temperature_2m": 24.0,
        "relative_humidity_2m": 45.0,
        "precipitation": 0.0,
        "snow_depth": 0.0,
        "surface_pressure": 1018.0,
        "cloud_cover": 8.0,
        "wind_speed_10m": 8.0,
        "wind_gusts_10m": 12.0,
        "wind_direction": 250,
    },
    "Cloudy": {
        "temperature_2m": 18.0,
        "relative_humidity_2m": 68.0,
        "precipitation": 0.2,
        "snow_depth": 0.0,
        "surface_pressure": 1013.0,
        "cloud_cover": 75.0,
        "wind_speed_10m": 16.0,
        "wind_gusts_10m": 24.0,
        "wind_direction": 230,
    },
    "Rain": {
        "temperature_2m": 17.0,
        "relative_humidity_2m": 86.0,
        "precipitation": 6.0,
        "snow_depth": 0.0,
        "surface_pressure": 1007.0,
        "cloud_cover": 92.0,
        "wind_speed_10m": 24.0,
        "wind_gusts_10m": 38.0,
        "wind_direction": 210,
    },
    "Storm": {
        "temperature_2m": 23.0,
        "relative_humidity_2m": 88.0,
        "precipitation": 18.0,
        "snow_depth": 0.0,
        "surface_pressure": 997.0,
        "cloud_cover": 98.0,
        "wind_speed_10m": 45.0,
        "wind_gusts_10m": 76.0,
        "wind_direction": 205,
    },
    "Snow": {
        "temperature_2m": -4.0,
        "relative_humidity_2m": 82.0,
        "precipitation": 5.0,
        "snow_depth": 0.25,
        "surface_pressure": 1005.0,
        "cloud_cover": 95.0,
        "wind_speed_10m": 24.0,
        "wind_gusts_10m": 42.0,
        "wind_direction": 20,
    },
    "Fog": {
        "temperature_2m": 9.0,
        "relative_humidity_2m": 97.0,
        "precipitation": 0.2,
        "snow_depth": 0.0,
        "surface_pressure": 1011.0,
        "cloud_cover": 100.0,
        "wind_speed_10m": 5.0,
        "wind_gusts_10m": 8.0,
        "wind_direction": 160,
    },
    "Windy": {
        "temperature_2m": 20.0,
        "relative_humidity_2m": 55.0,
        "precipitation": 0.5,
        "snow_depth": 0.0,
        "surface_pressure": 1009.0,
        "cloud_cover": 45.0,
        "wind_speed_10m": 44.0,
        "wind_gusts_10m": 68.0,
        "wind_direction": 275,
    },
}

TRAFFIC_PRESETS = {
    "Quiet": {"traffic_level": 0.25, "origin_departures": 250, "dest_arrivals": 250},
    "Normal": {"traffic_level": 0.45, "origin_departures": 550, "dest_arrivals": 550},
    "Busy": {"traffic_level": 0.70, "origin_departures": 900, "dest_arrivals": 900},
    "Very busy": {"traffic_level": 0.92, "origin_departures": 1250, "dest_arrivals": 1250},
}

RECENT_DELAY_PRESETS = {
    "Low": 0.08,
    "Some": 0.18,
    "Many": 0.35,
}


st.set_page_config(page_title="Flight Delay Predictor", layout="wide")


st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .risk-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .risk-percent {
        font-size: 56px;
        line-height: 1;
        font-weight: 800;
        color: #0f766e;
        margin: 4px 0 10px 0;
    }
    .risk-label {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 12px;
        font-weight: 700;
        background: #e0f2fe;
        color: #075985;
    }
    .action-box {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        padding: 12px 14px;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_airports() -> pd.DataFrame:
    if not AIRPORTS_PATH.exists():
        rows = [
            {"iata_code": code, "name": values[0], "latitude_deg": values[1], "longitude_deg": values[2]}
            for code, values in AIRPORT_FALLBACK.items()
        ]
        return pd.DataFrame(rows)

    cols = ["iata_code", "name", "latitude_deg", "longitude_deg", "scheduled_service"]
    airports = pd.read_csv(AIRPORTS_PATH, usecols=cols)
    airports = airports[
        airports["iata_code"].isin(POPULAR_AIRPORTS)
        & airports["latitude_deg"].notna()
        & airports["longitude_deg"].notna()
    ].copy()
    airports["name"] = airports["name"].fillna(airports["iata_code"])
    return airports.drop_duplicates("iata_code").sort_values("iata_code")


def airport_label_map(airports: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for _, row in airports.iterrows():
        code = str(row["iata_code"])
        labels[f"{code} - {row['name']}"] = code
    return labels


def haversine_miles(origin: str, dest: str, airports: pd.DataFrame) -> float:
    lookup = airports.set_index("iata_code")
    if origin not in lookup.index or dest not in lookup.index:
        return 500.0
    o = lookup.loc[origin]
    d = lookup.loc[dest]
    lat1 = math.radians(float(o["latitude_deg"]))
    lon1 = math.radians(float(o["longitude_deg"]))
    lat2 = math.radians(float(d["latitude_deg"]))
    lon2 = math.radians(float(d["longitude_deg"]))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def distance_group(distance_miles: float) -> int:
    return int(max(1, min(11, math.ceil(distance_miles / 250))))


def dep_time_slot(hour: int) -> str:
    if 5 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 21:
        return "Evening"
    return "Night"


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def holiday_flags(flight_date: date) -> dict[str, int]:
    fixed_holidays = {(1, 1), (7, 4), (11, 11), (12, 24), (12, 25), (12, 31)}
    month_day = (flight_date.month, flight_date.day)
    near_holiday = int(
        month_day in fixed_holidays
        or (flight_date.month == 11 and 20 <= flight_date.day <= 30)
        or (flight_date.month == 12 and 20 <= flight_date.day <= 31)
    )
    return {
        "IS_HOLIDAY": int(month_day in fixed_holidays),
        "IS_NEAR_HOLIDAY": near_holiday,
        "IS_PEAK_TRAVEL": int(flight_date.month in (6, 7, 8, 11, 12)),
        "IS_SUMMER_BREAK": int(flight_date.month in (6, 7, 8)),
        "IS_WINTER_BREAK": int(flight_date.month in (12, 1)),
        "IS_SPRING_BREAK": int(flight_date.month == 3),
        "IS_SCHOOL_BREAK": int(flight_date.month in (3, 6, 7, 8, 12, 1)),
    }


def build_feature_row(
    flight_date: date,
    departure_time: time,
    airline_code: str,
    origin: str,
    dest: str,
    weather_name: str,
    traffic_name: str,
    recent_delay_name: str,
    airports: pd.DataFrame,
) -> pd.DataFrame:
    hour = int(departure_time.hour)
    distance = haversine_miles(origin, dest, airports)
    weather = WEATHER_PRESETS[weather_name]
    traffic = TRAFFIC_PRESETS[traffic_name]
    wind_rad = math.radians(weather["wind_direction"])
    row = {
        "Year": flight_date.year,
        "Quarter": int(math.ceil(flight_date.month / 3)),
        "Month": flight_date.month,
        "DayofMonth": flight_date.day,
        "DayOfWeek": int(flight_date.isoweekday()),
        "CRSDepHour": hour,
        "is_peak_hour": int(hour in (7, 8, 9, 16, 17, 18, 19)),
        "Distance": round(distance, 1),
        "DistanceGroup": distance_group(distance),
        "Origin": origin,
        "Dest": dest,
        "Operating_Airline": airline_code,
        "ROUTE": f"{origin}-{dest}",
        "Origin_freq": traffic["origin_departures"],
        "Dest_freq": traffic["dest_arrivals"],
        "IS_WEEKEND": int(flight_date.weekday() >= 5),
        "prev_delay": RECENT_DELAY_PRESETS[recent_delay_name],
        "traffic_level": traffic["traffic_level"],
        "temperature_2m": weather["temperature_2m"],
        "relative_humidity_2m": weather["relative_humidity_2m"],
        "precipitation": weather["precipitation"],
        "snow_depth": weather["snow_depth"],
        "surface_pressure": weather["surface_pressure"],
        "cloud_cover": weather["cloud_cover"],
        "wind_speed_10m": weather["wind_speed_10m"],
        "wind_gusts_10m": weather["wind_gusts_10m"],
        "wind_dir_sin": math.sin(wind_rad),
        "wind_dir_cos": math.cos(wind_rad),
    }
    row.update(holiday_flags(flight_date))

    for code in ["9E", "AA", "AS", "B6", "C5", "DL", "F9", "G4", "G7", "HA", "MQ", "NK", "OH", "OO", "PT", "QX", "UA", "WN", "YV", "YX", "ZW"]:
        row[f"Operating_Airline_{code}"] = int(code == airline_code)

    for slot in ["Afternoon", "Evening", "Morning", "Night"]:
        row[f"DEP_TIME_SLOT_{slot}"] = int(slot == dep_time_slot(hour))

    for name in ["Fall", "Spring", "Summer", "Winter"]:
        row[f"SEASON_{name}"] = int(name == season(flight_date.month))

    return pd.DataFrame([row])


def score_frame(artifact: dict, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probabilities = artifact["pipeline"].predict_proba(frame.copy())[:, 1]
    result = frame.copy()
    result["delay_probability"] = probabilities
    result["predicted_delay"] = (probabilities >= threshold).astype(int)
    return result


def risk_label(probability: float, threshold: float) -> str:
    if probability < 0.15:
        return "Low risk"
    if probability < threshold:
        return "Moderate risk"
    if probability < 0.35:
        return "Delay likely"
    return "High risk"


def action_text(label: str) -> str:
    if label in {"Delay likely", "High risk"}:
        return "Check flight updates early, allow extra time, and watch for gate or crew schedule changes."
    if label == "Moderate risk":
        return "The flight has some delay pressure. Keep alerts on and re-check closer to departure."
    return "The flight looks reasonably stable, but keep normal flight alerts switched on."


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


if not MODEL_PATH.exists():
    st.error(f"Model artifact not found: {MODEL_PATH}")
    st.stop()

artifact = load_artifact()
metadata = load_metadata()
airports = load_airports()
airport_labels = airport_label_map(airports)
airport_options = list(airport_labels)
default_threshold = float(artifact.get("threshold", metadata.get("best_threshold", 0.21)))

st.title("Flight Delay Predictor")

left, right = st.columns([1, 1], gap="large")

with left:
    with st.form("flight_form"):
        st.subheader("Flight Details")
        route_cols = st.columns(2)
        origin_index = list(airport_labels.values()).index("JFK") if "JFK" in airport_labels.values() else 0
        dest_index = list(airport_labels.values()).index("LAX") if "LAX" in airport_labels.values() else min(1, len(airport_options) - 1)
        origin_label = route_cols[0].selectbox("From", airport_options, index=origin_index)
        dest_label = route_cols[1].selectbox("To", airport_options, index=dest_index)

        schedule_cols = st.columns(2)
        selected_date = schedule_cols[0].date_input("Flight date", value=date(2025, 7, 18))
        selected_time = schedule_cols[1].time_input("Departure time", value=time(17, 30))

        airline_label = st.selectbox("Airline", list(AIRLINES), index=list(AIRLINES).index("American Airlines"))

        st.subheader("Current Conditions")
        weather_name = st.selectbox("Weather", list(WEATHER_PRESETS), index=list(WEATHER_PRESETS).index("Cloudy"))

        traffic_name = "Normal"
        recent_delay_name = "Low"
        with st.expander("Optional airport status"):
            traffic_name = st.selectbox(
                "Airport traffic",
                list(TRAFFIC_PRESETS),
                index=list(TRAFFIC_PRESETS).index("Normal"),
            )
            recent_delay_name = st.selectbox(
                "Known recent delays",
                list(RECENT_DELAY_PRESETS),
                index=list(RECENT_DELAY_PRESETS).index("Low"),
            )

        submitted = st.form_submit_button("Predict delay", type="primary")

origin = airport_labels[origin_label]
dest = airport_labels[dest_label]
airline_code = AIRLINES[airline_label]
feature_row = build_feature_row(
    selected_date,
    selected_time,
    airline_code,
    origin,
    dest,
    weather_name,
    traffic_name,
    recent_delay_name,
    airports,
)

if origin == dest:
    right.warning("Choose different origin and destination airports.")
elif submitted or "last_probability" not in st.session_state:
    scored = score_frame(artifact, feature_row, default_threshold)
    st.session_state.last_probability = float(scored["delay_probability"].iloc[0])
    st.session_state.last_prediction = int(scored["predicted_delay"].iloc[0])
    st.session_state.last_row = feature_row

probability = float(st.session_state.get("last_probability", 0.0))
label = risk_label(probability, default_threshold)
classification = "Delay likely" if probability >= default_threshold else "Delay not likely"

with right:
    st.subheader("Delay Probability")
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-percent">{probability:.0%}</div>
            <span class="risk-label">{label}</span>
            <div class="action-box">{action_text(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, probability))

    result_cols = st.columns(3)
    result_cols[0].metric("Prediction", classification)
    result_cols[1].metric("Route", f"{origin} to {dest}")
    result_cols[2].metric("Distance", f"{float(feature_row['Distance'].iloc[0]):,.0f} mi")

    summary = pd.DataFrame(
        [
            {"Item": "Airline", "Value": f"{airline_label} ({airline_code})"},
            {"Item": "Departure", "Value": f"{selected_date:%b %d, %Y} at {selected_time:%H:%M}"},
            {"Item": "Weather", "Value": weather_name},
            {"Item": "Airport traffic", "Value": traffic_name},
            {"Item": "Recent delays", "Value": recent_delay_name},
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")

with st.expander("Batch scoring for staff"):
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    sample_path = APP_DIR / "sample_flights.csv"
    if sample_path.exists():
        st.download_button(
            "Sample CSV",
            data=sample_path.read_bytes(),
            file_name="sample_flights.csv",
            mime="text/csv",
        )

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        missing = [column for column in ["Origin", "Dest", "Operating_Airline", "CRSDepHour"] if column not in raw_df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            batch_scored = score_frame(artifact, raw_df, default_threshold)
            batch_scored["Delay Probability"] = batch_scored["delay_probability"].map(lambda value: f"{value:.1%}")
            batch_scored["Output"] = batch_scored["predicted_delay"].map({0: "Delay not likely", 1: "Delay likely"})
            st.metric("Flights scored", f"{len(batch_scored):,}")
            preview_cols = [
                col
                for col in ["Year", "Month", "DayofMonth", "Operating_Airline", "Origin", "Dest", "CRSDepHour", "Delay Probability", "Output"]
                if col in batch_scored.columns
            ]
            st.dataframe(batch_scored[preview_cols].head(50), width="stretch")
            st.download_button(
                "Download predictions",
                data=csv_bytes(batch_scored),
                file_name="flight_delay_predictions.csv",
                mime="text/csv",
            )
