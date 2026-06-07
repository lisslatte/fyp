from __future__ import annotations

import argparse
import gc
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CACHE_DIR = PROJECT_ROOT / "weather_safe_model_artifacts_meteostat_upgraded_v4_v1_2_1"
ARTIFACT_DIR = PROJECT_ROOT / "weather_safe_model_artifacts_meteostat_upgraded_v4_v1_2_2"
AIRPORTS_CSV = PROJECT_ROOT / "airports.csv"
NOAA_DATA_DIR = PROJECT_ROOT / "fly" / "noaa_data"
SOURCE_WEATHER_HISTORY = SOURCE_CACHE_DIR / "weather_safe_history_meteostat_hourly.parquet"

SOURCE_FILES = {
    "train": SOURCE_CACHE_DIR / "train_2022_2023_2500000_weather_safe_upgraded_v11.parquet",
    "cv": SOURCE_CACHE_DIR / "cv_2024_250000_weather_safe_upgraded_v11.parquet",
    "test": SOURCE_CACHE_DIR / "test_2025_250000_weather_safe_upgraded_v11.parquet",
}

OUTPUT_FILES = {
    "train": ARTIFACT_DIR / "train_2022_2023_2500000_weather_safe_upgraded_v12.parquet",
    "cv": ARTIFACT_DIR / "cv_2024_250000_weather_safe_upgraded_v12.parquet",
    "test": ARTIFACT_DIR / "test_2025_250000_weather_safe_upgraded_v12.parquet",
}

AUX_FILES = {
    "airport_metadata": ARTIFACT_DIR / "airport_metadata_v1_2_2.csv",
    "station_inventory": ARTIFACT_DIR / "noaa_station_inventory_v1_2_2.csv",
    "hourly_station_mapping": ARTIFACT_DIR / "hourly_station_mapping_v1_2_2.csv",
    "enhanced_weather_history": ARTIFACT_DIR / "enhanced_weather_history_v1_2_2.parquet",
    "metar_hourly": ARTIFACT_DIR / "metar_hourly_features_v1_2_2.parquet",
    "station_daily_snow": ARTIFACT_DIR / "station_daily_snow_features_v1_2_2.parquet",
    "origin_airport_month_norms": ARTIFACT_DIR / "origin_airport_month_weather_norms_v1_2_2.csv",
    "ghcn_stations_text": ARTIFACT_DIR / "ghcnd_stations_v1_2_2.txt",
    "ghcn_inventory_text": ARTIFACT_DIR / "ghcnd_inventory_v1_2_2.txt",
    "ghcn_station_summary": ARTIFACT_DIR / "ghcnd_station_summary_v1_2_2.csv",
    "ghcn_snow_mapping": ARTIFACT_DIR / "ghcnd_snow_mapping_v1_2_2.csv",
    "ghcn_snwd_mapping": ARTIFACT_DIR / "ghcnd_snwd_mapping_v1_2_2.csv",
    "ghcn_daily_features": ARTIFACT_DIR / "ghcnd_daily_features_v1_2_2.parquet",
}
GHCN_RAW_DIR = ARTIFACT_DIR / "ghcnd_daily_raw_v1_2_2"

HOURLY_STATION_MAX_DISTANCE_KM = 60.0
GHCN_DAILY_MAX_DISTANCE_KM = 75.0
LOW_VISIBILITY_METERS = 4_828.0
LOW_CEILING_METERS = 305.0
FREEZING_TEMP_C = 2.0
GHCN_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCN_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
GHCN_DAILY_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all"
GHCN_START_DATE = pd.Timestamp("2022-12-20")
GHCN_END_DATE = pd.Timestamp("2025-12-31")

SOURCE_WEATHER_COLUMNS = [
    "airport",
    "weather_ts",
    "prev_weather_risk",
    "recent_prcp_3h",
    "recent_prcp_6h",
    "recent_snow_6h",
    "recent_wspd_6h_mean",
    "recent_wpgt_6h_max",
    "recent_bad_weather_6h_rate",
    "recent_weather_risk_6h",
    "recent_weather_risk_12h",
    "snow_missing",
    "wpgt_missing",
    "wspd_missing",
    "prcp_missing",
]

ORIGIN_TREND_COLUMNS = [
    "origin_recent_weather_risk_max_12h",
    "origin_weather_risk_delta_3h",
    "origin_weather_risk_delta_6h",
]

DEST_WEATHER_COLUMNS = [
    "dest_prev_weather_risk",
    "dest_recent_prcp_3h",
    "dest_recent_prcp_6h",
    "dest_recent_snow_6h",
    "dest_recent_wspd_6h_mean",
    "dest_recent_wpgt_6h_max",
    "dest_recent_bad_weather_6h_rate",
    "dest_recent_weather_risk_6h",
    "dest_recent_weather_risk_12h",
    "dest_snow_missing",
    "dest_wpgt_missing",
    "dest_wspd_missing",
    "dest_prcp_missing",
    "dest_weather_signal",
    "dest_recent_bad_weather_hours_6h",
    "dest_gust_gap_6h",
    "dest_recent_weather_risk_max_12h",
    "dest_weather_risk_delta_3h",
    "dest_weather_risk_delta_6h",
    "dest_weather_data_missing",
]

STATION_HOURLY_FEATURE_COLUMNS = [
    "recent_temp_3h_mean",
    "recent_temp_drop_3h",
    "freezing_flag",
    "metar_precip_now_flag",
    "freezing_precip_risk",
    "metar_snow_now_flag",
    "metar_snow_3h_count",
    "metar_snow_6h_count",
    "metar_freezing_precip_now_flag",
    "metar_freezing_precip_6h_count",
    "metar_low_vis_now_flag",
    "metar_low_vis_3h_count",
    "metar_low_ceiling_now_flag",
    "metar_low_ceiling_3h_count",
    "metar_thunder_now_flag",
    "metar_thunder_3h_count",
    "wind_volatility_6h",
    "recent_bad_weather_hours_12h",
]

ORIGIN_METAR_COLUMNS = [
    "origin_recent_temp_3h_mean",
    "origin_recent_temp_drop_3h",
    "origin_freezing_flag",
    "origin_metar_precip_now_flag",
    "origin_freezing_precip_risk",
    "origin_metar_snow_now_flag",
    "origin_metar_snow_3h_count",
    "origin_metar_snow_6h_count",
    "origin_metar_freezing_precip_now_flag",
    "origin_metar_freezing_precip_6h_count",
    "origin_metar_low_vis_now_flag",
    "origin_metar_low_vis_3h_count",
    "origin_metar_low_ceiling_now_flag",
    "origin_metar_low_ceiling_3h_count",
    "origin_metar_thunder_now_flag",
    "origin_metar_thunder_3h_count",
    "origin_wind_volatility_6h",
    "origin_recent_bad_weather_hours_12h",
    "origin_metar_data_missing",
]

DEST_METAR_COLUMNS = [
    "dest_recent_temp_3h_mean",
    "dest_recent_temp_drop_3h",
    "dest_freezing_flag",
    "dest_metar_precip_now_flag",
    "dest_freezing_precip_risk",
    "dest_metar_snow_now_flag",
    "dest_metar_snow_3h_count",
    "dest_metar_snow_6h_count",
    "dest_metar_freezing_precip_now_flag",
    "dest_metar_freezing_precip_6h_count",
    "dest_metar_low_vis_now_flag",
    "dest_metar_low_vis_3h_count",
    "dest_metar_low_ceiling_now_flag",
    "dest_metar_low_ceiling_3h_count",
    "dest_metar_thunder_now_flag",
    "dest_metar_thunder_3h_count",
    "dest_wind_volatility_6h",
    "dest_recent_bad_weather_hours_12h",
    "dest_metar_data_missing",
]

STATION_DAILY_COLUMNS = [
    "snowfall_today",
    "snowfall_3d_sum",
    "snow_depth_today",
]

ORIGIN_DAILY_COLUMNS = [
    "origin_snowfall_today",
    "origin_snowfall_3d_sum",
    "origin_snow_depth_today",
    "origin_snow_data_missing",
]

DEST_DAILY_COLUMNS = [
    "dest_snowfall_today",
    "dest_snowfall_3d_sum",
    "dest_snow_depth_today",
    "dest_snow_data_missing",
]

VALIDATION_COLUMNS = [
    "Year",
    "FlightDate",
    "Month",
    "Origin",
    "Dest",
    "depdelay15",
    "origin_recent_weather_risk_max_12h",
    "origin_recent_temp_3h_mean",
    "origin_metar_snow_6h_count",
    "origin_snowfall_today",
    "origin_snow_depth_today",
    "dest_recent_weather_risk_12h",
    "dest_metar_low_ceiling_now_flag",
    "route_weather_bottleneck",
    "origin_weather_signal_zscore_vs_airport_month",
]

NUMERIC_DEFAULTS = {
    "origin_recent_weather_risk_max_12h": 0.0,
    "origin_weather_risk_delta_3h": 0.0,
    "origin_weather_risk_delta_6h": 0.0,
    "dest_prev_weather_risk": 0.0,
    "dest_recent_prcp_3h": 0.0,
    "dest_recent_prcp_6h": 0.0,
    "dest_recent_snow_6h": 0.0,
    "dest_recent_wspd_6h_mean": 0.0,
    "dest_recent_wpgt_6h_max": 0.0,
    "dest_recent_bad_weather_6h_rate": 0.0,
    "dest_recent_weather_risk_6h": 0.0,
    "dest_recent_weather_risk_12h": 0.0,
    "dest_weather_signal": 0.0,
    "dest_recent_bad_weather_hours_6h": 0.0,
    "dest_gust_gap_6h": 0.0,
    "dest_recent_weather_risk_max_12h": 0.0,
    "dest_weather_risk_delta_3h": 0.0,
    "dest_weather_risk_delta_6h": 0.0,
    "origin_recent_temp_3h_mean": 0.0,
    "origin_recent_temp_drop_3h": 0.0,
    "origin_metar_snow_3h_count": 0.0,
    "origin_metar_snow_6h_count": 0.0,
    "origin_metar_freezing_precip_6h_count": 0.0,
    "origin_metar_low_vis_3h_count": 0.0,
    "origin_metar_low_ceiling_3h_count": 0.0,
    "origin_metar_thunder_3h_count": 0.0,
    "origin_wind_volatility_6h": 0.0,
    "origin_recent_bad_weather_hours_12h": 0.0,
    "dest_recent_temp_3h_mean": 0.0,
    "dest_recent_temp_drop_3h": 0.0,
    "dest_metar_snow_3h_count": 0.0,
    "dest_metar_snow_6h_count": 0.0,
    "dest_metar_freezing_precip_6h_count": 0.0,
    "dest_metar_low_vis_3h_count": 0.0,
    "dest_metar_low_ceiling_3h_count": 0.0,
    "dest_metar_thunder_3h_count": 0.0,
    "dest_wind_volatility_6h": 0.0,
    "dest_recent_bad_weather_hours_12h": 0.0,
    "origin_snowfall_today": 0.0,
    "origin_snowfall_3d_sum": 0.0,
    "origin_snow_depth_today": 0.0,
    "dest_snowfall_today": 0.0,
    "dest_snowfall_3d_sum": 0.0,
    "dest_snow_depth_today": 0.0,
    "origin_recent_bad_weather_hours_6h": 0.0,
    "origin_gust_gap_6h": 0.0,
    "route_weather_bottleneck": 0.0,
    "origin_dest_weather_gap": 0.0,
    "origin_dest_temp_gap": 0.0,
    "route_snow_bottleneck": 0.0,
    "origin_dest_freezing_pair_flag": 0.0,
    "origin_weather_signal_zscore_vs_airport_month": 0.0,
    "origin_wspd_vs_airport_month_p90": 0.0,
    "origin_prcp_vs_airport_month_avg": 0.0,
}

FLAG_DEFAULTS = {
    "dest_snow_missing": 1,
    "dest_wpgt_missing": 1,
    "dest_wspd_missing": 1,
    "dest_prcp_missing": 1,
    "dest_weather_data_missing": 1,
    "origin_freezing_flag": 0,
    "origin_metar_precip_now_flag": 0,
    "origin_freezing_precip_risk": 0,
    "origin_metar_snow_now_flag": 0,
    "origin_metar_freezing_precip_now_flag": 0,
    "origin_metar_low_vis_now_flag": 0,
    "origin_metar_low_ceiling_now_flag": 0,
    "origin_metar_thunder_now_flag": 0,
    "origin_metar_data_missing": 1,
    "dest_freezing_flag": 0,
    "dest_metar_precip_now_flag": 0,
    "dest_freezing_precip_risk": 0,
    "dest_metar_snow_now_flag": 0,
    "dest_metar_freezing_precip_now_flag": 0,
    "dest_metar_low_vis_now_flag": 0,
    "dest_metar_low_ceiling_now_flag": 0,
    "dest_metar_thunder_now_flag": 0,
    "dest_metar_data_missing": 1,
    "origin_snow_data_missing": 1,
    "dest_snow_data_missing": 1,
}

SNOW_PATTERN = re.compile(r"(?<!T)(?:\+|-|VC)?(?:SH|DR|BL|FZ)?(?:SN|SG|PL|GS|IC)(?!O)")
PRECIP_PATTERN = re.compile(r"(?:\+|-|VC)?(?:SH|TS|FZ)?(?:RA|DZ|SN|SG|PL|GS|IC|UP)")
FREEZING_PATTERN = re.compile(r"(?:FZRA|FZDZ|FZFG|PL|IC)")
THUNDER_PATTERN = re.compile(r"(?:\bTS\b|VCTS|TSRA|TSSN|LTG)")
SNOWFALL_PATTERN = re.compile(r"24 HR SNOWFALL \(IN\):\s*([0-9.]+)")
SNOW_DEPTH_PATTERN = re.compile(r"SNOW DEPTH \(IN\):\s*([0-9.]+)")


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype("int32")
    hours = (values // 100).clip(lower=0, upper=23)
    minutes = (values % 100).clip(lower=0, upper=59)
    return (hours * 60 + minutes).astype("int32")


def normalize_station_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def add_scheduled_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flight_date = pd.to_datetime(out["FlightDate"], errors="coerce").dt.normalize()
    dep_minutes = hhmm_to_minutes(out["CRSDepTime"])
    elapsed_minutes = pd.to_numeric(out["CRSElapsedTime"], errors="coerce").fillna(0).clip(lower=0)

    out["scheduled_dep_ts"] = flight_date + pd.to_timedelta(dep_minutes, unit="m")
    out["scheduled_arr_ts"] = out["scheduled_dep_ts"] + pd.to_timedelta(elapsed_minutes, unit="m")
    out["scheduled_arr_date"] = out["scheduled_arr_ts"].dt.normalize()
    out["FlightDate"] = flight_date
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def parse_compact_signed_tenth(value: Any) -> float | None:
    if pd.isna(value):
        return None
    token = str(value).split(",")[0].strip()
    if not token:
        return None
    if token in {"9999", "+9999", "-9999"}:
        return None
    try:
        return float(int(token)) / 10.0
    except ValueError:
        return None


def parse_wnd_speed(value: Any) -> float | None:
    if pd.isna(value):
        return None
    parts = str(value).split(",")
    if len(parts) < 4:
        return None
    token = parts[3].strip()
    if token in {"9999", ""}:
        return None
    try:
        return float(int(token)) / 10.0
    except ValueError:
        return None


def parse_distance_token(value: Any, missing_token: str) -> float | None:
    if pd.isna(value):
        return None
    token = str(value).split(",")[0].strip()
    if token in {missing_token, ""}:
        return None
    try:
        return float(int(token))
    except ValueError:
        return None


def clean_metar_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper()
    return re.sub(r"\s+", " ", text).strip()


def has_regex_match(pattern: re.Pattern[str], text: str) -> int:
    if not text:
        return 0
    return int(bool(pattern.search(text)))


def parse_snowfall_inches(text: str) -> float | None:
    if not text:
        return None
    match = SNOWFALL_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_snow_depth_inches(text: str) -> float | None:
    if not text:
        return None
    match = SNOW_DEPTH_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def compute_origin_weather_signal(prev_weather_risk: pd.Series, risk_6h: pd.Series, risk_12h: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(prev_weather_risk, errors="coerce").fillna(0)
        + 0.7 * pd.to_numeric(risk_6h, errors="coerce").fillna(0)
        + 0.4 * pd.to_numeric(risk_12h, errors="coerce").fillna(0)
    ).astype("float32")


def safe_ratio(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
    den_num = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    result = pd.to_numeric(num, errors="coerce") / den_num
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value).astype("float32")


def load_airport_metadata(airport_codes: set[str]) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not AIRPORTS_CSV.exists():
        raise FileNotFoundError(f"Airport coordinate file not found: {AIRPORTS_CSV}")

    airports = pd.read_csv(
        AIRPORTS_CSV,
        usecols=["iata_code", "latitude_deg", "longitude_deg", "name", "municipality", "iso_region"],
    )
    airports["iata_code"] = airports["iata_code"].astype("string").str.upper()
    airports = airports.dropna(subset=["iata_code", "latitude_deg", "longitude_deg"]).drop_duplicates("iata_code")
    airports = airports[airports["iata_code"].isin(sorted(airport_codes))].copy()

    tf = TimezoneFinder()
    airports["timezone_name"] = airports.apply(
        lambda row: tf.timezone_at(lng=float(row["longitude_deg"]), lat=float(row["latitude_deg"])) or "UTC",
        axis=1,
    )
    airports.to_csv(AUX_FILES["airport_metadata"], index=False)
    return airports


def build_noaa_station_inventory(force_rebuild: bool = False) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = AUX_FILES["station_inventory"]
    if target.exists() and not force_rebuild:
        cached = pd.read_csv(
            target,
            dtype={
                "station_id": "string",
                "csv_path": "string",
                "call_sign": "string",
                "name": "string",
                "timezone_name": "string",
            },
        )
        cached["station_id"] = cached["station_id"].map(normalize_station_id).astype("string")
        cached["lat"] = pd.to_numeric(cached["lat"], errors="coerce")
        cached["lon"] = pd.to_numeric(cached["lon"], errors="coerce")
        return cached.dropna(subset=["station_id", "lat", "lon"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    tf = TimezoneFinder()
    for csv_path in sorted(NOAA_DATA_DIR.glob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, usecols=["LATITUDE", "LONGITUDE", "CALL_SIGN", "NAME"], nrows=1)
        except Exception:
            continue
        if sample.empty:
            continue
        lat = float(sample["LATITUDE"].iloc[0])
        lon = float(sample["LONGITUDE"].iloc[0])
        rows.append(
            {
                "station_id": normalize_station_id(csv_path.stem),
                "csv_path": str(csv_path),
                "lat": lat,
                "lon": lon,
                "call_sign": str(sample["CALL_SIGN"].iloc[0]).strip(),
                "name": str(sample["NAME"].iloc[0]).strip(),
                "timezone_name": tf.timezone_at(lng=lon, lat=lat) or "UTC",
            }
        )

    inventory = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    if inventory.empty:
        raise RuntimeError(f"No NOAA station files were discovered under {NOAA_DATA_DIR}")
    inventory["station_id"] = inventory["station_id"].map(normalize_station_id).astype("string")
    inventory.to_csv(target, index=False)
    return inventory


def map_airports_to_points(
    airports: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_id_col: str,
    max_distance_km: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_rows = list(candidates.itertuples(index=False))
    for airport in airports.itertuples(index=False):
        airport_code = getattr(airport, "iata_code", None)
        if airport_code is None:
            airport_code = getattr(airport, "airport")
        best_id = None
        best_distance = math.inf
        for candidate in candidate_rows:
            distance_km = haversine_km(
                float(airport.latitude_deg),
                float(airport.longitude_deg),
                float(getattr(candidate, "lat")),
                float(getattr(candidate, "lon")),
            )
            if distance_km < best_distance:
                best_distance = distance_km
                best_id = getattr(candidate, candidate_id_col)
        rows.append(
            {
                "airport": airport_code,
                candidate_id_col: normalize_station_id(best_id) if best_distance <= max_distance_km else None,
                "distance_km": None if best_distance == math.inf else round(best_distance, 3),
                "has_mapping": int(best_distance <= max_distance_km),
            }
        )
    return pd.DataFrame(rows)


def build_enhanced_weather_history(force_rebuild: bool = False) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = AUX_FILES["enhanced_weather_history"]
    if target.exists() and not force_rebuild:
        return pd.read_parquet(target)

    if not SOURCE_WEATHER_HISTORY.exists():
        raise FileNotFoundError(
            f"Missing source weather history: {SOURCE_WEATHER_HISTORY}\n"
            "Run enterprise_delay_ensemble_v1_2_1\\build_feature_store_v1_2_1.py first."
        )

    weather = pd.read_parquet(SOURCE_WEATHER_HISTORY, columns=SOURCE_WEATHER_COLUMNS)
    weather["airport"] = weather["airport"].astype("string").str.upper()
    weather["weather_ts"] = pd.to_datetime(weather["weather_ts"], errors="coerce")
    weather = weather.sort_values(["airport", "weather_ts"], kind="mergesort").reset_index(drop=True)

    grp = weather.groupby("airport", observed=True)
    weather["recent_weather_risk_max_12h"] = (
        grp["recent_weather_risk_6h"].transform(lambda s: s.rolling(12, min_periods=1).max()).fillna(0).astype("float32")
    )
    weather["weather_risk_delta_3h"] = (
        pd.to_numeric(weather["recent_weather_risk_6h"], errors="coerce").fillna(0)
        - grp["recent_weather_risk_6h"].shift(3).fillna(0)
    ).astype("float32")
    weather["weather_risk_delta_6h"] = (
        pd.to_numeric(weather["recent_weather_risk_6h"], errors="coerce").fillna(0)
        - grp["recent_weather_risk_6h"].shift(6).fillna(0)
    ).astype("float32")

    weather.to_parquet(target, index=False)
    return weather


def build_metar_hourly_and_daily_features(
    station_inventory: pd.DataFrame,
    hourly_station_ids: set[str],
    daily_station_ids: set[str],
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    hourly_target = AUX_FILES["metar_hourly"]
    daily_target = AUX_FILES["station_daily_snow"]
    if hourly_target.exists() and daily_target.exists() and not force_rebuild:
        return pd.read_parquet(hourly_target), pd.read_parquet(daily_target)

    selected_station_ids = sorted(
        {
            station_id
            for station_id in (
                normalize_station_id(value) for value in (hourly_station_ids | daily_station_ids)
            )
            if station_id
        }
    )
    inventory = station_inventory.copy()
    inventory["station_id"] = inventory["station_id"].map(normalize_station_id).astype("string")
    inventory = inventory[inventory["station_id"].isin(selected_station_ids)].copy()
    inventory_map = {row.station_id: row for row in inventory.itertuples(index=False)}

    hourly_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []

    for station_id in selected_station_ids:
        station = inventory_map.get(station_id)
        if station is None:
            continue
        path = Path(station.csv_path)
        try:
            raw = pd.read_csv(
                path,
                usecols=["DATE", "REPORT_TYPE", "TMP", "WND", "VIS", "CIG", "REM"],
                low_memory=False,
            )
        except Exception:
            continue

        raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
        raw = raw.dropna(subset=["DATE"]).copy()
        if raw.empty:
            continue

        timezone_name = station.timezone_name if isinstance(station.timezone_name, str) and station.timezone_name else "UTC"
        local_ts = raw["DATE"].dt.tz_convert(timezone_name).dt.tz_localize(None)
        raw["local_ts"] = local_ts

        hourly_mask = ~raw["REPORT_TYPE"].fillna("").astype("string").str.startswith("SOD")
        if station_id in hourly_station_ids:
            hourly = raw.loc[hourly_mask, ["local_ts", "TMP", "WND", "VIS", "CIG", "REM"]].copy()
            hourly = hourly.drop_duplicates(subset=["local_ts"], keep="last").sort_values("local_ts")
            if not hourly.empty:
                metar_text = hourly["REM"].map(clean_metar_text)
                hourly["temp_c"] = hourly["TMP"].map(parse_compact_signed_tenth)
                hourly["wind_speed_mps"] = hourly["WND"].map(parse_wnd_speed)
                hourly["visibility_m"] = hourly["VIS"].map(lambda v: parse_distance_token(v, "999999"))
                hourly["ceiling_m"] = hourly["CIG"].map(lambda v: parse_distance_token(v, "99999"))
                hourly["metar_snow_now_flag"] = metar_text.map(lambda text: has_regex_match(SNOW_PATTERN, text)).astype("int8")
                hourly["metar_precip_now_flag"] = metar_text.map(lambda text: has_regex_match(PRECIP_PATTERN, text)).astype("int8")
                hourly["metar_freezing_precip_now_flag"] = metar_text.map(lambda text: has_regex_match(FREEZING_PATTERN, text)).astype("int8")
                hourly["metar_thunder_now_flag"] = metar_text.map(lambda text: has_regex_match(THUNDER_PATTERN, text)).astype("int8")
                hourly["metar_low_vis_now_flag"] = (
                    pd.to_numeric(hourly["visibility_m"], errors="coerce").fillna(np.inf) <= LOW_VISIBILITY_METERS
                ).astype("int8")
                hourly["metar_low_ceiling_now_flag"] = (
                    pd.to_numeric(hourly["ceiling_m"], errors="coerce").fillna(np.inf) <= LOW_CEILING_METERS
                ).astype("int8")
                hourly["freezing_flag"] = (
                    pd.to_numeric(hourly["temp_c"], errors="coerce").fillna(np.inf) <= FREEZING_TEMP_C
                ).astype("int8")
                hourly["freezing_precip_risk"] = (
                    ((hourly["freezing_flag"] == 1) & (hourly["metar_precip_now_flag"] == 1))
                    | (hourly["metar_freezing_precip_now_flag"] == 1)
                ).astype("int8")
                hourly["metar_adverse_now_flag"] = (
                    hourly[
                        [
                            "metar_snow_now_flag",
                            "metar_freezing_precip_now_flag",
                            "metar_low_vis_now_flag",
                            "metar_low_ceiling_now_flag",
                            "metar_thunder_now_flag",
                        ]
                    ]
                    .max(axis=1)
                    .astype("int8")
                )

                hourly = hourly.set_index("local_ts")
                hourly["recent_temp_3h_mean"] = (
                    pd.to_numeric(hourly["temp_c"], errors="coerce").rolling("3h", min_periods=1).mean().fillna(0).astype("float32")
                )
                rolling_temp_max = pd.to_numeric(hourly["temp_c"], errors="coerce").rolling("3h", min_periods=1).max()
                hourly["recent_temp_drop_3h"] = (
                    rolling_temp_max - pd.to_numeric(hourly["temp_c"], errors="coerce")
                ).clip(lower=0).fillna(0).astype("float32")
                hourly["wind_volatility_6h"] = (
                    pd.to_numeric(hourly["wind_speed_mps"], errors="coerce")
                    .rolling("6h", min_periods=2)
                    .std()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["metar_snow_3h_count"] = (
                    pd.to_numeric(hourly["metar_snow_now_flag"], errors="coerce")
                    .rolling("3h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["metar_snow_6h_count"] = (
                    pd.to_numeric(hourly["metar_snow_now_flag"], errors="coerce")
                    .rolling("6h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["metar_freezing_precip_6h_count"] = (
                    pd.to_numeric(hourly["metar_freezing_precip_now_flag"], errors="coerce")
                    .rolling("6h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["metar_low_vis_3h_count"] = (
                    pd.to_numeric(hourly["metar_low_vis_now_flag"], errors="coerce")
                    .rolling("3h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["metar_low_ceiling_3h_count"] = (
                    pd.to_numeric(hourly["metar_low_ceiling_now_flag"], errors="coerce")
                    .rolling("3h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["metar_thunder_3h_count"] = (
                    pd.to_numeric(hourly["metar_thunder_now_flag"], errors="coerce")
                    .rolling("3h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly["recent_bad_weather_hours_12h"] = (
                    pd.to_numeric(hourly["metar_adverse_now_flag"], errors="coerce")
                    .rolling("12h", min_periods=1)
                    .sum()
                    .fillna(0)
                    .astype("float32")
                )
                hourly = hourly.reset_index().rename(columns={"local_ts": "weather_ts"})
                hourly["station_id"] = station_id
                hourly_frames.append(
                    hourly[
                        ["station_id", "weather_ts", *STATION_HOURLY_FEATURE_COLUMNS]
                    ].copy()
                )

        if station_id in daily_station_ids:
            sod = raw.loc[~hourly_mask, ["local_ts", "REM"]].copy()
            if not sod.empty:
                sod["weather_date"] = pd.to_datetime(sod["local_ts"], errors="coerce").dt.normalize()
                sod["metar_text"] = sod["REM"].map(clean_metar_text)
                sod["snowfall_today"] = sod["metar_text"].map(parse_snowfall_inches)
                sod["snow_depth_today"] = sod["metar_text"].map(parse_snow_depth_inches)
                sod = (
                    sod.sort_values(["weather_date", "local_ts"])
                    .drop_duplicates(subset=["weather_date"], keep="last")
                    .reset_index(drop=True)
                )
                sod["snowfall_today"] = (
                    pd.to_numeric(sod["snowfall_today"], errors="coerce").fillna(0) * 25.4
                ).astype("float32")
                sod["snow_depth_today"] = (
                    pd.to_numeric(sod["snow_depth_today"], errors="coerce").fillna(0) * 25.4
                ).astype("float32")
                sod["snowfall_3d_sum"] = (
                    sod["snowfall_today"].rolling(3, min_periods=1).sum().fillna(0).astype("float32")
                )
                sod["station_id"] = station_id
                daily_frames.append(
                    sod[["station_id", "weather_date", *STATION_DAILY_COLUMNS]].copy()
                )

        del raw
        gc.collect()

    hourly_df = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame(columns=["station_id", "weather_ts", *STATION_HOURLY_FEATURE_COLUMNS])
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=["station_id", "weather_date", *STATION_DAILY_COLUMNS])

    hourly_df.to_parquet(hourly_target, index=False)
    daily_df.to_parquet(daily_target, index=False)
    return hourly_df, daily_df


def download_file_if_needed(url: str, target: Path, force_rebuild: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force_rebuild:
        return target
    urlretrieve(url, target)
    return target


def build_ghcn_station_summary(force_rebuild: bool = False) -> pd.DataFrame:
    target = AUX_FILES["ghcn_station_summary"]
    if target.exists() and not force_rebuild:
        cached = pd.read_csv(
            target,
            dtype={
                "station_id": "string",
                "state": "string",
                "name": "string",
            },
        )
        cached["lat"] = pd.to_numeric(cached["lat"], errors="coerce")
        cached["lon"] = pd.to_numeric(cached["lon"], errors="coerce")
        cached["has_snow"] = pd.to_numeric(cached["has_snow"], errors="coerce").fillna(0).astype("int8")
        cached["has_snwd"] = pd.to_numeric(cached["has_snwd"], errors="coerce").fillna(0).astype("int8")
        cached["first_year"] = pd.to_numeric(cached["first_year"], errors="coerce").fillna(0).astype("int16")
        cached["last_year"] = pd.to_numeric(cached["last_year"], errors="coerce").fillna(0).astype("int16")
        return cached.dropna(subset=["station_id", "lat", "lon"]).reset_index(drop=True)

    stations_path = download_file_if_needed(GHCN_STATIONS_URL, AUX_FILES["ghcn_stations_text"], force_rebuild=force_rebuild)
    inventory_path = download_file_if_needed(GHCN_INVENTORY_URL, AUX_FILES["ghcn_inventory_text"], force_rebuild=force_rebuild)

    stations = pd.read_fwf(
        stations_path,
        colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76, 79), (80, 85)],
        names=["station_id", "lat", "lon", "elev", "state", "name", "gsn_flag", "hcn_flag", "wmo_id"],
        dtype={"station_id": "string", "state": "string", "name": "string"},
    )
    inventory = pd.read_fwf(
        inventory_path,
        colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
        names=["station_id", "lat", "lon", "element", "first_year", "last_year"],
        dtype={"station_id": "string", "element": "string"},
    )
    inventory["station_id"] = inventory["station_id"].map(normalize_station_id).astype("string")
    inventory["element"] = inventory["element"].astype("string").str.upper()
    inventory = inventory[
        inventory["element"].isin(["SNOW", "SNWD"])
        & (pd.to_numeric(inventory["last_year"], errors="coerce").fillna(0) >= 2025)
    ].copy()
    if inventory.empty:
        raise RuntimeError("GHCN inventory did not return any candidate SNOW/SNWD stations through 2025.")

    summary = (
        inventory.groupby("station_id", observed=True)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            has_snow=("element", lambda s: int("SNOW" in set(s.astype(str)))),
            has_snwd=("element", lambda s: int("SNWD" in set(s.astype(str)))),
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
        )
        .reset_index()
    )
    stations["station_id"] = stations["station_id"].map(normalize_station_id).astype("string")
    summary = summary.merge(stations[["station_id", "state", "name"]], how="left", on="station_id")
    summary["lat"] = pd.to_numeric(summary["lat"], errors="coerce")
    summary["lon"] = pd.to_numeric(summary["lon"], errors="coerce")
    summary["first_year"] = pd.to_numeric(summary["first_year"], errors="coerce").fillna(0).astype("int16")
    summary["last_year"] = pd.to_numeric(summary["last_year"], errors="coerce").fillna(0).astype("int16")
    summary["has_snow"] = pd.to_numeric(summary["has_snow"], errors="coerce").fillna(0).astype("int8")
    summary["has_snwd"] = pd.to_numeric(summary["has_snwd"], errors="coerce").fillna(0).astype("int8")
    summary = summary.dropna(subset=["station_id", "lat", "lon"]).reset_index(drop=True)
    summary.to_csv(target, index=False)
    return summary


def download_ghcn_station_file(station_id: str, force_rebuild: bool = False) -> Path | None:
    station_id = normalize_station_id(station_id)
    if not station_id:
        return None
    target = GHCN_RAW_DIR / f"{station_id}.dly"
    if target.exists() and not force_rebuild:
        return target
    GHCN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(f"{GHCN_DAILY_BASE_URL}/{station_id}.dly", target)
    except Exception:
        return None
    return target


def parse_ghcn_dly_file(path: Path, station_id: str) -> pd.DataFrame:
    records: dict[pd.Timestamp, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if len(line) < 269:
                continue
            element = line[17:21]
            if element not in {"SNOW", "SNWD"}:
                continue
            try:
                year = int(line[11:15])
                month = int(line[15:17])
            except ValueError:
                continue
            for day in range(1, 32):
                offset = 21 + (day - 1) * 8
                value_text = line[offset : offset + 5]
                qflag = line[offset + 6 : offset + 7]
                try:
                    weather_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                if weather_date < GHCN_START_DATE or weather_date > GHCN_END_DATE:
                    continue
                if qflag.strip():
                    continue
                try:
                    value = int(value_text)
                except ValueError:
                    continue
                if value == -9999:
                    continue
                row = records.setdefault(weather_date, {"station_id": station_id, "weather_date": weather_date})
                if element == "SNOW":
                    row["snowfall_today"] = float(value)
                else:
                    row["snow_depth_today"] = float(value)

    if not records:
        return pd.DataFrame(columns=["station_id", "weather_date", *STATION_DAILY_COLUMNS])

    daily = pd.DataFrame(records.values()).sort_values("weather_date").reset_index(drop=True)
    daily["snowfall_today"] = pd.to_numeric(daily.get("snowfall_today"), errors="coerce").astype("float32")
    daily["snow_depth_today"] = pd.to_numeric(daily.get("snow_depth_today"), errors="coerce").astype("float32")
    daily["snowfall_3d_sum"] = (
        pd.to_numeric(daily["snowfall_today"], errors="coerce").fillna(0).rolling(3, min_periods=1).sum().astype("float32")
    )
    return daily[["station_id", "weather_date", *STATION_DAILY_COLUMNS]]


def build_ghcn_daily_features(station_ids: set[str], force_rebuild: bool = False) -> pd.DataFrame:
    target = AUX_FILES["ghcn_daily_features"]
    if target.exists() and not force_rebuild:
        return pd.read_parquet(target)

    selected_station_ids = sorted({station_id for station_id in (normalize_station_id(value) for value in station_ids) if station_id})
    if not selected_station_ids:
        empty = pd.DataFrame(columns=["station_id", "weather_date", *STATION_DAILY_COLUMNS])
        empty.to_parquet(target, index=False)
        return empty

    downloaded_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(download_ghcn_station_file, station_id, force_rebuild): station_id
            for station_id in selected_station_ids
        }
        for future in as_completed(future_map):
            path = future.result()
            if path is not None:
                downloaded_paths.append(path)

    daily_frames: list[pd.DataFrame] = []
    for path in sorted(downloaded_paths):
        parsed = parse_ghcn_dly_file(path, station_id=path.stem)
        if not parsed.empty:
            daily_frames.append(parsed)

    daily = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames
        else pd.DataFrame(columns=["station_id", "weather_date", *STATION_DAILY_COLUMNS])
    )
    daily.to_parquet(target, index=False)
    return daily


def prefix_columns(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in exclude}
    return df.rename(columns=rename_map)


def merge_asof_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_by: str,
    right_by: str,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return left

    left_work = left.copy()
    right_work = right.copy()
    left_work[left_by] = left_work[left_by].astype("string")
    right_work[right_by] = right_work[right_by].astype("string")

    left_valid = left_work[left_work[left_on].notna()].copy()
    left_missing = left_work[left_work[left_on].isna()].copy()
    if left_valid.empty:
        return left
    right_work = right_work[right_work[right_on].notna()].copy()
    if right_work.empty:
        return left

    merged = pd.merge_asof(
        left_valid.sort_values([left_on, left_by], kind="mergesort"),
        right_work.sort_values([right_on, right_by], kind="mergesort"),
        left_on=left_on,
        right_on=right_on,
        left_by=left_by,
        right_by=right_by,
        direction="backward",
        allow_exact_matches=True,
    )
    if not left_missing.empty:
        new_cols = [col for col in merged.columns if col not in left.columns]
        for column in new_cols:
            left_missing[column] = np.nan
        merged = pd.concat([merged, left_missing[merged.columns]], axis=0)
    return merged.sort_index()


def build_origin_airport_month_norms(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train_df.groupby(["Origin", "Month"], observed=True)
        .agg(
            origin_weather_signal_month_mean=("origin_weather_signal", "mean"),
            origin_weather_signal_month_std=("origin_weather_signal", lambda s: float(np.nanstd(pd.to_numeric(s, errors="coerce")))),
            origin_wspd_month_p90=("origin_recent_wspd_6h_mean", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
            origin_prcp_month_mean=("origin_recent_prcp_6h", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
    )
    grouped["origin_weather_signal_month_std"] = grouped["origin_weather_signal_month_std"].replace(0, np.nan).fillna(1.0)
    grouped["origin_wspd_month_p90"] = grouped["origin_wspd_month_p90"].replace(0, np.nan).fillna(1.0)
    grouped["origin_prcp_month_mean"] = grouped["origin_prcp_month_mean"].replace(0, np.nan).fillna(1.0)
    grouped.to_csv(AUX_FILES["origin_airport_month_norms"], index=False)
    return grouped


def apply_origin_airport_month_norms(df: pd.DataFrame, norms: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(norms, how="left", on=["Origin", "Month"])
    out["origin_weather_signal_month_mean"] = pd.to_numeric(out["origin_weather_signal_month_mean"], errors="coerce").fillna(0.0)
    out["origin_weather_signal_month_std"] = pd.to_numeric(out["origin_weather_signal_month_std"], errors="coerce").replace(0, np.nan).fillna(1.0)
    out["origin_wspd_month_p90"] = pd.to_numeric(out["origin_wspd_month_p90"], errors="coerce").replace(0, np.nan).fillna(1.0)
    out["origin_prcp_month_mean"] = pd.to_numeric(out["origin_prcp_month_mean"], errors="coerce").replace(0, np.nan).fillna(1.0)
    out["origin_weather_signal_zscore_vs_airport_month"] = (
        (pd.to_numeric(out["origin_weather_signal"], errors="coerce").fillna(0) - out["origin_weather_signal_month_mean"])
        / out["origin_weather_signal_month_std"]
    ).fillna(0).astype("float32")
    out["origin_wspd_vs_airport_month_p90"] = safe_ratio(out["origin_recent_wspd_6h_mean"], out["origin_wspd_month_p90"])
    out["origin_prcp_vs_airport_month_avg"] = safe_ratio(out["origin_recent_prcp_6h"], out["origin_prcp_month_mean"])
    return out.drop(
        columns=[
            "origin_weather_signal_month_mean",
            "origin_weather_signal_month_std",
            "origin_wspd_month_p90",
            "origin_prcp_month_mean",
        ],
        errors="ignore",
    )


def fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column, default in NUMERIC_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default).astype("float32")
    for column, default in FLAG_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default).astype("int8")
    return out


def augment_split(
    base_df: pd.DataFrame,
    enhanced_weather_history: pd.DataFrame,
    metar_hourly: pd.DataFrame,
    ghcn_daily: pd.DataFrame,
    hourly_mapping: pd.DataFrame,
    snowfall_mapping: pd.DataFrame,
    snowdepth_mapping: pd.DataFrame,
) -> pd.DataFrame:
    out = add_scheduled_timestamps(base_df)
    out["Origin"] = out["Origin"].astype("string").str.upper()
    out["Dest"] = out["Dest"].astype("string").str.upper()
    out["Month"] = pd.to_numeric(out["Month"], errors="coerce").fillna(0).astype("int16")

    weather_trend = enhanced_weather_history[
        ["airport", "weather_ts", "recent_weather_risk_max_12h", "weather_risk_delta_3h", "weather_risk_delta_6h"]
    ].copy()
    weather_trend = prefix_columns(weather_trend, "origin_", exclude={"airport", "weather_ts"}).rename(
        columns={"weather_ts": "origin_weather_ts_lookup"}
    )
    weather_trend["airport"] = weather_trend["airport"].astype("string").str.upper()
    out = merge_asof_features(
        left=out.rename(columns={"Origin": "_origin_key"}),
        right=weather_trend.rename(columns={"airport": "_origin_key"}),
        left_on="scheduled_dep_ts",
        right_on="origin_weather_ts_lookup",
        left_by="_origin_key",
        right_by="_origin_key",
    ).rename(columns={"_origin_key": "Origin"})

    dest_weather = enhanced_weather_history[
        [
            "airport",
            "weather_ts",
            "prev_weather_risk",
            "recent_prcp_3h",
            "recent_prcp_6h",
            "recent_snow_6h",
            "recent_wspd_6h_mean",
            "recent_wpgt_6h_max",
            "recent_bad_weather_6h_rate",
            "recent_weather_risk_6h",
            "recent_weather_risk_12h",
            "snow_missing",
            "wpgt_missing",
            "wspd_missing",
            "prcp_missing",
            "recent_weather_risk_max_12h",
            "weather_risk_delta_3h",
            "weather_risk_delta_6h",
        ]
    ].copy()
    dest_weather = prefix_columns(dest_weather, "dest_", exclude={"airport", "weather_ts"}).rename(
        columns={"weather_ts": "dest_weather_ts_lookup"}
    )
    dest_weather["airport"] = dest_weather["airport"].astype("string").str.upper()
    out = merge_asof_features(
        left=out.rename(columns={"Dest": "_dest_key"}),
        right=dest_weather.rename(columns={"airport": "_dest_key"}),
        left_on="scheduled_arr_ts",
        right_on="dest_weather_ts_lookup",
        left_by="_dest_key",
        right_by="_dest_key",
    ).rename(columns={"_dest_key": "Dest"})

    out["dest_weather_signal"] = compute_origin_weather_signal(
        out["dest_prev_weather_risk"],
        out["dest_recent_weather_risk_6h"],
        out["dest_recent_weather_risk_12h"],
    )
    out["dest_recent_bad_weather_hours_6h"] = (
        pd.to_numeric(out["dest_recent_bad_weather_6h_rate"], errors="coerce").fillna(0) * 6.0
    ).astype("float32")
    out["dest_gust_gap_6h"] = (
        pd.to_numeric(out["dest_recent_wpgt_6h_max"], errors="coerce").fillna(0)
        - pd.to_numeric(out["dest_recent_wspd_6h_mean"], errors="coerce").fillna(0)
    ).clip(lower=0).astype("float32")
    out["dest_weather_data_missing"] = (
        pd.to_numeric(out["dest_recent_weather_risk_6h"], errors="coerce").isna()
    ).astype("int8")

    out["origin_recent_bad_weather_hours_6h"] = (
        pd.to_numeric(out["origin_recent_bad_weather_6h_rate"], errors="coerce").fillna(0) * 6.0
    ).astype("float32")
    out["origin_gust_gap_6h"] = (
        pd.to_numeric(out["origin_recent_wpgt_6h_max"], errors="coerce").fillna(0)
        - pd.to_numeric(out["origin_recent_wspd_6h_mean"], errors="coerce").fillna(0)
    ).clip(lower=0).astype("float32")

    hourly_mapping = hourly_mapping.rename(
        columns={"airport": "mapped_airport", "station_id": "mapped_station_id", "has_mapping": "mapped_has_mapping"}
    )
    out = out.merge(
        hourly_mapping[["mapped_airport", "mapped_station_id", "mapped_has_mapping"]],
        how="left",
        left_on="Origin",
        right_on="mapped_airport",
    )
    out = out.rename(
        columns={
            "mapped_station_id": "origin_hourly_station_id",
            "mapped_has_mapping": "origin_hourly_has_station",
        }
    ).drop(columns=["mapped_airport"], errors="ignore")
    out = out.merge(
        hourly_mapping[["mapped_airport", "mapped_station_id", "mapped_has_mapping"]],
        how="left",
        left_on="Dest",
        right_on="mapped_airport",
    )
    out = out.rename(
        columns={
            "mapped_station_id": "dest_hourly_station_id",
            "mapped_has_mapping": "dest_hourly_has_station",
        }
    ).drop(columns=["mapped_airport"], errors="ignore")
    out["origin_hourly_station_id"] = out["origin_hourly_station_id"].astype("string").fillna("__NONE__")
    out["dest_hourly_station_id"] = out["dest_hourly_station_id"].astype("string").fillna("__NONE__")
    out["origin_hourly_has_station"] = pd.to_numeric(out["origin_hourly_has_station"], errors="coerce").fillna(0).astype("int8")
    out["dest_hourly_has_station"] = pd.to_numeric(out["dest_hourly_has_station"], errors="coerce").fillna(0).astype("int8")

    station_hourly = prefix_columns(metar_hourly.copy(), "origin_", exclude={"station_id", "weather_ts"}).rename(
        columns={
            "station_id": "origin_metar_station_lookup",
            "weather_ts": "origin_metar_weather_ts_lookup",
        }
    )
    out = merge_asof_features(
        left=out,
        right=station_hourly,
        left_on="scheduled_dep_ts",
        right_on="origin_metar_weather_ts_lookup",
        left_by="origin_hourly_station_id",
        right_by="origin_metar_station_lookup",
    )
    station_hourly_dest = prefix_columns(metar_hourly.copy(), "dest_", exclude={"station_id", "weather_ts"}).rename(
        columns={
            "station_id": "dest_metar_station_lookup",
            "weather_ts": "dest_metar_weather_ts_lookup",
        }
    )
    out = merge_asof_features(
        left=out,
        right=station_hourly_dest,
        left_on="scheduled_arr_ts",
        right_on="dest_metar_weather_ts_lookup",
        left_by="dest_hourly_station_id",
        right_by="dest_metar_station_lookup",
    )
    out["origin_metar_data_missing"] = (
        pd.to_numeric(out["origin_recent_temp_3h_mean"], errors="coerce").isna()
        | (pd.to_numeric(out["origin_hourly_has_station"], errors="coerce").fillna(0) == 0)
    ).astype("int8")
    out["dest_metar_data_missing"] = (
        pd.to_numeric(out["dest_recent_temp_3h_mean"], errors="coerce").isna()
        | (pd.to_numeric(out["dest_hourly_has_station"], errors="coerce").fillna(0) == 0)
    ).astype("int8")

    snowfall_mapping = snowfall_mapping.rename(
        columns={"airport": "mapped_airport", "station_id": "mapped_station_id", "has_mapping": "mapped_has_mapping"}
    )
    out = out.merge(
        snowfall_mapping[["mapped_airport", "mapped_station_id", "mapped_has_mapping"]],
        how="left",
        left_on="Origin",
        right_on="mapped_airport",
    )
    out = out.rename(
        columns={
            "mapped_station_id": "origin_snowfall_station_id",
            "mapped_has_mapping": "origin_snowfall_has_station",
        }
    ).drop(columns=["mapped_airport"], errors="ignore")
    out = out.merge(
        snowfall_mapping[["mapped_airport", "mapped_station_id", "mapped_has_mapping"]],
        how="left",
        left_on="Dest",
        right_on="mapped_airport",
    )
    out = out.rename(
        columns={
            "mapped_station_id": "dest_snowfall_station_id",
            "mapped_has_mapping": "dest_snowfall_has_station",
        }
    ).drop(columns=["mapped_airport"], errors="ignore")
    out["origin_snowfall_station_id"] = out["origin_snowfall_station_id"].astype("string").fillna("__NONE__")
    out["dest_snowfall_station_id"] = out["dest_snowfall_station_id"].astype("string").fillna("__NONE__")
    out["origin_snowfall_has_station"] = pd.to_numeric(out["origin_snowfall_has_station"], errors="coerce").fillna(0).astype("int8")
    out["dest_snowfall_has_station"] = pd.to_numeric(out["dest_snowfall_has_station"], errors="coerce").fillna(0).astype("int8")

    snowdepth_mapping = snowdepth_mapping.rename(
        columns={"airport": "mapped_airport", "station_id": "mapped_station_id", "has_mapping": "mapped_has_mapping"}
    )
    out = out.merge(
        snowdepth_mapping[["mapped_airport", "mapped_station_id", "mapped_has_mapping"]],
        how="left",
        left_on="Origin",
        right_on="mapped_airport",
    )
    out = out.rename(
        columns={
            "mapped_station_id": "origin_snowdepth_station_id",
            "mapped_has_mapping": "origin_snowdepth_has_station",
        }
    ).drop(columns=["mapped_airport"], errors="ignore")
    out = out.merge(
        snowdepth_mapping[["mapped_airport", "mapped_station_id", "mapped_has_mapping"]],
        how="left",
        left_on="Dest",
        right_on="mapped_airport",
    )
    out = out.rename(
        columns={
            "mapped_station_id": "dest_snowdepth_station_id",
            "mapped_has_mapping": "dest_snowdepth_has_station",
        }
    ).drop(columns=["mapped_airport"], errors="ignore")
    out["origin_snowdepth_station_id"] = out["origin_snowdepth_station_id"].astype("string").fillna("__NONE__")
    out["dest_snowdepth_station_id"] = out["dest_snowdepth_station_id"].astype("string").fillna("__NONE__")
    out["origin_snowdepth_has_station"] = pd.to_numeric(out["origin_snowdepth_has_station"], errors="coerce").fillna(0).astype("int8")
    out["dest_snowdepth_has_station"] = pd.to_numeric(out["dest_snowdepth_has_station"], errors="coerce").fillna(0).astype("int8")

    snowfall_origin = prefix_columns(
        ghcn_daily[["station_id", "weather_date", "snowfall_today", "snowfall_3d_sum"]].copy(),
        "origin_",
        exclude={"station_id", "weather_date"},
    ).rename(
        columns={
            "station_id": "origin_snowfall_station_lookup",
            "weather_date": "origin_snowfall_date_lookup",
        }
    )
    out = out.merge(
        snowfall_origin,
        how="left",
        left_on=["origin_snowfall_station_id", "FlightDate"],
        right_on=["origin_snowfall_station_lookup", "origin_snowfall_date_lookup"],
    )
    snowfall_dest = prefix_columns(
        ghcn_daily[["station_id", "weather_date", "snowfall_today", "snowfall_3d_sum"]].copy(),
        "dest_",
        exclude={"station_id", "weather_date"},
    ).rename(
        columns={
            "station_id": "dest_snowfall_station_lookup",
            "weather_date": "dest_snowfall_date_lookup",
        }
    )
    out = out.merge(
        snowfall_dest,
        how="left",
        left_on=["dest_snowfall_station_id", "scheduled_arr_date"],
        right_on=["dest_snowfall_station_lookup", "dest_snowfall_date_lookup"],
    )

    snowdepth_origin = prefix_columns(
        ghcn_daily[["station_id", "weather_date", "snow_depth_today"]].copy(),
        "origin_",
        exclude={"station_id", "weather_date"},
    ).rename(
        columns={
            "station_id": "origin_snowdepth_station_lookup",
            "weather_date": "origin_snowdepth_date_lookup",
        }
    )
    out = out.merge(
        snowdepth_origin,
        how="left",
        left_on=["origin_snowdepth_station_id", "FlightDate"],
        right_on=["origin_snowdepth_station_lookup", "origin_snowdepth_date_lookup"],
    )
    snowdepth_dest = prefix_columns(
        ghcn_daily[["station_id", "weather_date", "snow_depth_today"]].copy(),
        "dest_",
        exclude={"station_id", "weather_date"},
    ).rename(
        columns={
            "station_id": "dest_snowdepth_station_lookup",
            "weather_date": "dest_snowdepth_date_lookup",
        }
    )
    out = out.merge(
        snowdepth_dest,
        how="left",
        left_on=["dest_snowdepth_station_id", "scheduled_arr_date"],
        right_on=["dest_snowdepth_station_lookup", "dest_snowdepth_date_lookup"],
    )
    out["origin_snow_data_missing"] = (
        (
            pd.to_numeric(out["origin_snowfall_today"], errors="coerce").isna()
            & pd.to_numeric(out["origin_snow_depth_today"], errors="coerce").isna()
        )
        | (
            (pd.to_numeric(out["origin_snowfall_has_station"], errors="coerce").fillna(0) == 0)
            & (pd.to_numeric(out["origin_snowdepth_has_station"], errors="coerce").fillna(0) == 0)
        )
    ).astype("int8")
    out["dest_snow_data_missing"] = (
        (
            pd.to_numeric(out["dest_snowfall_today"], errors="coerce").isna()
            & pd.to_numeric(out["dest_snow_depth_today"], errors="coerce").isna()
        )
        | (
            (pd.to_numeric(out["dest_snowfall_has_station"], errors="coerce").fillna(0) == 0)
            & (pd.to_numeric(out["dest_snowdepth_has_station"], errors="coerce").fillna(0) == 0)
        )
    ).astype("int8")

    out = fill_defaults(out)
    out["route_weather_bottleneck"] = np.maximum(
        pd.to_numeric(out["origin_recent_weather_risk_12h"], errors="coerce").fillna(0),
        pd.to_numeric(out["dest_recent_weather_risk_12h"], errors="coerce").fillna(0),
    ).astype("float32")
    out["origin_dest_weather_gap"] = (
        pd.to_numeric(out["origin_recent_weather_risk_12h"], errors="coerce").fillna(0)
        - pd.to_numeric(out["dest_recent_weather_risk_12h"], errors="coerce").fillna(0)
    ).astype("float32")
    out["origin_dest_temp_gap"] = (
        pd.to_numeric(out["origin_recent_temp_3h_mean"], errors="coerce").fillna(0)
        - pd.to_numeric(out["dest_recent_temp_3h_mean"], errors="coerce").fillna(0)
    ).astype("float32")
    out["route_snow_bottleneck"] = np.maximum(
        pd.to_numeric(out["origin_snowfall_today"], errors="coerce").fillna(0),
        pd.to_numeric(out["dest_snowfall_today"], errors="coerce").fillna(0),
    ).astype("float32")
    out["origin_dest_freezing_pair_flag"] = np.maximum(
        pd.to_numeric(out["origin_freezing_flag"], errors="coerce").fillna(0),
        pd.to_numeric(out["dest_freezing_flag"], errors="coerce").fillna(0),
    ).astype("float32")

    return out.drop(
        columns=[
            "origin_weather_ts_lookup",
            "dest_weather_ts_lookup",
            "origin_metar_weather_ts_lookup",
            "dest_metar_weather_ts_lookup",
            "origin_metar_station_lookup",
            "dest_metar_station_lookup",
            "origin_snowfall_station_lookup",
            "dest_snowfall_station_lookup",
            "origin_snowfall_date_lookup",
            "dest_snowfall_date_lookup",
            "origin_snowdepth_station_lookup",
            "dest_snowdepth_station_lookup",
            "origin_snowdepth_date_lookup",
            "dest_snowdepth_date_lookup",
            "origin_hourly_station_id",
            "dest_hourly_station_id",
            "origin_snowfall_station_id",
            "dest_snowfall_station_id",
            "origin_snowdepth_station_id",
            "dest_snowdepth_station_id",
            "origin_hourly_has_station",
            "dest_hourly_has_station",
            "origin_snowfall_has_station",
            "dest_snowfall_has_station",
            "origin_snowdepth_has_station",
            "dest_snowdepth_has_station",
            "scheduled_dep_ts",
            "scheduled_arr_ts",
            "scheduled_arr_date",
        ],
        errors="ignore",
    )


def validate_output_frame(df: pd.DataFrame, source: str) -> dict[str, Any]:
    missing_cols = [col for col in VALIDATION_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{source} is missing expected v1.2.2 weather columns: {missing_cols}")

    stats = {
        "origin_metar_missing_rate": float(pd.to_numeric(df["origin_metar_data_missing"], errors="coerce").fillna(1).mean()),
        "origin_snow_missing_rate": float(pd.to_numeric(df["origin_snow_data_missing"], errors="coerce").fillna(1).mean()),
        "dest_metar_missing_rate": float(pd.to_numeric(df["dest_metar_data_missing"], errors="coerce").fillna(1).mean()),
        "dest_snow_missing_rate": float(pd.to_numeric(df["dest_snow_data_missing"], errors="coerce").fillna(1).mean()),
        "route_weather_bottleneck_max": float(pd.to_numeric(df["route_weather_bottleneck"], errors="coerce").fillna(0).max()),
        "origin_temp_signal_max": float(pd.to_numeric(df["origin_recent_temp_3h_mean"], errors="coerce").fillna(0).abs().max()),
    }
    if stats["origin_metar_missing_rate"] >= 0.60:
        raise ValueError(
            f"{source} has METAR coverage that is too sparse for v1.2.2: "
            f"origin_metar_missing_rate={stats['origin_metar_missing_rate']:.2%}"
        )
    if stats["origin_temp_signal_max"] <= 0:
        raise ValueError(f"{source} has no usable temperature signal.")
    if stats["route_weather_bottleneck_max"] <= 0:
        raise ValueError(f"{source} has no usable destination weather bottleneck signal.")
    return stats


def ensure_source_feature_store() -> None:
    missing = [str(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing v1.2.1 source feature-store files:\n"
            + "\n".join(missing)
            + "\nRun enterprise_delay_ensemble_v1_2_1\\build_feature_store_v1_2_1.py first."
        )


def collect_used_airports() -> set[str]:
    used: set[str] = set()
    for path in SOURCE_FILES.values():
        sample = pd.read_parquet(path, columns=["Origin", "Dest"])
        used.update(sample["Origin"].dropna().astype(str).str.upper().tolist())
        used.update(sample["Dest"].dropna().astype(str).str.upper().tolist())
    return used


def build_feature_store(force_rebuild_aux: bool = False) -> dict[str, Any]:
    ensure_source_feature_store()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    used_airports = collect_used_airports()
    airport_meta = load_airport_metadata(used_airports)
    station_inventory = build_noaa_station_inventory(force_rebuild=force_rebuild_aux)
    ghcn_station_summary = build_ghcn_station_summary(force_rebuild=force_rebuild_aux)

    hourly_mapping = map_airports_to_points(
        airport_meta,
        station_inventory[["station_id", "lat", "lon"]].copy(),
        candidate_id_col="station_id",
        max_distance_km=HOURLY_STATION_MAX_DISTANCE_KM,
    )
    snowfall_mapping = map_airports_to_points(
        airport_meta,
        ghcn_station_summary.loc[ghcn_station_summary["has_snow"] == 1, ["station_id", "lat", "lon"]].copy(),
        candidate_id_col="station_id",
        max_distance_km=GHCN_DAILY_MAX_DISTANCE_KM,
    )
    snowdepth_mapping = map_airports_to_points(
        airport_meta,
        ghcn_station_summary.loc[ghcn_station_summary["has_snwd"] == 1, ["station_id", "lat", "lon"]].copy(),
        candidate_id_col="station_id",
        max_distance_km=GHCN_DAILY_MAX_DISTANCE_KM,
    )
    hourly_mapping.to_csv(AUX_FILES["hourly_station_mapping"], index=False)
    snowfall_mapping.to_csv(AUX_FILES["ghcn_snow_mapping"], index=False)
    snowdepth_mapping.to_csv(AUX_FILES["ghcn_snwd_mapping"], index=False)

    enhanced_weather_history = build_enhanced_weather_history(force_rebuild=force_rebuild_aux)
    metar_hourly, _ = build_metar_hourly_and_daily_features(
        station_inventory=station_inventory,
        hourly_station_ids={
            station_id
            for station_id in (normalize_station_id(value) for value in hourly_mapping["station_id"].tolist())
            if station_id
        },
        daily_station_ids=set(),
        force_rebuild=force_rebuild_aux,
    )
    ghcn_daily = build_ghcn_daily_features(
        station_ids={
            station_id
            for station_id in (
                normalize_station_id(value)
                for value in (snowfall_mapping["station_id"].tolist() + snowdepth_mapping["station_id"].tolist())
            )
            if station_id
        },
        force_rebuild=force_rebuild_aux,
    )

    split_stats: dict[str, Any] = {}
    train_base = pd.read_parquet(SOURCE_FILES["train"])
    train_augmented = augment_split(
        base_df=train_base,
        enhanced_weather_history=enhanced_weather_history,
        metar_hourly=metar_hourly,
        ghcn_daily=ghcn_daily,
        hourly_mapping=hourly_mapping,
        snowfall_mapping=snowfall_mapping,
        snowdepth_mapping=snowdepth_mapping,
    )
    origin_norms = build_origin_airport_month_norms(train_augmented)
    train_augmented = apply_origin_airport_month_norms(train_augmented, origin_norms)
    split_stats["train"] = validate_output_frame(train_augmented, source="train v1.2.2 feature cache")
    train_augmented.to_parquet(OUTPUT_FILES["train"], index=False)
    del train_base, train_augmented
    gc.collect()

    for split_name in ["cv", "test"]:
        base_df = pd.read_parquet(SOURCE_FILES[split_name])
        augmented = augment_split(
            base_df=base_df,
            enhanced_weather_history=enhanced_weather_history,
            metar_hourly=metar_hourly,
            ghcn_daily=ghcn_daily,
            hourly_mapping=hourly_mapping,
            snowfall_mapping=snowfall_mapping,
            snowdepth_mapping=snowdepth_mapping,
        )
        augmented = apply_origin_airport_month_norms(augmented, origin_norms)
        split_stats[split_name] = validate_output_frame(augmented, source=f"{split_name} v1.2.2 feature cache")
        augmented.to_parquet(OUTPUT_FILES[split_name], index=False)
        del base_df, augmented
        gc.collect()

    coverage_manifest = {
        "source_cache_dir": str(SOURCE_CACHE_DIR),
        "output_cache_dir": str(ARTIFACT_DIR),
        "hourly_station_max_distance_km": HOURLY_STATION_MAX_DISTANCE_KM,
        "ghcn_daily_max_distance_km": GHCN_DAILY_MAX_DISTANCE_KM,
        "used_airports": int(len(used_airports)),
        "hourly_station_airport_coverage": int(hourly_mapping["has_mapping"].sum()),
        "ghcn_snow_airport_coverage": int(snowfall_mapping["has_mapping"].sum()),
        "ghcn_snow_depth_airport_coverage": int(snowdepth_mapping["has_mapping"].sum()),
        "metar_hourly_rows": int(len(metar_hourly)),
        "ghcn_daily_rows": int(len(ghcn_daily)),
        "split_stats": split_stats,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (ARTIFACT_DIR / "feature_store_manifest.json").write_text(json.dumps(coverage_manifest, indent=2), encoding="utf-8")
    return coverage_manifest


def validate_outputs() -> None:
    for split_name, path in OUTPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected v1.2.2 parquet missing: {path}")
        df = pd.read_parquet(path, columns=VALIDATION_COLUMNS + ["origin_metar_data_missing", "origin_snow_data_missing", "dest_metar_data_missing", "dest_snow_data_missing"])
        stats = validate_output_frame(df, source=f"{split_name} output validation")
        print(
            f"[validate] {path.name}: origin_metar_missing={stats['origin_metar_missing_rate']:.2%}, "
            f"origin_snow_missing={stats['origin_snow_missing_rate']:.2%}, "
            f"dest_metar_missing={stats['dest_metar_missing_rate']:.2%}, "
            f"dest_snow_missing={stats['dest_snow_missing_rate']:.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v1.2.2 weather-augmented feature store with METAR event features, daily snow summaries, destination weather joins, and airport-normalized weather context."
    )
    parser.add_argument(
        "--force-rebuild-aux",
        action="store_true",
        help="Rebuild the intermediate NOAA/weather augmentation caches even if they already exist.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip output parquet validation after building the v1.2.2 feature store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_feature_store(force_rebuild_aux=args.force_rebuild_aux)
    if not args.skip_validate:
        validate_outputs()
    print(
        "[v1.2.2 feature-store] completed with "
        f"hourly_airport_coverage={manifest['hourly_station_airport_coverage']}/{manifest['used_airports']} "
        f"and ghcn_snow_airport_coverage={manifest['ghcn_snow_airport_coverage']}/{manifest['used_airports']}"
    )


if __name__ == "__main__":
    main()
