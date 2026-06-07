from __future__ import annotations

import argparse
import gc
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FEATURE_CACHE_DIR = PROJECT_ROOT / "weather_safe_model_artifacts_meteostat_upgraded_v4_v1_2_2"

CACHE_FILES = {
    "train": FEATURE_CACHE_DIR / "train_2022_2023_2500000_weather_safe_upgraded_v12.parquet",
    "cv": FEATURE_CACHE_DIR / "cv_2024_250000_weather_safe_upgraded_v12.parquet",
    "test": FEATURE_CACHE_DIR / "test_2025_250000_weather_safe_upgraded_v12.parquet",
}

SAFE_CACHE_COLUMNS = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Marketing_Airline_Network",
    "Flight_Number_Marketing_Airline",
    "Operating_Airline",
    "Tail_Number",
    "OriginAirportID",
    "Origin",
    "OriginState",
    "DestAirportID",
    "Dest",
    "DestState",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "depdelay15",
    "route",
    "carrier_flight_key",
    "dep_hour",
    "arr_hour",
    "dep_hour_sin",
    "dep_hour_cos",
    "arr_hour_sin",
    "arr_hour_cos",
    "is_weekend",
    "is_overnight_sched",
    "same_state_route",
    "dep_commute_bank",
    "arr_commute_bank",
    "is_major_holiday",
    "is_near_major_holiday",
    "is_peak_travel",
    "origin_prev_weather_risk",
    "origin_recent_prcp_3h",
    "origin_recent_prcp_6h",
    "origin_recent_snow_6h",
    "origin_recent_wspd_6h_mean",
    "origin_recent_wpgt_6h_max",
    "origin_recent_bad_weather_6h_rate",
    "origin_recent_weather_risk_6h",
    "origin_recent_weather_risk_12h",
    "origin_snow_missing",
    "origin_wpgt_missing",
    "origin_wspd_missing",
    "origin_prcp_missing",
    "weather_data_missing_origin",
    "origin_weather_signal",
    "origin_recent_bad_weather_flag",
    "origin_frequency",
    "route_frequency",
    "origin_dep_hour_volume",
    "origin_day_departures_so_far",
    "carrier_day_departures_so_far",
    "carrier_origin_day_departures_so_far",
    "route_day_departures_so_far",
    "tail_departures_so_far",
    "flight_number_departures_so_far",
    "carrier_delay_rate",
    "operating_delay_rate",
    "origin_delay_rate",
    "route_delay_rate",
    "origin_hour_delay_rate",
    "airline_route_delay_rate",
    "flight_number_delay_rate",
    "origin_day_delay_rate_so_far",
    "carrier_day_delay_rate_so_far",
    "carrier_origin_day_delay_rate_so_far",
    "route_day_delay_rate_so_far",
    "origin_departures_last_30m",
    "origin_departures_last_1h",
    "origin_departures_last_3h",
    "carrier_origin_departures_last_1h",
    "carrier_origin_departures_last_3h",
    "origin_delays_last_30m_count",
    "origin_delays_last_1h_count",
    "origin_delays_last_3h_count",
    "carrier_origin_delays_last_3h_count",
    "origin_delay_rate_last_30m",
    "origin_delay_rate_last_1h",
    "origin_delay_rate_last_3h",
    "carrier_origin_delay_rate_last_3h",
    "prev_tail_delay15",
    "prev_tail_gap_hours_capped",
    "prev_tail_arrdelay_minutes_capped",
    "tail_turnaround_minutes_capped",
    "prev_flight_number_delay15",
    "prev_flight_number_arrdelay_minutes_capped",
    "traffic_pressure",
    "has_prev_tail_history",
    "has_prev_flight_number_history",
    "tail_turnaround_short_flag",
    "tail_turnaround_tight_flag",
    "tail_recovery_pressure",
    "flight_number_recovery_pressure",
    "origin_recent_pressure_score",
    "origin_recent_bad_weather_hours_6h",
    "origin_gust_gap_6h",
    "origin_recent_weather_risk_max_12h",
    "origin_weather_risk_delta_3h",
    "origin_weather_risk_delta_6h",
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
    "origin_snowfall_today",
    "origin_snowfall_3d_sum",
    "origin_snow_depth_today",
    "origin_snow_data_missing",
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
    "dest_snowfall_today",
    "dest_snowfall_3d_sum",
    "dest_snow_depth_today",
    "dest_snow_data_missing",
    "route_weather_bottleneck",
    "origin_dest_weather_gap",
    "origin_dest_temp_gap",
    "route_snow_bottleneck",
    "origin_dest_freezing_pair_flag",
    "origin_weather_signal_zscore_vs_airport_month",
    "origin_wspd_vs_airport_month_p90",
    "origin_prcp_vs_airport_month_avg",
]

CATBOOST_CATEGORICAL_COLUMNS = [
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Tail_Number",
    "Origin",
    "OriginState",
    "Dest",
    "DestState",
    "route",
    "carrier_flight_key",
    "Flight_Number_Marketing_Airline",
    "OriginAirportID",
    "DestAirportID",
]
LIGHTGBM_CATEGORICAL_COLUMNS = [
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Origin",
    "OriginState",
    "Dest",
    "DestState",
    "route",
    "OriginAirportID",
    "DestAirportID",
]
XGBOOST_CATEGORICAL_COLUMNS = [
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "OriginAirportID",
    "DestAirportID",
]
HIGH_CARDINALITY_IDENTIFIER_COLUMNS = [
    "Tail_Number",
    "route",
    "carrier_flight_key",
]
NUMERIC_CODED_CATEGORICAL_COLUMNS = [
    "Flight_Number_Marketing_Airline",
    "OriginAirportID",
    "DestAirportID",
]
TABULAR_CATEGORICAL_CANDIDATES = [
    "Marketing_Airline_Network",
    "Operating_Airline",
    "OriginState",
    "DestState",
    "Origin",
    "Dest",
    "OriginAirportID",
    "DestAirportID",
]
TABULAR_CARDINALITY_LIMIT = {
    "histgb": 450,
    "random_forest": 450,
    "extra_trees": 450,
    "balanced_rf": 450,
}

MODE_CONFIG = {
    "quick": {
        "cv_rows": 100_000,
        "test_rows": 100_000,
        "booster_train_rows": 600_000,
        "forest_train_rows": 250_000,
    },
    "full": {
        "cv_rows": 250_000,
        "test_rows": 250_000,
        "booster_train_rows": 2_000_000,
        "forest_train_rows": 500_000,
    },
}

GOAL_ACCURACY = 0.70
GOAL_RECALL = 0.70
GOAL_PRECISION = 0.35
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 2.0
SELECTION_PRECISION_FLOOR = 0.30
SELECTION_MAX_POSITIVE_RATE = 0.60
THRESHOLDS = np.round(np.linspace(0.05, 0.90, 171), 3)
FINALIST_COUNT = 3
MODEL_SAMPLE_SEEDS = {
    "CatBoost": 1101,
    "LightGBM": 1102,
    "Hist Gradient Boosting": 1103,
    "Random Forest": 1104,
    "Extra Trees": 1105,
    "Balanced Random Forest": 1106,
    "XGBoost": 1107,
}
AIRLINE_THRESHOLD_MIN_CV_ROWS = 400
AIRLINE_THRESHOLD_MIN_TEST_ROWS = 200
AIRLINE_THRESHOLD_MIN_CLASS_COUNT = 25
WEATHER_REQUIRED_COLUMNS = [
    "origin_recent_prcp_3h",
    "origin_recent_prcp_6h",
    "origin_recent_wspd_6h_mean",
    "origin_recent_weather_risk_6h",
    "origin_recent_weather_risk_12h",
    "origin_weather_signal",
    "weather_data_missing_origin",
    "origin_recent_weather_risk_max_12h",
    "origin_recent_temp_3h_mean",
    "origin_metar_snow_6h_count",
    "origin_snowfall_today",
    "origin_snow_depth_today",
    "dest_recent_weather_risk_12h",
    "route_weather_bottleneck",
    "origin_weather_signal_zscore_vs_airport_month",
]
WEATHER_VARIANCE_COLUMNS = [
    "origin_recent_prcp_3h",
    "origin_recent_wspd_6h_mean",
    "origin_recent_weather_risk_6h",
    "origin_recent_weather_risk_12h",
    "origin_weather_signal",
    "origin_recent_weather_risk_max_12h",
    "origin_recent_temp_3h_mean",
    "origin_metar_snow_6h_count",
    "origin_snowfall_today",
    "dest_recent_weather_risk_12h",
    "route_weather_bottleneck",
    "origin_weather_signal_zscore_vs_airport_month",
]
MIN_WEATHER_VARIANCE_COLUMNS = 2
MAX_WEATHER_MISSING_RATE = 0.98
THRESHOLD_DIR: Path | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    feature_view: str
    train_row_limit: int
    params: dict[str, Any]


@dataclass
class TrainedCandidate:
    spec: ModelSpec
    model: Any
    feature_bundle: dict[str, Any]
    feature_names: list[str]
    cv_scores: np.ndarray
    threshold: float
    threshold_table: pd.DataFrame
    cv_metrics: dict[str, float]
    fit_seconds: float


def safe_divide(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
    result = num.astype("float32") / den.replace(0, np.nan).astype("float32")
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value).astype("float32")


def stable_sample_month_aware(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size >= len(df):
        return df.reset_index(drop=True)

    work = df.copy()
    flight_month = pd.to_datetime(work["FlightDate"], errors="coerce").dt.to_period("M").astype("string").fillna("UNK")
    work["_sampling_stratum"] = flight_month + "|" + work["depdelay15"].astype("string")

    group_sizes = work["_sampling_stratum"].value_counts(dropna=False)
    desired = group_sizes / float(len(work)) * sample_size
    target_counts = np.floor(desired).astype(int)
    target_counts = target_counts.clip(lower=0, upper=group_sizes)

    remainder = int(sample_size - int(target_counts.sum()))
    if remainder > 0:
        order = (desired - target_counts).sort_values(ascending=False).index.tolist()
        for key in order:
            if remainder == 0:
                break
            if target_counts[key] < group_sizes[key]:
                target_counts[key] += 1
                remainder -= 1

    sampled_parts: list[pd.DataFrame] = []
    for stratum, group in work.groupby("_sampling_stratum", observed=True):
        take_n = int(target_counts.get(stratum, 0))
        if take_n <= 0:
            continue
        stratum_seed = seed + zlib.crc32(str(stratum).encode("utf-8")) % 100_000
        sampled_parts.append(group.sample(n=take_n, random_state=stratum_seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled.drop(columns="_sampling_stratum", errors="ignore")


def add_enterprise_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flight_date = pd.to_datetime(out["FlightDate"], errors="coerce")

    out["day_of_year"] = flight_date.dt.dayofyear.fillna(0).astype("int16")
    out["week_of_year"] = flight_date.dt.isocalendar().week.fillna(0).astype("int16")
    out["is_month_start"] = flight_date.dt.is_month_start.fillna(False).astype("int8")
    out["is_month_end"] = flight_date.dt.is_month_end.fillna(False).astype("int8")

    out["month_sin_enterprise"] = np.sin(2 * np.pi * out["Month"] / 12.0).astype("float32")
    out["month_cos_enterprise"] = np.cos(2 * np.pi * out["Month"] / 12.0).astype("float32")
    out["day_of_week_sin_enterprise"] = np.sin(2 * np.pi * out["DayOfWeek"] / 7.0).astype("float32")
    out["day_of_week_cos_enterprise"] = np.cos(2 * np.pi * out["DayOfWeek"] / 7.0).astype("float32")
    out["day_of_year_sin_enterprise"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0).astype("float32")
    out["day_of_year_cos_enterprise"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0).astype("float32")

    out["distance_per_sched_min"] = safe_divide(out["Distance"], out["CRSElapsedTime"].clip(lower=1))
    out["turnaround_buffer_ratio"] = safe_divide(
        out["tail_turnaround_minutes_capped"], out["CRSElapsedTime"].clip(lower=1)
    )
    out["tail_recovery_gap_minutes"] = (
        out["prev_tail_arrdelay_minutes_capped"] - out["tail_turnaround_minutes_capped"]
    ).astype("float32")
    out["flight_recovery_gap_ratio"] = safe_divide(
        out["prev_flight_number_arrdelay_minutes_capped"], out["CRSElapsedTime"].clip(lower=1)
    )

    out["origin_recent_delay_intensity"] = safe_divide(
        out["origin_delays_last_3h_count"], out["origin_departures_last_3h"].clip(lower=1)
    )
    out["origin_short_term_spike"] = safe_divide(
        out["origin_delays_last_30m_count"], out["origin_delays_last_3h_count"].clip(lower=1)
    )
    out["carrier_origin_recent_intensity"] = safe_divide(
        out["carrier_origin_delays_last_3h_count"], out["carrier_origin_departures_last_3h"].clip(lower=1)
    )
    out["origin_delay_momentum"] = (
        out["origin_delay_rate_last_30m"] - out["origin_delay_rate_last_3h"]
    ).astype("float32")
    out["origin_delay_acceleration"] = (
        out["origin_delay_rate_last_1h"] - out["origin_delay_rate_last_3h"]
    ).astype("float32")
    out["carrier_delay_momentum"] = (
        out["carrier_origin_delay_rate_last_3h"] - out["carrier_origin_day_delay_rate_so_far"]
    ).astype("float32")
    out["route_delay_momentum"] = (
        out["route_day_delay_rate_so_far"] - out["route_delay_rate"]
    ).astype("float32")

    out["weather_pressure_interaction"] = (
        out["origin_recent_weather_risk_6h"] * out["origin_departures_last_3h"]
    ).astype("float32")
    out["weather_ops_stress"] = (
        out["origin_recent_weather_risk_12h"] * out["origin_recent_pressure_score"]
    ).astype("float32")
    out["weather_recovery_stress"] = (
        out["origin_recent_weather_risk_6h"] * (out["tail_recovery_pressure"] + out["flight_number_recovery_pressure"])
    ).astype("float32")
    out["network_pressure_gap"] = (
        out["origin_recent_pressure_score"] - out["traffic_pressure"]
    ).astype("float32")
    out["route_weather_stress"] = (
        out["route_delay_rate"] * out["origin_weather_signal"]
    ).astype("float32")
    out["carrier_weather_stress"] = (
        out["carrier_delay_rate"] * out["origin_recent_weather_risk_6h"]
    ).astype("float32")
    out["weather_peak_bank_stress"] = (
        out["origin_recent_weather_risk_6h"] * out["dep_commute_bank"] * out["origin_departures_last_3h"]
    ).astype("float32")
    out["weather_turn_stress"] = (
        out["origin_recent_weather_risk_6h"] * out["tail_turnaround_tight_flag"]
    ).astype("float32")
    out["snow_ops_stress"] = (
        (out["origin_snowfall_today"] + 0.5 * out["origin_snowfall_3d_sum"] + 0.1 * out["origin_snow_depth_today"])
        * out["origin_departures_last_3h"]
    ).astype("float32")
    out["freezing_recovery_stress"] = (
        out["origin_freezing_precip_risk"] * (out["tail_recovery_pressure"] + out["flight_number_recovery_pressure"])
    ).astype("float32")
    out["destination_weather_pressure"] = (
        out["dest_recent_weather_risk_12h"] * out["route_frequency"]
    ).astype("float32")
    out["route_weather_gap_abs"] = np.abs(out["origin_dest_weather_gap"]).astype("float32")
    out["route_temp_gap_abs"] = np.abs(out["origin_dest_temp_gap"]).astype("float32")
    out["snow_route_bottleneck_stress"] = (
        out["route_snow_bottleneck"] * out["route_frequency"]
    ).astype("float32")
    out["dest_weather_recovery_stress"] = (
        out["dest_recent_weather_risk_6h"] * (out["tail_recovery_pressure"] + out["flight_number_recovery_pressure"])
    ).astype("float32")
    out["airport_normalized_weather_pressure"] = (
        out["origin_weather_signal_zscore_vs_airport_month"] * out["origin_departures_last_3h"]
    ).astype("float32")
    out["weather_disruption_mix"] = (
        out["route_weather_bottleneck"]
        + 0.35 * out["origin_recent_bad_weather_hours_12h"]
        + 0.25 * out["dest_recent_bad_weather_hours_12h"]
        + 0.50 * out["origin_metar_low_vis_3h_count"]
        + 0.50 * out["dest_metar_low_ceiling_now_flag"]
        + 0.50 * out["origin_metar_thunder_3h_count"]
    ).astype("float32")
    out["snow_freeze_combo"] = (
        out["route_snow_bottleneck"] * (1.0 + out["origin_dest_freezing_pair_flag"])
    ).astype("float32")

    out["origin_volume_log"] = np.log1p(out["origin_departures_last_3h"]).astype("float32")
    out["carrier_volume_log"] = np.log1p(out["carrier_origin_departures_last_3h"]).astype("float32")
    out["tail_usage_log"] = np.log1p(out["tail_departures_so_far"]).astype("float32")
    out["flight_number_usage_log"] = np.log1p(out["flight_number_departures_so_far"]).astype("float32")
    out["route_usage_log"] = np.log1p(out["route_frequency"]).astype("float32")
    return out


def validate_weather_feature_frame(df: pd.DataFrame, source: str) -> dict[str, Any]:
    missing_cols = [col for col in WEATHER_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{source} is missing required weather feature columns: {missing_cols}")

    missing_rate = float(pd.to_numeric(df["weather_data_missing_origin"], errors="coerce").fillna(1).mean())
    varying_cols = [col for col in WEATHER_VARIANCE_COLUMNS if df[col].nunique(dropna=False) > 1]
    active_cols = [
        col
        for col in WEATHER_VARIANCE_COLUMNS
        if float(pd.to_numeric(df[col], errors="coerce").fillna(0).abs().max()) > 0
    ]

    if missing_rate >= MAX_WEATHER_MISSING_RATE:
        raise ValueError(
            f"{source} has weather missing for {missing_rate:.2%} of rows. "
            "Run enterprise_delay_ensemble_v1_2_2\\build_feature_store_v1_2_2.py to rebuild a valid weather-aware cache."
        )
    if len(varying_cols) < MIN_WEATHER_VARIANCE_COLUMNS:
        raise ValueError(
            f"{source} does not contain enough varying weather features. "
            f"Only these vary: {varying_cols}."
        )
    if not active_cols:
        raise ValueError(f"{source} has no non-zero weather signal columns.")

    return {
        "weather_missing_rate": missing_rate,
        "varying_weather_cols": varying_cols,
        "active_weather_cols": active_cols,
    }


def load_split(split_name: str) -> pd.DataFrame:
    path = CACHE_FILES[split_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing engineered feature cache: {path}\n"
            "Run enterprise_delay_ensemble_v1_2_2\\build_feature_store_v1_2_2.py first."
        )
    df = pd.read_parquet(path, columns=SAFE_CACHE_COLUMNS)
    weather_stats = validate_weather_feature_frame(df, source=f"{split_name} feature cache ({path.name})")
    print(
        f"[weather-validate] {split_name}: missing_rate={weather_stats['weather_missing_rate']:.2%}, "
        f"active_cols={weather_stats['active_weather_cols'][:4]}"
    )
    return add_enterprise_features(df)


def load_data(mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = MODE_CONFIG[mode]
    train_df = load_split("train")
    cv_df = stable_sample_month_aware(load_split("cv"), cfg["cv_rows"], SEED + 100)
    test_df = stable_sample_month_aware(load_split("test"), cfg["test_rows"], SEED + 200)
    return train_df, cv_df, test_df


def get_model_categorical_columns(df: pd.DataFrame, spec: ModelSpec) -> list[str]:
    if spec.family == "catboost":
        candidates = CATBOOST_CATEGORICAL_COLUMNS
    elif spec.family == "lightgbm":
        candidates = LIGHTGBM_CATEGORICAL_COLUMNS
    elif spec.family == "xgboost":
        candidates = XGBOOST_CATEGORICAL_COLUMNS
    else:
        max_cardinality = TABULAR_CARDINALITY_LIMIT.get(spec.family, 80)
        candidates = [
            col
            for col in TABULAR_CATEGORICAL_CANDIDATES
            if col in df.columns and df[col].astype("string").nunique(dropna=True) <= max_cardinality
        ]
    return [col for col in candidates if col in df.columns]


def get_model_numeric_columns(df: pd.DataFrame, categorical_cols: list[str], spec: ModelSpec) -> list[str]:
    excluded = {"depdelay15", "FlightDate", *HIGH_CARDINALITY_IDENTIFIER_COLUMNS}
    if spec.feature_view == "dense_onehot":
        excluded.update(NUMERIC_CODED_CATEGORICAL_COLUMNS)

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in excluded or col in categorical_cols:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def fit_native_feature_bundle(train_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> dict[str, Any]:
    category_levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        levels = pd.Index(pd.unique(train_df[col].astype("string").fillna("UNK"))).astype("string").tolist()
        if "UNK" not in levels:
            levels.append("UNK")
        category_levels[col] = levels
    return {
        "kind": "native",
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "category_levels": category_levels,
        "feature_names": numeric_cols + categorical_cols,
    }


def build_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def fit_dense_onehot_bundle(train_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "kind": "dense_onehot",
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    imputer = SimpleImputer(strategy="median")
    if numeric_cols:
        imputer.fit(train_df[numeric_cols])
    bundle["imputer"] = imputer

    if categorical_cols:
        encoder = build_onehot_encoder()
        encoder.fit(train_df[categorical_cols].astype("string").fillna("UNK"))
        cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        encoder = None
        cat_feature_names = []
    bundle["encoder"] = encoder
    bundle["feature_names"] = numeric_cols + cat_feature_names
    return bundle


def transform_frame(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    numeric_cols: list[str] = bundle["numeric_cols"]
    categorical_cols: list[str] = bundle["categorical_cols"]

    if bundle["kind"] == "native":
        out = df[numeric_cols + categorical_cols].copy()
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        for col in categorical_cols:
            categories = bundle["category_levels"][col]
            values = out[col].astype("string").fillna("UNK")
            values = values.where(values.isin(categories), "UNK")
            categories = bundle["category_levels"][col]
            out[col] = pd.Categorical(values, categories=categories)
        return out

    numeric_df = pd.DataFrame(index=df.index)
    if numeric_cols:
        numeric_array = bundle["imputer"].transform(df[numeric_cols])
        numeric_df = pd.DataFrame(numeric_array, columns=numeric_cols, index=df.index, dtype=np.float32)

    categorical_df = pd.DataFrame(index=df.index)
    if categorical_cols:
        encoded = bundle["encoder"].transform(df[categorical_cols].astype("string").fillna("UNK"))
        cat_cols = bundle["feature_names"][len(numeric_cols):]
        categorical_df = pd.DataFrame(encoded, columns=cat_cols, index=df.index, dtype=np.float32)

    return pd.concat([numeric_df, categorical_df], axis=1)


def build_threshold_table(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_array = y_true.to_numpy()
    total = len(y_array)
    positives = max(1, int(y_array.sum()))
    negatives = max(1, total - positives)

    for threshold in THRESHOLDS:
        pred = (scores >= threshold).astype("int8")
        tp = int(((pred == 1) & (y_array == 1)).sum())
        fp = int(((pred == 1) & (y_array == 0)).sum())
        tn = int(((pred == 0) & (y_array == 0)).sum())
        fn = int(((pred == 0) & (y_array == 1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accuracy": float(accuracy_score(y_array, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_array, pred)),
                "precision": float(precision_score(y_array, pred, zero_division=0)),
                "recall": float(recall_score(y_array, pred, zero_division=0)),
                "f1": float(f1_score(y_array, pred, zero_division=0)),
                "positive_prediction_rate": float(pred.mean()),
                "false_negative_rate": float(fn / positives),
                "false_positive_rate": float(fp / negatives),
                "expected_cost_per_1k": float(((FALSE_NEGATIVE_COST * fn) + (FALSE_POSITIVE_COST * fp)) / total * 1000.0),
            }
        )
    return pd.DataFrame(rows)


def metric_priority(metrics: dict[str, float]) -> float:
    acc_gap = max(0.0, GOAL_ACCURACY - metrics["accuracy"])
    recall_gap = max(0.0, GOAL_RECALL - metrics["recall"])
    precision_gap = max(0.0, SELECTION_PRECISION_FLOOR - metrics["precision"])
    return (
        5.0 * float(metrics["recall"] >= GOAL_RECALL)
        + 2.5 * float(metrics["precision"] >= SELECTION_PRECISION_FLOOR)
        + 5.5 * metrics["recall"]
        + 1.9 * metrics["f1"]
        + 1.5 * metrics["precision"]
        + 1.0 * metrics["accuracy"]
        + 0.6 * metrics["balanced_accuracy"]
        - 10.0 * recall_gap
        - 2.5 * acc_gap
        - 3.0 * precision_gap
        - 0.015 * metrics.get("expected_cost_per_1k", 0.0)
        - 0.5 * metrics.get("false_negative_rate", 0.0)
        - 0.1 * metrics.get("false_positive_rate", 0.0)
    )


def choose_recall_first_row(
    table: pd.DataFrame,
    precision_floor: float = SELECTION_PRECISION_FLOOR,
    max_positive_rate: float = SELECTION_MAX_POSITIVE_RATE,
) -> pd.Series:
    recall_target = table[
        (table["recall"] >= GOAL_RECALL)
        & (table["precision"] >= precision_floor)
    ]
    if not recall_target.empty:
        return recall_target.sort_values(
            ["recall", "precision", "f1", "accuracy", "expected_cost_per_1k"],
            ascending=[False, False, False, False, True],
        ).iloc[0]

    precision_ok = table[table["precision"] >= precision_floor]
    bounded = precision_ok[precision_ok["positive_prediction_rate"] <= max_positive_rate]
    pool = bounded if not bounded.empty else precision_ok
    if not pool.empty:
        return pool.sort_values(
            ["recall", "f1", "precision", "accuracy", "expected_cost_per_1k"],
            ascending=[False, False, False, False, True],
        ).iloc[0]

    bounded = table[table["positive_prediction_rate"] <= max_positive_rate]
    if not bounded.empty:
        return bounded.sort_values(
            ["recall", "precision", "f1", "accuracy"],
            ascending=[False, False, False, False],
        ).iloc[0]

    return table.sort_values(
        ["recall", "precision", "f1", "accuracy"],
        ascending=[False, False, False, False],
    ).iloc[0]


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    table = build_threshold_table(y_true, scores)
    table["selection_score"] = table.apply(lambda row: metric_priority(row.to_dict()), axis=1)

    strict = table[
        (table["accuracy"] >= GOAL_ACCURACY)
        & (table["recall"] >= GOAL_RECALL)
        & (table["precision"] >= GOAL_PRECISION)
    ]
    if not strict.empty:
        chosen = strict.sort_values(
            ["expected_cost_per_1k", "f1", "recall", "accuracy"],
            ascending=[True, False, False, False],
        ).iloc[0]
        return chosen.to_dict(), table

    chosen = choose_recall_first_row(table)
    return chosen.to_dict(), table


def evaluate_binary_predictions(y_true: pd.Series | np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype="int8")
    pred_array = np.asarray(pred, dtype="int8")
    score_array = np.asarray(scores, dtype="float32")
    tp = int(((pred_array == 1) & (y_array == 1)).sum())
    fp = int(((pred_array == 1) & (y_array == 0)).sum())
    tn = int(((pred_array == 0) & (y_array == 0)).sum())
    fn = int(((pred_array == 0) & (y_array == 1)).sum())
    positives = max(1, int(y_array.sum()))
    negatives = max(1, len(y_array) - positives)
    if np.unique(y_array).size > 1:
        roc_auc = float(roc_auc_score(y_array, score_array))
        average_precision = float(average_precision_score(y_array, score_array))
    else:
        roc_auc = float("nan")
        average_precision = float("nan")
    metrics = {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(y_array, pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, pred_array)),
        "precision": float(precision_score(y_array, pred_array, zero_division=0)),
        "recall": float(recall_score(y_array, pred_array, zero_division=0)),
        "f1": float(f1_score(y_array, pred_array, zero_division=0)),
        "positive_prediction_rate": float(pred_array.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_negative_rate": float(fn / positives),
        "false_positive_rate": float(fp / negatives),
        "expected_cost_per_1k": float(((FALSE_NEGATIVE_COST * fn) + (FALSE_POSITIVE_COST * fp)) / len(y_array) * 1000.0),
    }
    return metrics


def evaluate_scores(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype("int8")
    metrics = evaluate_binary_predictions(y_true=y_true, pred=pred, scores=scores)
    metrics["threshold"] = float(threshold)
    metrics["selection_score"] = float(metric_priority(metrics))
    metrics["meets_accuracy_goal"] = float(metrics["accuracy"] >= GOAL_ACCURACY)
    metrics["meets_recall_goal"] = float(metrics["recall"] >= GOAL_RECALL)
    metrics["meets_precision_goal"] = float(metrics["precision"] >= GOAL_PRECISION)
    return metrics


def current_pos_weight(y: pd.Series) -> float:
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    return max(1.0, negatives / max(1.0, positives))


def fit_catboost(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    y_cv: pd.Series,
    params: dict[str, Any],
    categorical_cols: list[str],
) -> tuple[Any, np.ndarray]:
    cat_features = [train_x.columns.get_loc(col) for col in categorical_cols]
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    model.fit(
        Pool(train_x, y_train, cat_features=cat_features),
        eval_set=Pool(cv_x, y_cv, cat_features=cat_features),
        use_best_model=True,
        early_stopping_rounds=120,
    )
    return model, model.predict_proba(cv_x)[:, 1]


def fit_lightgbm(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    y_cv: pd.Series,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    model = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(
        train_x,
        y_train,
        eval_set=[(cv_x, y_cv)],
        eval_metric="auc",
        callbacks=[early_stopping(120, verbose=False)],
    )
    return model, model.predict_proba(cv_x)[:, 1]


def fit_xgboost(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    y_cv: pd.Series,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=current_pos_weight(y_train),
        random_state=SEED,
        n_jobs=-1,
        **params,
    )
    model.fit(train_x, y_train, eval_set=[(cv_x, y_cv)], verbose=False)
    return model, model.predict_proba(cv_x)[:, 1]


def fit_histgb(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    sample_weight = np.where(y_train.to_numpy() == 1, current_pos_weight(y_train), 1.0)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=True,
        random_state=SEED,
        **params,
    )
    model.fit(train_x, y_train, sample_weight=sample_weight)
    return model, model.predict_proba(cv_x)[:, 1]


def fit_random_forest(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
        **params,
    )
    model.fit(train_x, y_train)
    return model, model.predict_proba(cv_x)[:, 1]


def fit_extra_trees(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    model = ExtraTreesClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
        **params,
    )
    model.fit(train_x, y_train)
    return model, model.predict_proba(cv_x)[:, 1]


def fit_balanced_rf(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    model = BalancedRandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        **params,
    )
    model.fit(train_x, y_train)
    return model, model.predict_proba(cv_x)[:, 1]


def save_feature_importance(model: Any, feature_names: list[str], output_dir: Path, model_name: str) -> None:
    try:
        if hasattr(model, "get_feature_importance"):
            values = model.get_feature_importance()
        elif hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        else:
            return
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.asarray(values, dtype="float64"),
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(output_dir / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv", index=False)
    except Exception:
        return


def fit_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    output_dir: Path,
) -> TrainedCandidate:
    sample_seed = MODEL_SAMPLE_SEEDS[spec.name]
    sampled_train = stable_sample_month_aware(train_df, spec.train_row_limit, sample_seed)
    y_train = sampled_train["depdelay15"].astype("int8")
    y_cv = cv_df["depdelay15"].astype("int8")

    categorical_cols = get_model_categorical_columns(sampled_train, spec)
    numeric_cols = get_model_numeric_columns(sampled_train, categorical_cols, spec)

    start = time.perf_counter()
    if spec.feature_view == "native":
        feature_bundle = fit_native_feature_bundle(sampled_train, numeric_cols, categorical_cols)
    else:
        feature_bundle = fit_dense_onehot_bundle(sampled_train, numeric_cols, categorical_cols)

    train_x = transform_frame(sampled_train, feature_bundle)
    cv_x = transform_frame(cv_df, feature_bundle)
    feature_names = feature_bundle["feature_names"]

    if spec.family == "catboost":
        model, cv_scores = fit_catboost(train_x, y_train, cv_x, y_cv, spec.params, categorical_cols)
    elif spec.family == "lightgbm":
        model, cv_scores = fit_lightgbm(train_x, y_train, cv_x, y_cv, spec.params)
    elif spec.family == "xgboost":
        model, cv_scores = fit_xgboost(train_x, y_train, cv_x, y_cv, spec.params)
    elif spec.family == "histgb":
        model, cv_scores = fit_histgb(train_x, y_train, cv_x, spec.params)
    elif spec.family == "random_forest":
        model, cv_scores = fit_random_forest(train_x, y_train, cv_x, spec.params)
    elif spec.family == "extra_trees":
        model, cv_scores = fit_extra_trees(train_x, y_train, cv_x, spec.params)
    elif spec.family == "balanced_rf":
        model, cv_scores = fit_balanced_rf(train_x, y_train, cv_x, spec.params)
    else:
        raise ValueError(f"Unknown model family: {spec.family}")

    fit_seconds = time.perf_counter() - start
    chosen_threshold, threshold_table = choose_threshold(y_cv, cv_scores)
    cv_metrics = evaluate_scores(y_cv, cv_scores, chosen_threshold["threshold"])
    save_feature_importance(model, feature_names, output_dir, spec.name)

    del sampled_train, train_x, cv_x
    gc.collect()
    return TrainedCandidate(
        spec=spec,
        model=model,
        feature_bundle=feature_bundle,
        feature_names=feature_names,
        cv_scores=cv_scores,
        threshold=float(chosen_threshold["threshold"]),
        threshold_table=threshold_table,
        cv_metrics=cv_metrics,
        fit_seconds=fit_seconds,
    )


def predict_scores(candidate: TrainedCandidate, df: pd.DataFrame) -> np.ndarray:
    frame = transform_frame(df, candidate.feature_bundle)
    scores = candidate.model.predict_proba(frame)[:, 1]
    del frame
    gc.collect()
    return scores


def candidate_to_cv_row(candidate: TrainedCandidate) -> dict[str, Any]:
    return {
        "model_name": candidate.spec.name,
        "family": candidate.spec.family,
        "feature_view": candidate.spec.feature_view,
        "train_rows_used": candidate.spec.train_row_limit,
        "fit_seconds": round(candidate.fit_seconds, 2),
        "threshold": candidate.threshold,
        "blend_members": "",
        "blend_weights": "",
        **{f"cv_{k}": v for k, v in candidate.cv_metrics.items()},
    }


def blend_top_models_cv(
    base_results_df: pd.DataFrame,
    cv_prob_map: dict[str, np.ndarray],
    y_cv: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if THRESHOLD_DIR is None:
        raise RuntimeError("Threshold directory has not been initialized.")

    blend_rows: list[dict[str, Any]] = []
    blended_cv: dict[str, np.ndarray] = {}
    top_models = base_results_df.sort_values("cv_selection_score", ascending=False).head(3)["model_name"].tolist()
    if len(top_models) < 2:
        return blend_rows, blended_cv

    soft_name = "Soft Vote Top 3"
    soft_cv = np.mean([cv_prob_map[name] for name in top_models], axis=0)
    chosen_threshold, threshold_table = choose_threshold(y_cv, soft_cv)
    cv_metrics = evaluate_scores(y_cv, soft_cv, chosen_threshold["threshold"])
    blend_rows.append(
        {
            "model_name": soft_name,
            "family": "blend",
            "feature_view": "blend",
            "fit_seconds": 0.0,
            "train_rows_used": 0,
            "threshold": chosen_threshold["threshold"],
            "blend_members": ", ".join(top_models),
            "blend_weights": json.dumps([round(1.0 / len(top_models), 3)] * len(top_models)),
            **{f"cv_{k}": v for k, v in cv_metrics.items()},
        }
    )
    threshold_table.to_csv(THRESHOLD_DIR / "soft_vote_top_3_thresholds.csv", index=False)
    blended_cv[soft_name] = soft_cv

    best_row: dict[str, Any] | None = None
    best_cv_scores: np.ndarray | None = None
    steps = np.round(np.linspace(0.0, 1.0, 51), 3)
    for w1, w2 in product(steps, repeat=2):
        if w1 + w2 > 1.0:
            continue
        w3 = round(1.0 - w1 - w2, 3)
        weights = np.array([w1, w2, w3], dtype="float32")
        if np.count_nonzero(weights) < 2:
            continue
        cv_scores = np.average([cv_prob_map[name] for name in top_models], axis=0, weights=weights)
        chosen, table = choose_threshold(y_cv, cv_scores)
        cv_metrics = evaluate_scores(y_cv, cv_scores, chosen["threshold"])
        row = {
            "model_name": "Weighted Blend Top 3",
            "family": "blend",
            "feature_view": "blend",
            "fit_seconds": 0.0,
            "train_rows_used": 0,
            "threshold": chosen["threshold"],
            "blend_members": ", ".join(top_models),
            "blend_weights": json.dumps([round(float(x), 3) for x in weights]),
            **{f"cv_{k}": v for k, v in cv_metrics.items()},
        }
        if best_row is None or row["cv_selection_score"] > best_row["cv_selection_score"]:
            best_row = row
            best_cv_scores = cv_scores
            table.to_csv(THRESHOLD_DIR / "weighted_blend_top_3_thresholds.csv", index=False)

    if best_row is not None and best_cv_scores is not None:
        blend_rows.append(best_row)
        blended_cv[best_row["model_name"]] = best_cv_scores

    return blend_rows, blended_cv


def build_model_specs(mode: str) -> list[ModelSpec]:
    cfg = MODE_CONFIG[mode]
    specs = [
        ModelSpec("CatBoost", "catboost", "native", cfg["booster_train_rows"], {
            "iterations": 2400,
            "depth": 8,
            "learning_rate": 0.025,
            "l2_leaf_reg": 12.0,
            "bagging_temperature": 0.75,
            "random_strength": 1.5,
        }),
        ModelSpec("LightGBM", "lightgbm", "native", cfg["booster_train_rows"], {
            "n_estimators": 2200,
            "learning_rate": 0.022,
            "num_leaves": 127,
            "min_child_samples": 140,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.5,
        }),
        ModelSpec("Hist Gradient Boosting", "histgb", "dense_onehot", cfg["forest_train_rows"], {
            "max_iter": 700,
            "learning_rate": 0.04,
            "max_leaf_nodes": 127,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            "validation_fraction": 0.1,
        }),
        ModelSpec("XGBoost", "xgboost", "native", cfg["booster_train_rows"], {
            "n_estimators": 1600,
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.5,
            "gamma": 0.1,
        }),
        ModelSpec("Random Forest", "random_forest", "dense_onehot", cfg["forest_train_rows"], {
            "n_estimators": 500,
            "max_depth": 18,
            "min_samples_leaf": 8,
            "max_features": "sqrt",
        }),
        ModelSpec("Extra Trees", "extra_trees", "dense_onehot", cfg["forest_train_rows"], {
            "n_estimators": 500,
            "max_depth": 18,
            "min_samples_leaf": 6,
            "max_features": "sqrt",
        }),
    ]
    if mode == "quick":
        specs.append(
            ModelSpec("Balanced Random Forest", "balanced_rf", "dense_onehot", cfg["forest_train_rows"], {
                "n_estimators": 500,
                "max_depth": 16,
                "min_samples_leaf": 8,
                "max_features": "sqrt",
                "replacement": True,
            })
        )
    return specs


def save_dataset_summary(train_df: pd.DataFrame, cv_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = pd.DataFrame(
        [
            {"split": "train", "years": "2022,2023", "rows": len(train_df), "delay_rate": float(train_df["depdelay15"].mean())},
            {"split": "cv", "years": "2024", "rows": len(cv_df), "delay_rate": float(cv_df["depdelay15"].mean())},
            {"split": "test", "years": "2025", "rows": len(test_df), "delay_rate": float(test_df["depdelay15"].mean())},
        ]
    )
    summary_df.to_csv(output_dir / "dataset_summary.csv", index=False)


def select_operating_profiles(table: pd.DataFrame) -> dict[str, float]:
    profiles: dict[str, float] = {}

    profiles["balanced_default"] = float(choose_recall_first_row(table)["threshold"])
    profiles["recall_leaning"] = float(
        choose_recall_first_row(
            table,
            precision_floor=max(0.0, SELECTION_PRECISION_FLOOR - 0.02),
            max_positive_rate=min(0.70, SELECTION_MAX_POSITIVE_RATE + 0.05),
        )["threshold"]
    )

    acc_guardrail = table[
        (table["accuracy"] >= GOAL_ACCURACY)
        & (table["precision"] >= SELECTION_PRECISION_FLOOR)
    ]
    if acc_guardrail.empty:
        acc_guardrail = table[table["precision"] >= SELECTION_PRECISION_FLOOR]
    if acc_guardrail.empty:
        acc_guardrail = table
    profiles["accuracy_guardrail"] = float(
        acc_guardrail.sort_values(
            ["recall", "accuracy", "precision", "expected_cost_per_1k"],
            ascending=[False, False, False, True],
        ).iloc[0]["threshold"]
    )

    cost_control = table[table["precision"] >= GOAL_PRECISION]
    if cost_control.empty:
        cost_control = table[table["precision"] >= SELECTION_PRECISION_FLOOR]
    if cost_control.empty:
        cost_control = table
    recall_ready = cost_control[cost_control["recall"] >= 0.45]
    if not recall_ready.empty:
        cost_control = recall_ready
    profiles["cost_control"] = float(
        cost_control.sort_values(
            ["accuracy", "precision", "expected_cost_per_1k", "recall"],
            ascending=[False, False, True, False],
        ).iloc[0]["threshold"]
    )
    return profiles


def write_deployment_threshold_profiles(
    candidate_name: str,
    threshold_table: pd.DataFrame,
    y_test: pd.Series,
    test_scores: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    profiles = select_operating_profiles(threshold_table)
    rows: list[dict[str, Any]] = []
    rationale_map = {
        "balanced_default": "Recall-first default operating point selected on validation.",
        "recall_leaning": "More aggressive recall-first threshold to catch additional delayed flights.",
        "accuracy_guardrail": "Highest-recall operating point that still targets 70%+ accuracy when available.",
        "cost_control": "Higher accuracy / precision operating point for tighter intervention budgets.",
    }

    threshold_lookup = threshold_table.set_index("threshold")
    for profile_name, threshold in profiles.items():
        cv_row = threshold_lookup.loc[threshold].to_dict()
        test_metrics = evaluate_scores(y_test, test_scores, threshold)
        rows.append(
            {
                "model_name": candidate_name,
                "profile": profile_name,
                "threshold": threshold,
                "cv_accuracy": cv_row["accuracy"],
                "cv_precision": cv_row["precision"],
                "cv_recall": cv_row["recall"],
                "cv_f1": cv_row["f1"],
                "cv_expected_cost_per_1k": cv_row["expected_cost_per_1k"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_expected_cost_per_1k": test_metrics["expected_cost_per_1k"],
                "positive_prediction_rate": test_metrics["positive_prediction_rate"],
                "rationale": rationale_map[profile_name],
            }
        )

    profiles_df = pd.DataFrame(rows)
    profiles_df.to_csv(output_dir / "deployment_threshold_profiles.csv", index=False)
    return profiles_df


def compute_slice_metrics(
    df: pd.DataFrame,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    group_col: str,
    output_path: Path,
    top_n: int | None = None,
) -> None:
    pred = (scores >= threshold).astype("int8")
    work = df[[group_col]].copy()
    work["actual"] = y_true.to_numpy()
    work["pred"] = pred
    work["score"] = scores

    if top_n is not None:
        keep_values = work[group_col].value_counts().head(top_n).index
        work = work[work[group_col].isin(keep_values)].copy()

    rows: list[dict[str, Any]] = []
    for value, group in work.groupby(group_col, observed=True):
        if len(group) < 100:
            continue
        rows.append(
            {
                group_col: value,
                "rows": len(group),
                "delay_rate": float(group["actual"].mean()),
                "accuracy": float(accuracy_score(group["actual"], group["pred"])),
                "precision": float(precision_score(group["actual"], group["pred"], zero_division=0)),
                "recall": float(recall_score(group["actual"], group["pred"], zero_division=0)),
                "f1": float(f1_score(group["actual"], group["pred"], zero_division=0)),
                "average_score": float(group["score"].mean()),
                "positive_prediction_rate": float(group["pred"].mean()),
            }
        )
    pd.DataFrame(rows).sort_values("rows", ascending=False).to_csv(output_path, index=False)


def run_airline_threshold_experiment(
    cv_df: pd.DataFrame,
    y_cv: pd.Series,
    cv_scores: np.ndarray,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_scores: np.ndarray,
    global_threshold: float,
    output_path: Path,
) -> pd.DataFrame:
    cv_airlines = cv_df["Operating_Airline"].astype("string").fillna("UNK").to_numpy()
    test_airlines = test_df["Operating_Airline"].astype("string").fillna("UNK").to_numpy()
    y_cv_array = y_cv.to_numpy(dtype="int8")
    y_test_array = y_test.to_numpy(dtype="int8")

    policy_pred = (test_scores >= global_threshold).astype("int8")
    rows: list[dict[str, Any]] = []

    for airline in sorted(pd.Index(test_airlines).unique().astype("string").tolist()):
        test_mask = test_airlines == airline
        if not test_mask.any():
            continue

        cv_mask = cv_airlines == airline
        cv_rows = int(cv_mask.sum())
        cv_pos = int(y_cv_array[cv_mask].sum()) if cv_rows else 0
        cv_neg = cv_rows - cv_pos
        threshold = float(global_threshold)
        threshold_source = "global_fallback"
        if cv_rows:
            cv_metrics = evaluate_binary_predictions(
                y_true=y_cv_array[cv_mask],
                pred=(cv_scores[cv_mask] >= threshold).astype("int8"),
                scores=cv_scores[cv_mask],
            )
        else:
            cv_metrics = {
                "precision": float("nan"),
                "recall": float("nan"),
                "accuracy": float("nan"),
                "positive_prediction_rate": float("nan"),
            }

        if (
            cv_rows >= AIRLINE_THRESHOLD_MIN_CV_ROWS
            and cv_pos >= AIRLINE_THRESHOLD_MIN_CLASS_COUNT
            and cv_neg >= AIRLINE_THRESHOLD_MIN_CLASS_COUNT
        ):
            chosen, _ = choose_threshold(pd.Series(y_cv_array[cv_mask]), cv_scores[cv_mask])
            threshold = float(chosen["threshold"])
            threshold_source = "airline_specific"
            cv_metrics = chosen

        policy_pred[test_mask] = (test_scores[test_mask] >= threshold).astype("int8")
        if int(test_mask.sum()) < AIRLINE_THRESHOLD_MIN_TEST_ROWS:
            continue

        test_metrics = evaluate_binary_predictions(
            y_true=y_test_array[test_mask],
            pred=policy_pred[test_mask],
            scores=test_scores[test_mask],
        )
        global_test_metrics = evaluate_scores(
            y_true=pd.Series(y_test_array[test_mask]),
            scores=test_scores[test_mask],
            threshold=global_threshold,
        )
        rows.append(
            {
                "Operating_Airline": airline,
                "threshold_source": threshold_source,
                "selected_threshold": threshold,
                "global_threshold": global_threshold,
                "cv_rows": cv_rows,
                "cv_delay_rate": float(y_cv_array[cv_mask].mean()) if cv_rows else float("nan"),
                "test_rows": int(test_mask.sum()),
                "test_delay_rate": float(y_test_array[test_mask].mean()),
                "cv_precision": float(cv_metrics["precision"]),
                "cv_recall": float(cv_metrics["recall"]),
                "cv_accuracy": float(cv_metrics["accuracy"]),
                "cv_positive_prediction_rate": float(cv_metrics["positive_prediction_rate"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1": float(test_metrics["f1"]),
                "test_positive_prediction_rate": float(test_metrics["positive_prediction_rate"]),
                "global_test_precision": float(global_test_metrics["precision"]),
                "global_test_recall": float(global_test_metrics["recall"]),
                "global_test_accuracy": float(global_test_metrics["accuracy"]),
                "global_test_positive_prediction_rate": float(global_test_metrics["positive_prediction_rate"]),
                "recall_lift_vs_global": float(test_metrics["recall"] - global_test_metrics["recall"]),
                "precision_delta_vs_global": float(test_metrics["precision"] - global_test_metrics["precision"]),
                "accuracy_delta_vs_global": float(test_metrics["accuracy"] - global_test_metrics["accuracy"]),
            }
        )

    mixed_policy_metrics = evaluate_binary_predictions(y_true=y_test_array, pred=policy_pred, scores=test_scores)
    global_metrics = evaluate_scores(y_true=y_test, scores=test_scores, threshold=global_threshold)
    rows.append(
        {
            "Operating_Airline": "__overall__",
            "threshold_source": "mixed_policy",
            "selected_threshold": float("nan"),
            "global_threshold": global_threshold,
            "cv_rows": len(cv_df),
            "cv_delay_rate": float(y_cv.mean()),
            "test_rows": len(test_df),
            "test_delay_rate": float(y_test.mean()),
            "cv_precision": float("nan"),
            "cv_recall": float("nan"),
            "cv_accuracy": float("nan"),
            "cv_positive_prediction_rate": float("nan"),
            "test_precision": float(mixed_policy_metrics["precision"]),
            "test_recall": float(mixed_policy_metrics["recall"]),
            "test_accuracy": float(mixed_policy_metrics["accuracy"]),
            "test_f1": float(mixed_policy_metrics["f1"]),
            "test_positive_prediction_rate": float(mixed_policy_metrics["positive_prediction_rate"]),
            "global_test_precision": float(global_metrics["precision"]),
            "global_test_recall": float(global_metrics["recall"]),
            "global_test_accuracy": float(global_metrics["accuracy"]),
            "global_test_positive_prediction_rate": float(global_metrics["positive_prediction_rate"]),
            "recall_lift_vs_global": float(mixed_policy_metrics["recall"] - global_metrics["recall"]),
            "precision_delta_vs_global": float(mixed_policy_metrics["precision"] - global_metrics["precision"]),
            "accuracy_delta_vs_global": float(mixed_policy_metrics["accuracy"] - global_metrics["accuracy"]),
        }
    )

    experiment_df = pd.DataFrame(rows).sort_values(
        ["Operating_Airline", "test_rows"],
        ascending=[True, False],
    ).reset_index(drop=True)
    experiment_df.to_csv(output_path, index=False)
    return experiment_df


def make_bundle_payload(
    winner_row: pd.Series,
    base_candidates: dict[str, TrainedCandidate],
    threshold_profiles: pd.DataFrame,
) -> dict[str, Any]:
    if winner_row["family"] == "blend":
        member_names = [name.strip() for name in str(winner_row["blend_members"]).split(",") if name.strip()]
        member_weights = json.loads(winner_row["blend_weights"])
    else:
        member_names = [winner_row["model_name"]]
        member_weights = [1.0]

    components = {}
    for name in member_names:
        candidate = base_candidates[name]
        components[name] = {
            "spec": asdict(candidate.spec),
            "model": candidate.model,
            "feature_bundle": candidate.feature_bundle,
            "feature_names": candidate.feature_names,
        }

    return {
        "bundle_version": "v1.2.2",
        "model_name": winner_row["model_name"],
        "family": winner_row["family"],
        "default_threshold": float(winner_row["threshold"]),
        "member_names": member_names,
        "member_weights": member_weights,
        "threshold_profiles": threshold_profiles.to_dict(orient="records"),
        "components": components,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }


def sort_cv_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["cv_selection_score", "cv_recall", "cv_accuracy", "cv_precision", "cv_f1"],
        ascending=False,
    ).reset_index(drop=True)


def run_experiment(mode: str, output_dir: Path, resume_base_results: bool = False) -> None:
    global THRESHOLD_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    THRESHOLD_DIR = output_dir / "threshold_tables"
    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

    train_df, cv_df, test_df = load_data(mode)
    save_dataset_summary(train_df, cv_df, test_df, output_dir)
    y_cv = cv_df["depdelay15"].astype("int8")
    y_test = test_df["depdelay15"].astype("int8")

    base_candidates: dict[str, TrainedCandidate] = {}
    cv_prob_map: dict[str, np.ndarray] = {}
    model_specs = build_model_specs(mode)

    if resume_base_results and (output_dir / "base_model_results.csv").exists():
        base_results_df = sort_cv_results(pd.read_csv(output_dir / "base_model_results.csv"))
        top_base_names = base_results_df.head(3)["model_name"].tolist()
        spec_lookup = {spec.name: spec for spec in model_specs}
        for model_name in top_base_names:
            spec = spec_lookup[model_name]
            candidate = fit_model(spec, train_df, cv_df, output_dir)
            base_candidates[spec.name] = candidate
            cv_prob_map[spec.name] = candidate.cv_scores
            candidate.threshold_table.to_csv(THRESHOLD_DIR / f"{spec.name.lower().replace(' ', '_')}_thresholds.csv", index=False)
            gc.collect()
    else:
        base_rows: list[dict[str, Any]] = []
        for spec in model_specs:
            candidate = fit_model(spec, train_df, cv_df, output_dir)
            base_candidates[spec.name] = candidate
            base_rows.append(candidate_to_cv_row(candidate))
            cv_prob_map[spec.name] = candidate.cv_scores
            candidate.threshold_table.to_csv(THRESHOLD_DIR / f"{spec.name.lower().replace(' ', '_')}_thresholds.csv", index=False)
            pd.DataFrame(base_rows).to_csv(output_dir / "base_model_results_partial.csv", index=False)
            gc.collect()

        base_results_df = sort_cv_results(pd.DataFrame(base_rows))
        base_results_df.to_csv(output_dir / "base_model_results.csv", index=False)

    blend_rows, blended_cv = blend_top_models_cv(base_results_df, cv_prob_map, y_cv)
    blend_df = pd.DataFrame(blend_rows)
    if not blend_df.empty:
        blend_df = sort_cv_results(blend_df)
        blend_df.to_csv(output_dir / "blend_results.csv", index=False)
        cv_prob_map.update(blended_cv)

    combined_results = sort_cv_results(pd.concat([base_results_df, blend_df], ignore_index=True))
    combined_results.to_csv(output_dir / "cv_model_results_ranked.csv", index=False)

    finalists = combined_results.head(FINALIST_COUNT).copy()
    finalists.to_csv(output_dir / "cv_finalists.csv", index=False)

    required_base_names: set[str] = set()
    for _, row in finalists.iterrows():
        if row["family"] == "blend":
            required_base_names.update(name.strip() for name in str(row["blend_members"]).split(",") if name.strip())
        else:
            required_base_names.add(row["model_name"])

    test_prob_map: dict[str, np.ndarray] = {}
    for model_name in sorted(required_base_names):
        test_prob_map[model_name] = predict_scores(base_candidates[model_name], test_df)

    final_rows: list[dict[str, Any]] = []
    winner_row = combined_results.iloc[0]
    winner_scores: np.ndarray | None = None
    winner_cv_scores: np.ndarray | None = None
    winner_threshold_table: pd.DataFrame | None = None

    for _, row in finalists.iterrows():
        if row["family"] == "blend":
            member_names = [name.strip() for name in str(row["blend_members"]).split(",") if name.strip()]
            weights = np.array(json.loads(row["blend_weights"]), dtype="float32")
            scores = np.average([test_prob_map[name] for name in member_names], axis=0, weights=weights)
            threshold_table = None
        else:
            scores = test_prob_map[row["model_name"]]
            threshold_table = base_candidates[row["model_name"]].threshold_table

        test_metrics = evaluate_scores(y_test, scores, float(row["threshold"]))
        final_rows.append({**row.to_dict(), **{f"test_{k}": v for k, v in test_metrics.items()}})

        if row["model_name"] == winner_row["model_name"]:
            winner_cv_scores = cv_prob_map[row["model_name"]]
            winner_scores = scores
            if row["family"] == "blend":
                winner_threshold_table = build_threshold_table(y_cv, cv_prob_map[row["model_name"]])
                winner_threshold_table["selection_score"] = winner_threshold_table.apply(
                    lambda r: metric_priority(r.to_dict()), axis=1
                )
            else:
                winner_threshold_table = threshold_table

    final_holdout_df = pd.DataFrame(final_rows)
    final_holdout_df.to_csv(output_dir / "final_holdout_evaluation.csv", index=False)

    if winner_scores is None or winner_cv_scores is None or winner_threshold_table is None:
        raise RuntimeError("Champion model scores were not captured.")

    best_predictions = pd.DataFrame(
        {
            "predicted_probability": winner_scores,
            "predicted_delay_flag": (winner_scores >= float(winner_row["threshold"])).astype("int8"),
            "actual_delay_flag": y_test.to_numpy(),
        }
    )
    best_predictions.to_csv(output_dir / "best_model_test_predictions.csv", index=False)

    threshold_profiles = write_deployment_threshold_profiles(
        candidate_name=str(winner_row["model_name"]),
        threshold_table=winner_threshold_table,
        y_test=y_test,
        test_scores=winner_scores,
        output_dir=output_dir,
    )
    airline_threshold_experiment = run_airline_threshold_experiment(
        cv_df=cv_df,
        y_cv=y_cv,
        cv_scores=winner_cv_scores,
        test_df=test_df,
        y_test=y_test,
        test_scores=winner_scores,
        global_threshold=float(winner_row["threshold"]),
        output_path=output_dir / "airline_threshold_experiment.csv",
    )

    test_month = pd.to_datetime(test_df["FlightDate"], errors="coerce").dt.to_period("M").astype("string").fillna("UNK")
    compute_slice_metrics(
        df=pd.DataFrame({"flight_month": test_month}),
        y_true=y_test,
        scores=winner_scores,
        threshold=float(winner_row["threshold"]),
        group_col="flight_month",
        output_path=output_dir / "monthly_holdout_metrics.csv",
    )
    compute_slice_metrics(
        df=test_df,
        y_true=y_test,
        scores=winner_scores,
        threshold=float(winner_row["threshold"]),
        group_col="Operating_Airline",
        output_path=output_dir / "airline_holdout_metrics.csv",
        top_n=20,
    )

    bundle = make_bundle_payload(winner_row, base_candidates, threshold_profiles)
    joblib.dump(bundle, output_dir / "champion_model_bundle.joblib", compress=3)

    winner_test_row = final_holdout_df.loc[final_holdout_df["model_name"] == winner_row["model_name"]].iloc[0]
    summary = {
        "mode": mode,
        "train_years": [2022, 2023],
        "cv_year": 2024,
        "test_year": 2025,
        "goal_accuracy": GOAL_ACCURACY,
        "goal_recall": GOAL_RECALL,
        "goal_precision": GOAL_PRECISION,
        "best_model": winner_row["model_name"],
        "best_threshold": float(winner_row["threshold"]),
        "best_cv_metrics": {
            key.replace("cv_", ""): float(winner_row[key])
            for key in combined_results.columns
            if key.startswith("cv_")
        },
        "best_test_metrics": {
            key.replace("test_", ""): float(winner_test_row[key])
            for key in final_holdout_df.columns
            if key.startswith("test_")
        },
        "selection_precision_floor": SELECTION_PRECISION_FLOOR,
        "selection_max_positive_rate": SELECTION_MAX_POSITIVE_RATE,
        "bundle_path": str(output_dir / "champion_model_bundle.joblib"),
        "airline_threshold_experiment_path": str(output_dir / "airline_threshold_experiment.csv"),
        "airline_threshold_overall_recall_lift": float(
            airline_threshold_experiment.loc[
                airline_threshold_experiment["Operating_Airline"] == "__overall__",
                "recall_lift_vs_global",
            ].iloc[0]
        ),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the v1.2.2 enterprise flight delay ensemble with METAR and snow-aware weather features plus recall-first thresholding."
    )
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG), default="full")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume-base-results",
        action="store_true",
        help="Reuse an existing base_model_results.csv and retrain only the top base models needed to finish blending and holdout evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "artifacts" / f"{args.mode}_{timestamp}"
    run_experiment(mode=args.mode, output_dir=output_dir, resume_base_results=args.resume_base_results)


if __name__ == "__main__":
    main()
