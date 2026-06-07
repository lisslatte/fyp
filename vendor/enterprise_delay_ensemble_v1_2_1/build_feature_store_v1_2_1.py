from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import zlib
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "v4" / "flight_delay_time_safe_sampling_with_meteostat_upgraded_v4.ipynb"
ARTIFACT_DIR = PROJECT_ROOT / "weather_safe_model_artifacts_meteostat_upgraded_v4_v1_2_1"
WEATHER_CACHE_DIR = PROJECT_ROOT / "v4" / "weather_cache_meteostat_v4"
LEGACY_WEATHER_HISTORY_CANDIDATES = [
    PROJECT_ROOT / "v2" / "weather_safe_model_artifacts_meteostat_upgraded" / "weather_safe_history_meteostat_hourly.parquet",
    PROJECT_ROOT / "v4" / "weather_safe_model_artifacts_meteostat_upgraded_v4" / "weather_safe_history_meteostat_hourly.parquet",
    PROJECT_ROOT / "v4.1" / "weather_safe_model_artifacts_meteostat_upgraded_v4.1" / "weather_safe_history_meteostat_hourly.parquet",
]

TRAIN_YEARS = [2022, 2023]
CV_YEAR = 2024
TEST_YEAR = 2025
SAMPLE_ROWS_BY_YEAR = {
    2022: 1_000_000,
    2023: 1_500_000,
    2024: 250_000,
    2025: 250_000,
}

OUTPUT_FILES = [
    ARTIFACT_DIR / f"train_{'_'.join(map(str, TRAIN_YEARS))}_{sum(SAMPLE_ROWS_BY_YEAR[y] for y in TRAIN_YEARS)}_weather_safe_upgraded_v11.parquet",
    ARTIFACT_DIR / f"cv_{CV_YEAR}_{SAMPLE_ROWS_BY_YEAR[CV_YEAR]}_weather_safe_upgraded_v11.parquet",
    ARTIFACT_DIR / f"test_{TEST_YEAR}_{SAMPLE_ROWS_BY_YEAR[TEST_YEAR]}_weather_safe_upgraded_v11.parquet",
]
VALIDATION_COLUMNS = ["Year", "FlightDate", "Month", "Origin", "Dest", "depdelay15"]
WEATHER_HISTORY_COLUMNS = [
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
]
WEATHER_FEATURE_COLUMNS = [
    "origin_recent_prcp_3h",
    "origin_recent_prcp_6h",
    "origin_recent_wspd_6h_mean",
    "origin_recent_weather_risk_6h",
    "origin_recent_weather_risk_12h",
    "origin_weather_signal",
    "weather_data_missing_origin",
]
WEATHER_VARIANCE_COLUMNS = [
    "origin_recent_prcp_3h",
    "origin_recent_wspd_6h_mean",
    "origin_recent_weather_risk_6h",
    "origin_recent_weather_risk_12h",
    "origin_weather_signal",
]
MIN_WEATHER_HISTORY_ROWS = 100_000
MIN_WEATHER_AIRPORTS = 100
MIN_WEATHER_TIMESTAMPS = 1_000
MIN_WEATHER_VARIANCE_COLUMNS = 2
MAX_WEATHER_MISSING_RATE = 0.98


def weather_history_path() -> Path:
    return ARTIFACT_DIR / "weather_safe_history_meteostat_hourly.parquet"


def validate_weather_history_frame(weather_df: pd.DataFrame, source: str) -> dict[str, Any]:
    missing_cols = [col for col in WEATHER_HISTORY_COLUMNS if col not in weather_df.columns]
    if missing_cols:
        raise ValueError(f"{source} is missing required weather columns: {missing_cols}")
    if weather_df.empty or len(weather_df) < MIN_WEATHER_HISTORY_ROWS:
        raise ValueError(
            f"{source} does not contain enough weather rows. "
            f"Expected at least {MIN_WEATHER_HISTORY_ROWS:,}, found {len(weather_df):,}."
        )

    airport_count = int(weather_df["airport"].astype("string").nunique(dropna=True))
    timestamp_count = int(pd.to_datetime(weather_df["weather_ts"], errors="coerce").nunique(dropna=True))
    varying_cols = [
        col
        for col in WEATHER_HISTORY_COLUMNS
        if col not in {"airport", "weather_ts"} and weather_df[col].nunique(dropna=False) > 1
    ]

    if airport_count < MIN_WEATHER_AIRPORTS:
        raise ValueError(f"{source} only contains {airport_count} airports; expected at least {MIN_WEATHER_AIRPORTS}.")
    if timestamp_count < MIN_WEATHER_TIMESTAMPS:
        raise ValueError(
            f"{source} only contains {timestamp_count} distinct weather timestamps; "
            f"expected at least {MIN_WEATHER_TIMESTAMPS}."
        )
    if len(varying_cols) < MIN_WEATHER_VARIANCE_COLUMNS:
        raise ValueError(
            f"{source} has too little weather variation. "
            f"Only {len(varying_cols)} weather columns vary: {varying_cols}."
        )

    return {
        "rows": int(len(weather_df)),
        "airport_count": airport_count,
        "timestamp_count": timestamp_count,
        "varying_weather_cols": varying_cols,
    }


def validate_weather_history_path(path: Path) -> dict[str, Any]:
    weather_df = pd.read_parquet(path, columns=WEATHER_HISTORY_COLUMNS)
    stats = validate_weather_history_frame(weather_df, source=str(path))
    stats["path"] = str(path)
    return stats


def validate_weather_feature_frame(df: pd.DataFrame, source: str) -> dict[str, Any]:
    missing_cols = [col for col in WEATHER_FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{source} is missing engineered weather columns: {missing_cols}")

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
            "This indicates the weather join failed or an empty weather cache was reused."
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


def find_valid_legacy_weather_history() -> Path | None:
    for candidate in LEGACY_WEATHER_HISTORY_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            stats = validate_weather_history_path(candidate)
        except Exception as exc:  # noqa: BLE001
            print(f"[v1.2.1 feature-store] skipping invalid legacy weather cache {candidate}: {exc}")
            continue
        print(
            "[v1.2.1 feature-store] found valid legacy weather cache "
            f"{candidate} with {stats['rows']:,} rows across {stats['airport_count']} airports"
        )
        return candidate
    return None


def execute_notebook_cells(scope: dict[str, Any], start_idx: int = 2, end_idx: int = 8) -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    os.chdir(PROJECT_ROOT)
    for idx in range(start_idx, end_idx + 1):
        cell = nb.cells[idx]
        if cell.cell_type != "code":
            continue
        print(f"[v1.2.1 feature-store] executing notebook cell {idx}")
        try:
            exec(compile(cell.source, f"{NOTEBOOK_PATH.name}:cell_{idx}", "exec"), scope)
        except Exception as exc:
            raise RuntimeError(f"Notebook cell {idx} failed during v1.2.1 feature-store build") from exc


def month_target_stratified_sample(df: pd.DataFrame, sample_size: int, seed: int, target_col: str) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    work = df.copy()
    flight_month = pd.to_datetime(work["FlightDate"], errors="coerce").dt.to_period("M").astype("string").fillna("UNK")
    work["_sampling_stratum"] = flight_month + "|" + work[target_col].astype("string")

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


def seed_scope(scope: dict[str, Any]) -> None:
    scope["ARTIFACT_DIR"] = ARTIFACT_DIR
    scope["WEATHER_CACHE_DIR"] = WEATHER_CACHE_DIR
    scope["TRAIN_YEAR"] = TRAIN_YEARS[0]
    scope["CV_YEAR"] = CV_YEAR
    scope["TEST_YEAR"] = TEST_YEAR
    scope["SAMPLE_ROWS_BY_YEAR"] = SAMPLE_ROWS_BY_YEAR.copy()
    scope["WEATHER_START_DATE"] = "2022-12-20"
    scope["WEATHER_END_DATE"] = "2025-12-31"

    holiday_map = dict(scope["MAJOR_TRAVEL_HOLIDAYS"])
    holiday_map[2022] = [
        "2022-01-01",
        "2022-01-17",
        "2022-05-30",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2022-12-24",
        "2022-12-25",
        "2022-12-31",
    ]
    scope["MAJOR_TRAVEL_HOLIDAYS"] = holiday_map

    # Override the notebook's label-only sampler with a month-aware variant.
    def stable_sample_by_target(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
        return month_target_stratified_sample(df=df, sample_size=sample_size, seed=seed, target_col=scope["TARGET_COL"])

    scope["stable_sample_by_target"] = stable_sample_by_target


def ensure_weather_history(force_rebuild: bool = False) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = weather_history_path()

    if force_rebuild and target.exists():
        print(f"[v1.2.1 feature-store] removing existing weather cache before rebuild: {target}")
        target.unlink()

    if target.exists():
        try:
            stats = validate_weather_history_path(target)
        except Exception as exc:  # noqa: BLE001
            print(f"[v1.2.1 feature-store] removing invalid weather cache {target}: {exc}")
            target.unlink()
        else:
            print(
                "[v1.2.1 feature-store] using existing validated weather cache "
                f"{target} with {stats['rows']:,} rows"
            )
            return

    legacy = find_valid_legacy_weather_history()
    if legacy is not None:
        shutil.copy2(legacy, target)
        stats = validate_weather_history_path(target)
        print(
            "[v1.2.1 feature-store] copied validated weather cache "
            f"{legacy} -> {target} ({stats['rows']:,} rows)"
        )
        return

    print(
        "[v1.2.1 feature-store] no validated legacy weather cache was found. "
        "The notebook will rebuild Meteostat weather history from source."
    )


def build_feature_store(scope: dict[str, Any]) -> pd.DataFrame:
    load_year_sample = scope["load_year_sample"]
    build_safe_weather_history = scope["build_safe_weather_history"]
    build_train_features = scope["build_train_features"]
    build_reference_maps = scope["build_reference_maps"]
    apply_reference_features = scope["apply_reference_features"]

    train_frames = []
    for year in TRAIN_YEARS:
        print(f"[v1.2.1 feature-store] loading training year {year}")
        train_frames.append(load_year_sample(year, SAMPLE_ROWS_BY_YEAR[year]))
        gc.collect()
    train_raw = pd.concat(train_frames, ignore_index=True, sort=False)
    train_raw = train_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[v1.2.1 feature-store] loading cv year {CV_YEAR}")
    cv_raw = load_year_sample(CV_YEAR, SAMPLE_ROWS_BY_YEAR[CV_YEAR])
    print(f"[v1.2.1 feature-store] loading test year {TEST_YEAR}")
    test_raw = load_year_sample(TEST_YEAR, SAMPLE_ROWS_BY_YEAR[TEST_YEAR])

    summary = pd.DataFrame(
        [
            {"split": "train", "years": ",".join(map(str, TRAIN_YEARS)), "rows": len(train_raw), "delay_rate": float(train_raw[scope["TARGET_COL"]].mean())},
            {"split": "cv", "years": str(CV_YEAR), "rows": len(cv_raw), "delay_rate": float(cv_raw[scope["TARGET_COL"]].mean())},
            {"split": "test", "years": str(TEST_YEAR), "rows": len(test_raw), "delay_rate": float(test_raw[scope["TARGET_COL"]].mean())},
        ]
    )
    summary.to_csv(ARTIFACT_DIR / "dataset_summary.csv", index=False)

    weather_safe = build_safe_weather_history([train_raw, cv_raw, test_raw])
    weather_stats = validate_weather_history_frame(weather_safe, source="v1.2.1 weather history")
    print(
        "[v1.2.1 feature-store] weather history validated: "
        f"rows={weather_stats['rows']:,}, airports={weather_stats['airport_count']}, "
        f"timestamps={weather_stats['timestamp_count']}"
    )

    train_df = build_train_features(train_raw, weather_safe)
    train_weather_stats = validate_weather_feature_frame(train_df, source="v1.2.1 train features")
    refs = build_reference_maps(train_df)

    cv_df = apply_reference_features(cv_raw, refs, weather_safe, history_df=train_df)
    cv_weather_stats = validate_weather_feature_frame(cv_df, source="v1.2.1 cv features")
    history_for_test = pd.concat([train_df, cv_df], ignore_index=True, sort=False)
    test_df = apply_reference_features(test_raw, refs, weather_safe, history_df=history_for_test)
    test_weather_stats = validate_weather_feature_frame(test_df, source="v1.2.1 test features")

    train_df.to_parquet(OUTPUT_FILES[0], index=False)
    cv_df.to_parquet(OUTPUT_FILES[1], index=False)
    test_df.to_parquet(OUTPUT_FILES[2], index=False)

    manifest = {
        "train_years": TRAIN_YEARS,
        "cv_year": CV_YEAR,
        "test_year": TEST_YEAR,
        "sample_rows_by_year": SAMPLE_ROWS_BY_YEAR,
        "output_files": [str(path) for path in OUTPUT_FILES],
        "weather_history_path": str(weather_history_path()),
        "weather_history_rows": weather_stats["rows"],
        "train_weather_missing_rate": train_weather_stats["weather_missing_rate"],
        "cv_weather_missing_rate": cv_weather_stats["weather_missing_rate"],
        "test_weather_missing_rate": test_weather_stats["weather_missing_rate"],
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (ARTIFACT_DIR / "feature_store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def validate_outputs() -> None:
    read_columns = list(dict.fromkeys(VALIDATION_COLUMNS + WEATHER_FEATURE_COLUMNS))
    for path in OUTPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Expected v1.2.1 parquet missing: {path}")
        df = pd.read_parquet(path, columns=read_columns)
        weather_stats = validate_weather_feature_frame(df, source=path.name)
        print(
            "[validate] "
            f"{path.name}: {df.shape}, weather_missing_rate={weather_stats['weather_missing_rate']:.2%}, "
            f"active_weather_cols={weather_stats['active_weather_cols'][:4]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v1.2.1 weather-safe feature store with weather validation and non-empty weather guarantees."
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip parquet read-back validation after build.",
    )
    parser.add_argument(
        "--force-rebuild-weather-history",
        action="store_true",
        help="Delete any existing v1.2.1 weather history cache and rebuild or recopy a validated one.",
    )
    args = parser.parse_args()

    scope: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
        "display": lambda *args, **kwargs: None,
    }
    execute_notebook_cells(scope)
    seed_scope(scope)
    ensure_weather_history(force_rebuild=args.force_rebuild_weather_history)
    build_feature_store(scope)
    if not args.skip_validate:
        validate_outputs()
    print("[v1.2.1 feature-store] completed")


if __name__ == "__main__":
    main()
