from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    ARTIFACT_DIR,
    AIRPORTS_CSV,
    CV_YEAR,
    EXPERIMENT_DIR,
    LEGACY_WEATHER_HISTORY,
    NOTEBOOK_PATH,
    RAW_DIR,
    TEST_SPLITS,
    TEST_YEAR,
    VENDOR_DIR,
    WEATHER_CACHE_DIR,
    preset_config,
)


ORIGINAL_V121_DIR = VENDOR_DIR / "enterprise_delay_ensemble_v1_2_1"
if str(ORIGINAL_V121_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_V121_DIR))

import build_feature_store_v1_2_1 as v121  # noqa: E402


def patch_original_module(cfg: dict[str, Any]) -> None:
    train_rows_by_year = cfg["train_rows_by_year"]
    v121_feature_dir = cfg["v121_feature_dir"]
    v121_files = cfg["v121_files"]

    v121.PROJECT_ROOT = EXPERIMENT_DIR
    v121.NOTEBOOK_PATH = NOTEBOOK_PATH
    v121.WEATHER_CACHE_DIR = WEATHER_CACHE_DIR
    v121.LEGACY_WEATHER_HISTORY_CANDIDATES = [LEGACY_WEATHER_HISTORY]
    v121.ARTIFACT_DIR = v121_feature_dir
    v121.TRAIN_YEARS = sorted(train_rows_by_year)
    v121.CV_YEAR = CV_YEAR
    v121.TEST_YEAR = TEST_YEAR
    v121.SAMPLE_ROWS_BY_YEAR = {
        **train_rows_by_year,
        CV_YEAR: cfg["cv_rows"],
        TEST_YEAR: cfg["test_pool_rows"],
    }
    v121.OUTPUT_FILES = [
        v121_files["train"],
        v121_files["cv"],
        v121_files["test_1"],
        v121_files["test_2"],
        v121_files["test_3"],
    ]


def split_test_pool(test_pool: pd.DataFrame, test_pool_rows: int, test_rows_each: int) -> dict[str, pd.DataFrame]:
    if len(test_pool) < test_pool_rows:
        raise ValueError(f"Expected {test_pool_rows:,} test-pool rows, found {len(test_pool):,}.")
    shuffled = test_pool.sample(frac=1.0, random_state=20251225).reset_index(drop=True)
    return {
        split_name: shuffled.iloc[idx * test_rows_each : (idx + 1) * test_rows_each].copy().reset_index(drop=True)
        for idx, split_name in enumerate(TEST_SPLITS)
    }


def write_split_summary(raw_splits: dict[str, pd.DataFrame], target_col: str, v121_feature_dir: Path) -> None:
    rows = []
    for split_name, df in raw_splits.items():
        flight_dates = pd.to_datetime(df["FlightDate"], errors="coerce")
        rows.append(
            {
                "split": split_name,
                "rows": int(len(df)),
                "year_min": int(pd.to_numeric(df["Year"], errors="coerce").min()),
                "year_max": int(pd.to_numeric(df["Year"], errors="coerce").max()),
                "date_min": str(flight_dates.min().date()) if flight_dates.notna().any() else "",
                "date_max": str(flight_dates.max().date()) if flight_dates.notna().any() else "",
                "delay_rate": float(pd.to_numeric(df[target_col], errors="coerce").mean()),
            }
        )
    pd.DataFrame(rows).to_csv(v121_feature_dir / "raw_split_summary.csv", index=False)


def build_feature_store(preset: str, force_rebuild_weather_history: bool = False, skip_validate: bool = False) -> dict[str, Any]:
    cfg = preset_config(preset)
    patch_original_module(cfg)
    v121_feature_dir: Path = cfg["v121_feature_dir"]  # type: ignore[assignment]
    v121_files: dict[str, Path] = cfg["v121_files"]  # type: ignore[assignment]
    train_rows_by_year: dict[int, int] = cfg["train_rows_by_year"]  # type: ignore[assignment]
    cv_rows = int(cfg["cv_rows"])
    test_rows_each = int(cfg["test_rows_each"])
    test_pool_rows = int(cfg["test_pool_rows"])

    v121_feature_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    scope: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(v121.NOTEBOOK_PATH),
        "display": lambda *args, **kwargs: None,
    }

    v121.execute_notebook_cells(scope)
    v121.seed_scope(scope)
    scope["BASE_DIR"] = EXPERIMENT_DIR
    scope["RAW_DIR"] = RAW_DIR
    scope["AIRPORTS_CSV"] = AIRPORTS_CSV
    scope["WEATHER_CACHE_DIR"] = WEATHER_CACHE_DIR
    scope["ARTIFACT_DIR"] = v121_feature_dir
    if "ms" in scope:
        scope["ms"].config.cache_directory = str(WEATHER_CACHE_DIR)
    v121.ensure_weather_history(force_rebuild=force_rebuild_weather_history)

    load_year_sample = scope["load_year_sample"]
    build_safe_weather_history = scope["build_safe_weather_history"]
    build_train_features = scope["build_train_features"]
    build_reference_maps = scope["build_reference_maps"]
    apply_reference_features = scope["apply_reference_features"]
    target_col = scope["TARGET_COL"]

    train_frames = []
    for year, rows in train_rows_by_year.items():
        print(f"[custom v1.2.1] loading train year {year} with {rows:,} rows")
        train_frames.append(load_year_sample(year, rows))
        gc.collect()
    train_raw = pd.concat(train_frames, ignore_index=True, sort=False)
    train_raw = train_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[custom v1.2.1] loading cv year {CV_YEAR} with {cv_rows:,} rows")
    cv_raw = load_year_sample(CV_YEAR, cv_rows)

    print(f"[custom v1.2.1] loading {TEST_YEAR} test pool with {test_pool_rows:,} rows")
    test_pool_raw = load_year_sample(TEST_YEAR, test_pool_rows)
    raw_test_splits = split_test_pool(test_pool_raw, test_pool_rows=test_pool_rows, test_rows_each=test_rows_each)

    raw_splits = {"train": train_raw, "cv": cv_raw, **raw_test_splits}
    write_split_summary(raw_splits, target_col=target_col, v121_feature_dir=v121_feature_dir)

    weather_safe = build_safe_weather_history([train_raw, cv_raw, test_pool_raw])
    weather_stats = v121.validate_weather_history_frame(weather_safe, source="custom v1.2.1 weather history")

    print("[custom v1.2.1] building train features")
    train_df = build_train_features(train_raw, weather_safe)
    train_weather_stats = v121.validate_weather_feature_frame(train_df, source="custom train features")
    refs = build_reference_maps(train_df)

    print("[custom v1.2.1] building cv features")
    cv_df = apply_reference_features(cv_raw, refs, weather_safe, history_df=train_df)
    cv_weather_stats = v121.validate_weather_feature_frame(cv_df, source="custom cv features")

    history_for_tests = pd.concat([train_df, cv_df], ignore_index=True, sort=False)
    output_frames = {"train": train_df, "cv": cv_df}
    split_stats: dict[str, Any] = {
        "train": train_weather_stats,
        "cv": cv_weather_stats,
    }

    for split_name, raw_df in raw_test_splits.items():
        print(f"[custom v1.2.1] building {split_name} features")
        featured = apply_reference_features(raw_df, refs, weather_safe, history_df=history_for_tests)
        split_stats[split_name] = v121.validate_weather_feature_frame(featured, source=f"custom {split_name} features")
        output_frames[split_name] = featured
        gc.collect()

    for split_name, frame in output_frames.items():
        frame.to_parquet(v121_files[split_name], index=False)
        print(f"[custom v1.2.1] wrote {split_name}: {v121_files[split_name]}")

    source_weather = v121.weather_history_path()
    target_weather = v121_feature_dir / "weather_safe_history_meteostat_hourly.parquet"
    if source_weather.exists() and source_weather != target_weather:
        shutil.copy2(source_weather, target_weather)

    manifest = {
        "builder": "classifier_v1_2_2_original_3m/src/build_v121_feature_store_3m.py",
        "preset": preset,
        "train_rows_by_year": train_rows_by_year,
        "cv_year": CV_YEAR,
        "cv_rows": cv_rows,
        "test_year": TEST_YEAR,
        "test_rows_each": test_rows_each,
        "test_splits": TEST_SPLITS,
        "output_files": {name: str(path) for name, path in v121_files.items()},
        "weather_history_path": str(target_weather),
        "weather_history_rows": weather_stats["rows"],
        "split_weather_stats": split_stats,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (v121_feature_dir / "feature_store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not skip_validate:
        validate_outputs(preset)
    return manifest


def validate_outputs(preset: str) -> None:
    cfg = preset_config(preset)
    v121_files: dict[str, Path] = cfg["v121_files"]  # type: ignore[assignment]
    read_columns = list(dict.fromkeys(v121.VALIDATION_COLUMNS + v121.WEATHER_FEATURE_COLUMNS))
    for split_name, path in v121_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected custom v1.2.1 parquet missing: {path}")
        df = pd.read_parquet(path, columns=read_columns)
        stats = v121.validate_weather_feature_frame(df, source=f"{split_name} ({path.name})")
        print(f"[custom v1.2.1 validate] {split_name}: rows={len(df):,}, weather_missing={stats['weather_missing_rate']:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build custom full/quick v1.2.1 feature splits from RAW data.")
    parser.add_argument("--preset", choices=["full", "quick"], default="full")
    parser.add_argument("--force-rebuild-weather-history", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_feature_store(
        preset=args.preset,
        force_rebuild_weather_history=args.force_rebuild_weather_history,
        skip_validate=args.skip_validate,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
