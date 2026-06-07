from __future__ import annotations

from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parent

RAW_DIR = PROJECT_ROOT / "RAW"

RESOURCE_DIR = EXPERIMENT_DIR / "resources"
VENDOR_DIR = EXPERIMENT_DIR / "vendor"
AIRPORTS_CSV = RESOURCE_DIR / "airports.csv"
NOTEBOOK_PATH = RESOURCE_DIR / "v4" / "flight_delay_time_safe_sampling_with_meteostat_upgraded_v4.ipynb"
WEATHER_CACHE_DIR = RESOURCE_DIR / "v4" / "weather_cache_meteostat_v4"
NOAA_DATA_DIR = RESOURCE_DIR / "fly" / "noaa_data"
LEGACY_WEATHER_HISTORY = RESOURCE_DIR / "weather" / "weather_safe_history_meteostat_hourly.parquet"
V122_AUX_RESOURCE_DIR = RESOURCE_DIR / "weather" / "v1_2_2_aux"

ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
V121_FEATURE_DIR = ARTIFACT_DIR / "features_v1_2_1_custom_5m"
V122_FEATURE_DIR = ARTIFACT_DIR / "features_v1_2_2_custom_5m"
RESULTS_DIR = EXPERIMENT_DIR / "results"
EDA_OUTPUT_DIR = ARTIFACT_DIR / "raw_eda"

TOTAL_ROWS = 5_000_000
TRAIN_ROWS = 2_750_000
CV_ROWS = 750_000
TEST_ROWS_EACH = 500_000
TEST_SPLITS = ["test_1", "test_2", "test_3"]

# Preserve the original v1.x training-year balance while meeting the 55% total train allocation.
TRAIN_ROWS_BY_YEAR = {
    2022: 1_100_000,
    2023: 1_650_000,
}
CV_YEAR = 2024
TEST_YEAR = 2025
TEST_POOL_ROWS = TEST_ROWS_EACH * len(TEST_SPLITS)

V121_FILES = {
    "train": V121_FEATURE_DIR / "train_2022_2023_2750000_weather_safe_upgraded_v11.parquet",
    "cv": V121_FEATURE_DIR / "cv_2024_750000_weather_safe_upgraded_v11.parquet",
    "test_1": V121_FEATURE_DIR / "test_2025_a_500000_weather_safe_upgraded_v11.parquet",
    "test_2": V121_FEATURE_DIR / "test_2025_b_500000_weather_safe_upgraded_v11.parquet",
    "test_3": V121_FEATURE_DIR / "test_2025_c_500000_weather_safe_upgraded_v11.parquet",
}

V122_FILES = {
    "train": V122_FEATURE_DIR / "train_2022_2023_2750000_weather_safe_upgraded_v12.parquet",
    "cv": V122_FEATURE_DIR / "cv_2024_750000_weather_safe_upgraded_v12.parquet",
    "test_1": V122_FEATURE_DIR / "test_2025_a_500000_weather_safe_upgraded_v12.parquet",
    "test_2": V122_FEATURE_DIR / "test_2025_b_500000_weather_safe_upgraded_v12.parquet",
    "test_3": V122_FEATURE_DIR / "test_2025_c_500000_weather_safe_upgraded_v12.parquet",
}

SPLIT_ORDER = ["train", "cv", *TEST_SPLITS]

QUICK_TOTAL_ROWS = 60_000
QUICK_TRAIN_ROWS_BY_YEAR = {
    2022: 13_200,
    2023: 19_800,
}
QUICK_CV_ROWS = 9_000
QUICK_TEST_ROWS_EACH = 6_000
QUICK_TEST_POOL_ROWS = QUICK_TEST_ROWS_EACH * len(TEST_SPLITS)

QUICK_V121_FEATURE_DIR = ARTIFACT_DIR / "features_v1_2_1_quick"
QUICK_V122_FEATURE_DIR = ARTIFACT_DIR / "features_v1_2_2_quick"
QUICK_RESULTS_DIR = EXPERIMENT_DIR / "results_quick"

QUICK_V121_FILES = {
    "train": QUICK_V121_FEATURE_DIR / "train_2022_2023_33000_weather_safe_upgraded_v11.parquet",
    "cv": QUICK_V121_FEATURE_DIR / "cv_2024_9000_weather_safe_upgraded_v11.parquet",
    "test_1": QUICK_V121_FEATURE_DIR / "test_2025_a_6000_weather_safe_upgraded_v11.parquet",
    "test_2": QUICK_V121_FEATURE_DIR / "test_2025_b_6000_weather_safe_upgraded_v11.parquet",
    "test_3": QUICK_V121_FEATURE_DIR / "test_2025_c_6000_weather_safe_upgraded_v11.parquet",
}

QUICK_V122_FILES = {
    "train": QUICK_V122_FEATURE_DIR / "train_2022_2023_33000_weather_safe_upgraded_v12.parquet",
    "cv": QUICK_V122_FEATURE_DIR / "cv_2024_9000_weather_safe_upgraded_v12.parquet",
    "test_1": QUICK_V122_FEATURE_DIR / "test_2025_a_6000_weather_safe_upgraded_v12.parquet",
    "test_2": QUICK_V122_FEATURE_DIR / "test_2025_b_6000_weather_safe_upgraded_v12.parquet",
    "test_3": QUICK_V122_FEATURE_DIR / "test_2025_c_6000_weather_safe_upgraded_v12.parquet",
}


def preset_config(preset: str) -> dict[str, object]:
    if preset == "quick":
        return {
            "preset": "quick",
            "v121_feature_dir": QUICK_V121_FEATURE_DIR,
            "v122_feature_dir": QUICK_V122_FEATURE_DIR,
            "v121_files": QUICK_V121_FILES,
            "v122_files": QUICK_V122_FILES,
            "results_dir": QUICK_RESULTS_DIR,
            "train_rows_by_year": QUICK_TRAIN_ROWS_BY_YEAR,
            "cv_rows": QUICK_CV_ROWS,
            "test_rows_each": QUICK_TEST_ROWS_EACH,
            "test_pool_rows": QUICK_TEST_POOL_ROWS,
            "model_mode": "quick",
        }
    if preset == "full":
        return {
            "preset": "full",
            "v121_feature_dir": V121_FEATURE_DIR,
            "v122_feature_dir": V122_FEATURE_DIR,
            "v121_files": V121_FILES,
            "v122_files": V122_FILES,
            "results_dir": RESULTS_DIR,
            "train_rows_by_year": TRAIN_ROWS_BY_YEAR,
            "cv_rows": CV_ROWS,
            "test_rows_each": TEST_ROWS_EACH,
            "test_pool_rows": TEST_POOL_ROWS,
            "model_mode": "full",
        }
    raise ValueError(f"Unknown preset: {preset}")
