from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from typing import Any

import pandas as pd

from config import AIRPORTS_CSV, NOAA_DATA_DIR, SPLIT_ORDER, V122_AUX_RESOURCE_DIR, VENDOR_DIR, preset_config


ORIGINAL_V122_DIR = VENDOR_DIR / "enterprise_delay_ensemble_v1_2_2"
if str(ORIGINAL_V122_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_V122_DIR))

import build_feature_store_v1_2_2 as v122  # noqa: E402


V122_AUX_RESOURCE_FILES = {
    "airport_metadata_v1_2_2.csv": "airport_metadata_v1_2_2.csv",
    "noaa_station_inventory_v1_2_2.csv": "noaa_station_inventory_v1_2_2.csv",
    "enhanced_weather_history_v1_2_2.parquet": "enhanced_weather_history_v1_2_2.parquet",
    "metar_hourly_features_v1_2_2.parquet": "metar_hourly_features_v1_2_2.parquet",
    "station_daily_snow_features_v1_2_2.parquet": "station_daily_snow_features_v1_2_2.parquet",
    "ghcnd_stations_v1_2_2.txt": "ghcnd_stations_v1_2_2.txt",
    "ghcnd_inventory_v1_2_2.txt": "ghcnd_inventory_v1_2_2.txt",
    "ghcnd_station_summary_v1_2_2.csv": "ghcnd_station_summary_v1_2_2.csv",
    "ghcnd_daily_features_v1_2_2.parquet": "ghcnd_daily_features_v1_2_2.parquet",
}


def seed_auxiliary_weather_resources(v122_feature_dir: Any, force_rebuild_aux: bool) -> None:
    if force_rebuild_aux:
        return
    v122_feature_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in V122_AUX_RESOURCE_FILES.items():
        source = V122_AUX_RESOURCE_DIR / source_name
        if source.exists():
            shutil.copy2(source, v122_feature_dir / target_name)

    source_raw_dir = V122_AUX_RESOURCE_DIR / "ghcnd_daily_raw_v1_2_2"
    target_raw_dir = v122_feature_dir / "ghcnd_daily_raw_v1_2_2"
    if source_raw_dir.exists():
        if target_raw_dir.exists():
            shutil.rmtree(target_raw_dir)
        shutil.copytree(source_raw_dir, target_raw_dir)


def patch_original_module(cfg: dict[str, Any]) -> None:
    v121_feature_dir = cfg["v121_feature_dir"]
    v122_feature_dir = cfg["v122_feature_dir"]
    v121_files = cfg["v121_files"]
    v122_files = cfg["v122_files"]

    v122.SOURCE_CACHE_DIR = v121_feature_dir
    v122.ARTIFACT_DIR = v122_feature_dir
    v122.AIRPORTS_CSV = AIRPORTS_CSV
    v122.NOAA_DATA_DIR = NOAA_DATA_DIR
    v122.SOURCE_WEATHER_HISTORY = v121_feature_dir / "weather_safe_history_meteostat_hourly.parquet"
    v122.GHCN_RAW_DIR = v122_feature_dir / "ghcnd_daily_raw_v1_2_2"
    v122.SOURCE_FILES = dict(v121_files)
    v122.OUTPUT_FILES = dict(v122_files)
    v122.AUX_FILES = {
        "airport_metadata": v122_feature_dir / "airport_metadata_v1_2_2.csv",
        "station_inventory": v122_feature_dir / "noaa_station_inventory_v1_2_2.csv",
        "hourly_station_mapping": v122_feature_dir / "hourly_station_mapping_v1_2_2.csv",
        "enhanced_weather_history": v122_feature_dir / "enhanced_weather_history_v1_2_2.parquet",
        "metar_hourly": v122_feature_dir / "metar_hourly_features_v1_2_2.parquet",
        "station_daily_snow": v122_feature_dir / "station_daily_snow_features_v1_2_2.parquet",
        "origin_airport_month_norms": v122_feature_dir / "origin_airport_month_weather_norms_v1_2_2.csv",
        "ghcn_stations_text": v122_feature_dir / "ghcnd_stations_v1_2_2.txt",
        "ghcn_inventory_text": v122_feature_dir / "ghcnd_inventory_v1_2_2.txt",
        "ghcn_station_summary": v122_feature_dir / "ghcnd_station_summary_v1_2_2.csv",
        "ghcn_snow_mapping": v122_feature_dir / "ghcnd_snow_mapping_v1_2_2.csv",
        "ghcn_snwd_mapping": v122_feature_dir / "ghcnd_snwd_mapping_v1_2_2.csv",
        "ghcn_daily_features": v122_feature_dir / "ghcnd_daily_features_v1_2_2.parquet",
    }


def build_feature_store(preset: str, force_rebuild_aux: bool = False, skip_validate: bool = False) -> dict[str, Any]:
    cfg = preset_config(preset)
    patch_original_module(cfg)
    v121_feature_dir = cfg["v121_feature_dir"]
    v122_feature_dir = cfg["v122_feature_dir"]
    v121_files = cfg["v121_files"]
    v122_files = cfg["v122_files"]

    v122.ensure_source_feature_store()
    v122_feature_dir.mkdir(parents=True, exist_ok=True)
    seed_auxiliary_weather_resources(v122_feature_dir, force_rebuild_aux=force_rebuild_aux)

    used_airports = v122.collect_used_airports()
    airport_meta = v122.load_airport_metadata(used_airports)
    station_inventory = v122.build_noaa_station_inventory(force_rebuild=force_rebuild_aux)
    ghcn_station_summary = v122.build_ghcn_station_summary(force_rebuild=force_rebuild_aux)

    hourly_mapping = v122.map_airports_to_points(
        airport_meta,
        station_inventory[["station_id", "lat", "lon"]].copy(),
        candidate_id_col="station_id",
        max_distance_km=v122.HOURLY_STATION_MAX_DISTANCE_KM,
    )
    snowfall_mapping = v122.map_airports_to_points(
        airport_meta,
        ghcn_station_summary.loc[ghcn_station_summary["has_snow"] == 1, ["station_id", "lat", "lon"]].copy(),
        candidate_id_col="station_id",
        max_distance_km=v122.GHCN_DAILY_MAX_DISTANCE_KM,
    )
    snowdepth_mapping = v122.map_airports_to_points(
        airport_meta,
        ghcn_station_summary.loc[ghcn_station_summary["has_snwd"] == 1, ["station_id", "lat", "lon"]].copy(),
        candidate_id_col="station_id",
        max_distance_km=v122.GHCN_DAILY_MAX_DISTANCE_KM,
    )

    hourly_mapping.to_csv(v122.AUX_FILES["hourly_station_mapping"], index=False)
    snowfall_mapping.to_csv(v122.AUX_FILES["ghcn_snow_mapping"], index=False)
    snowdepth_mapping.to_csv(v122.AUX_FILES["ghcn_snwd_mapping"], index=False)

    enhanced_weather_history = v122.build_enhanced_weather_history(force_rebuild=force_rebuild_aux)
    metar_hourly, _ = v122.build_metar_hourly_and_daily_features(
        station_inventory=station_inventory,
        hourly_station_ids={
            station_id
            for station_id in (v122.normalize_station_id(value) for value in hourly_mapping["station_id"].tolist())
            if station_id
        },
        daily_station_ids=set(),
        force_rebuild=force_rebuild_aux,
    )
    ghcn_daily = v122.build_ghcn_daily_features(
        station_ids={
            station_id
            for station_id in (
                v122.normalize_station_id(value)
                for value in (snowfall_mapping["station_id"].tolist() + snowdepth_mapping["station_id"].tolist())
            )
            if station_id
        },
        force_rebuild=force_rebuild_aux,
    )

    split_stats: dict[str, Any] = {}

    print("[custom v1.2.2] augmenting train")
    train_base = pd.read_parquet(v121_files["train"])
    train_augmented = v122.augment_split(
        base_df=train_base,
        enhanced_weather_history=enhanced_weather_history,
        metar_hourly=metar_hourly,
        ghcn_daily=ghcn_daily,
        hourly_mapping=hourly_mapping,
        snowfall_mapping=snowfall_mapping,
        snowdepth_mapping=snowdepth_mapping,
    )
    origin_norms = v122.build_origin_airport_month_norms(train_augmented)
    train_augmented = v122.apply_origin_airport_month_norms(train_augmented, origin_norms)
    split_stats["train"] = v122.validate_output_frame(train_augmented, source="custom train v1.2.2 feature cache")
    train_augmented.to_parquet(v122_files["train"], index=False)
    del train_base, train_augmented
    gc.collect()

    for split_name in [name for name in SPLIT_ORDER if name != "train"]:
        print(f"[custom v1.2.2] augmenting {split_name}")
        base_df = pd.read_parquet(v121_files[split_name])
        augmented = v122.augment_split(
            base_df=base_df,
            enhanced_weather_history=enhanced_weather_history,
            metar_hourly=metar_hourly,
            ghcn_daily=ghcn_daily,
            hourly_mapping=hourly_mapping,
            snowfall_mapping=snowfall_mapping,
            snowdepth_mapping=snowdepth_mapping,
        )
        augmented = v122.apply_origin_airport_month_norms(augmented, origin_norms)
        split_stats[split_name] = v122.validate_output_frame(augmented, source=f"custom {split_name} v1.2.2 feature cache")
        augmented.to_parquet(v122_files[split_name], index=False)
        del base_df, augmented
        gc.collect()

    manifest = {
        "builder": "classifier_v1_2_2_original_3m/src/build_v122_feature_store_3m.py",
        "preset": preset,
        "source_cache_dir": str(v121_feature_dir),
        "output_cache_dir": str(v122_feature_dir),
        "used_airports": int(len(used_airports)),
        "hourly_station_airport_coverage": int(hourly_mapping["has_mapping"].sum()),
        "ghcn_snow_airport_coverage": int(snowfall_mapping["has_mapping"].sum()),
        "ghcn_snow_depth_airport_coverage": int(snowdepth_mapping["has_mapping"].sum()),
        "metar_hourly_rows": int(len(metar_hourly)),
        "ghcn_daily_rows": int(len(ghcn_daily)),
        "output_files": {name: str(path) for name, path in v122_files.items()},
        "split_stats": split_stats,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (v122_feature_dir / "feature_store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not skip_validate:
        validate_outputs(preset)
    return manifest


def validate_outputs(preset: str) -> None:
    cfg = preset_config(preset)
    patch_original_module(cfg)
    v122_files = cfg["v122_files"]
    validation_columns = v122.VALIDATION_COLUMNS + [
        "origin_metar_data_missing",
        "origin_snow_data_missing",
        "dest_metar_data_missing",
        "dest_snow_data_missing",
    ]
    for split_name, path in v122_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected custom v1.2.2 parquet missing: {path}")
        df = pd.read_parquet(path, columns=validation_columns)
        stats = v122.validate_output_frame(df, source=f"{split_name} output validation")
        print(
            f"[custom v1.2.2 validate] {split_name}: rows={len(df):,}, "
            f"origin_metar_missing={stats['origin_metar_missing_rate']:.2%}, "
            f"origin_snow_missing={stats['origin_snow_missing_rate']:.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build custom full/quick v1.2.2 feature splits.")
    parser.add_argument("--preset", choices=["full", "quick"], default="full")
    parser.add_argument("--force-rebuild-aux", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_feature_store(preset=args.preset, force_rebuild_aux=args.force_rebuild_aux, skip_validate=args.skip_validate)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
