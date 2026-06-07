# Classifier v1.2.2 Original Models, 5M Split

This is a fresh end-to-end experiment folder for the v1.2.2 classification logic with **no model blending**. It builds custom splits from RAW flight CSV files, applies the v1.2.2 weather feature augmentation, trains the original base models, and writes all outputs inside this folder.

The folder is self-contained except for `RAW`. It includes local copies of the airport metadata, weather notebook, NOAA station cache, Meteostat/weather history cache, and the original v1.2.1/v1.2.2 implementation files under `resources/` and `vendor/`.

The training run also includes a dense one-hot **Logistic Regression** baseline. Its coefficients are exported to `logistic_regression_coefficients.csv`.

## Split Design

Total rows: `5,000,000`

| Split | Year(s) | Rows | Ratio |
|---|---:|---:|---:|
| Train | 2022-2023 | 2,750,000 | 55% |
| CV | 2024 | 750,000 | 15% |
| Test 1 | 2025 | 500,000 | 10% |
| Test 2 | 2025 | 500,000 | 10% |
| Test 3 | 2025 | 500,000 | 10% |

Train uses 1,100,000 rows from 2022 and 1,650,000 rows from 2023, preserving the previous v1.x balance between the two training years.

## Run

From the parent folder that contains both `RAW` and `classifier_v1_2_2_original_3m`:

```powershell
pip install -r .\classifier_v1_2_2_original_3m\requirements.txt

python .\classifier_v1_2_2_original_3m\src\build_v121_feature_store_3m.py
python .\classifier_v1_2_2_original_3m\src\build_v122_feature_store_3m.py
python .\classifier_v1_2_2_original_3m\src\train_original_models_3m.py
```

Or run the wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\classifier_v1_2_2_original_3m\run_all.ps1
```

Quick smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File .\classifier_v1_2_2_original_3m\run_quick.ps1
```

Or step by step:

```powershell
python .\classifier_v1_2_2_original_3m\src\build_v121_feature_store_3m.py --preset quick
python .\classifier_v1_2_2_original_3m\src\build_v122_feature_store_3m.py --preset quick
python .\classifier_v1_2_2_original_3m\src\train_original_models_3m.py --preset quick
```

## Outputs

Feature stores:

- `artifacts/features_v1_2_1_custom_5m`
- `artifacts/features_v1_2_2_custom_5m`
- quick feature stores use `artifacts/features_v1_2_1_quick` and `artifacts/features_v1_2_2_quick`

Local required resources:

- `resources/airports.csv`
- `resources/v4/flight_delay_time_safe_sampling_with_meteostat_upgraded_v4.ipynb`
- `resources/v4/weather_cache_meteostat_v4`
- `resources/fly/noaa_data`
- `resources/weather/weather_safe_history_meteostat_hourly.parquet`
- `vendor/enterprise_delay_ensemble_v1_2_1`
- `vendor/enterprise_delay_ensemble_v1_2_2`

Training results:

- `results/model_results_all_splits.csv`
- `results/model_results_cv_ranked.csv`
- `results/winning_model_split_comparison.csv`
- `results/winning_model_metric_chart.png`
- `results/winning_model_bundle.joblib`
- `results/predictions/winning_model_<split>_predictions.csv`
- quick training results use `results_quick`

EDA notebook:

- `notebooks/raw_delay_eda.ipynb`
