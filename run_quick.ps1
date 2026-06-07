$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path -Parent $root

Set-Location $project

python .\classifier_v1_2_2_original_3m\src\build_v121_feature_store_3m.py --preset quick
python .\classifier_v1_2_2_original_3m\src\build_v122_feature_store_3m.py --preset quick
python .\classifier_v1_2_2_original_3m\src\train_original_models_3m.py --preset quick

