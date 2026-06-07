from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SPLIT_ORDER, TEST_SPLITS, VENDOR_DIR, preset_config


ORIGINAL_V122_DIR = VENDOR_DIR / "enterprise_delay_ensemble_v1_2_2"
if str(ORIGINAL_V122_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_V122_DIR))

import train_enterprise_delay_ensemble_v1_2_2 as base  # noqa: E402


LOGISTIC_MODEL_NAME = "Logistic Regression"
LOGISTIC_FAMILY = "logistic_regression"


def patch_original_module(cfg: dict[str, Any]) -> None:
    v122_files = cfg["v122_files"]
    base.FEATURE_CACHE_DIR = v122_files["train"].parent
    base.CACHE_FILES = dict(v122_files)
    base.TABULAR_CARDINALITY_LIMIT[LOGISTIC_FAMILY] = 450


def load_custom_data(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    patch_original_module(cfg)
    return {split_name: base.load_split(split_name) for split_name in SPLIT_ORDER}


def predict_scores_chunked(candidate: base.TrainedCandidate, df: pd.DataFrame, chunk_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        frame = base.transform_frame(df.iloc[start:end], candidate.feature_bundle)
        chunks.append(candidate.model.predict_proba(frame)[:, 1].astype("float32"))
        del frame
        gc.collect()
    return np.concatenate(chunks)


def save_logistic_coefficients(model: Any, feature_names: list[str], output_dir: Path) -> None:
    try:
        logistic = model.named_steps["logisticregression"]
        coefs = logistic.coef_[0]
    except Exception:
        return
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs.astype("float64"),
            "abs_coefficient": np.abs(coefs.astype("float64")),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_dir / "logistic_regression_coefficients.csv", index=False)


def fit_logistic_regression(
    train_x: pd.DataFrame,
    y_train: pd.Series,
    cv_x: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            class_weight="balanced",
            random_state=base.SEED,
            n_jobs=-1,
            **params,
        ),
    )
    model.fit(train_x, y_train)
    return model, model.predict_proba(cv_x)[:, 1]


def fit_model(spec: base.ModelSpec, train_df: pd.DataFrame, cv_df: pd.DataFrame, output_dir: Path) -> base.TrainedCandidate:
    if spec.family != LOGISTIC_FAMILY:
        return base.fit_model(spec, train_df, cv_df, output_dir)

    sample_seed = base.MODEL_SAMPLE_SEEDS.get(spec.name, 1201)
    sampled_train = base.stable_sample_month_aware(train_df, spec.train_row_limit, sample_seed)
    y_train = sampled_train["depdelay15"].astype("int8")
    y_cv = cv_df["depdelay15"].astype("int8")

    categorical_cols = base.get_model_categorical_columns(sampled_train, spec)
    numeric_cols = base.get_model_numeric_columns(sampled_train, categorical_cols, spec)

    start = time.perf_counter()
    feature_bundle = base.fit_dense_onehot_bundle(sampled_train, numeric_cols, categorical_cols)
    train_x = base.transform_frame(sampled_train, feature_bundle)
    cv_x = base.transform_frame(cv_df, feature_bundle)
    feature_names = feature_bundle["feature_names"]

    model, cv_scores = fit_logistic_regression(train_x, y_train, cv_x, spec.params)
    fit_seconds = time.perf_counter() - start
    chosen_threshold, threshold_table = base.choose_threshold(y_cv, cv_scores)
    cv_metrics = base.evaluate_scores(y_cv, cv_scores, chosen_threshold["threshold"])
    save_logistic_coefficients(model, feature_names, output_dir)

    del sampled_train, train_x, cv_x
    gc.collect()
    return base.TrainedCandidate(
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


def evaluate_candidate_on_split(
    candidate: base.TrainedCandidate,
    split_name: str,
    df: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, Any]:
    y_true = df["depdelay15"].astype("int8")
    metrics = base.evaluate_scores(y_true, scores, candidate.threshold)
    return {
        "model_name": candidate.spec.name,
        "family": candidate.spec.family,
        "feature_view": candidate.spec.feature_view,
        "split": split_name,
        "rows": int(len(df)),
        "threshold": float(candidate.threshold),
        "train_rows_used": int(candidate.spec.train_row_limit),
        **metrics,
    }


def save_winner_predictions(
    winner: base.TrainedCandidate,
    split_name: str,
    df: pd.DataFrame,
    scores: np.ndarray,
    output_dir: Path,
) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "row_number": np.arange(len(df), dtype=np.int64),
            "predicted_probability": scores,
            "predicted_delay_flag": (scores >= winner.threshold).astype("int8"),
            "actual_delay_flag": df["depdelay15"].astype("int8").to_numpy(),
        }
    )
    for col in ["FlightDate", "Marketing_Airline_Network", "Operating_Airline", "Origin", "Dest"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()
    out.to_csv(pred_dir / f"winning_model_{split_name}_predictions.csv", index=False)


def plot_winner_metrics(winner_split_df: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    plot_df = winner_split_df.set_index("split").loc[SPLIT_ORDER, metric_cols]

    ax = plot_df.plot(kind="bar", figsize=(14, 7), width=0.82)
    ax.set_title("Winning Original v1.2.2 Model Metrics by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "winning_model_metric_chart.png", dpi=180)
    plt.close()


def plot_model_cv_leaderboard(cv_ranked: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    cv_metric_cols = [f"cv_{col}" for col in metric_cols]
    plot_df = cv_ranked.set_index("model_name")[cv_metric_cols].rename(columns={f"cv_{col}": col for col in metric_cols})
    ax = plot_df.plot(kind="bar", figsize=(14, 7), width=0.82)
    ax.set_title("Original v1.2.2 Base Model CV Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "all_models_cv_comparison_chart.png", dpi=180)
    plt.close()


def make_single_model_bundle(
    winner: base.TrainedCandidate,
    winner_row: pd.Series,
    all_results: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "bundle_version": "v1.2.2_original_single_model_custom_5m",
        "model_name": winner.spec.name,
        "family": winner.spec.family,
        "feature_view": winner.spec.feature_view,
        "default_threshold": float(winner.threshold),
        "spec": asdict(winner.spec),
        "model": winner.model,
        "feature_bundle": winner.feature_bundle,
        "feature_names": winner.feature_names,
        "selection_row": winner_row.to_dict(),
        "all_split_metrics": all_results.to_dict(orient="records"),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }


def train_and_evaluate(preset: str, output_dir: Path | None, chunk_size: int, include_balanced_rf: bool) -> dict[str, Any]:
    cfg = preset_config(preset)
    if output_dir is None:
        output_dir = cfg["results_dir"]  # type: ignore[assignment]
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir = output_dir / "threshold_tables"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = output_dir / "model_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    base.THRESHOLD_DIR = threshold_dir

    data = load_custom_data(cfg)
    train_df = data["train"]
    cv_df = data["cv"]
    fit_cv_rows = int(base.MODE_CONFIG[str(cfg["model_mode"])]["cv_rows"])
    fit_cv_df = base.stable_sample_month_aware(cv_df, fit_cv_rows, base.SEED + 101)

    model_specs = base.build_model_specs(str(cfg["model_mode"]))
    model_specs.append(
        base.ModelSpec(
            LOGISTIC_MODEL_NAME,
            LOGISTIC_FAMILY,
            "dense_onehot",
            int(base.MODE_CONFIG[str(cfg["model_mode"])]["forest_train_rows"]),
            {
                "C": 0.25,
                "max_iter": 1200,
                "solver": "saga",
                "penalty": "l2",
                "tol": 1e-3,
            },
        )
    )
    if include_balanced_rf:
        existing_names = {spec.name for spec in model_specs}
        balanced_specs = [
            spec
            for spec in base.build_model_specs("quick")
            if spec.name == "Balanced Random Forest" and spec.name not in existing_names
        ]
        model_specs.extend(balanced_specs)

    candidates: dict[str, base.TrainedCandidate] = {}
    result_rows: list[dict[str, Any]] = []
    fit_cv_rows_out: dict[str, dict[str, Any]] = {}

    for spec in model_specs:
        print(f"[custom train] fitting original model: {spec.name}")
        start = time.perf_counter()
        cache_path = model_cache_dir / f"{spec.name.lower().replace(' ', '_')}_candidate.joblib"
        if cache_path.exists():
            candidate = joblib.load(cache_path)
            print(f"[custom train] loaded cached model: {cache_path}")
        else:
            candidate = fit_model(spec, train_df, fit_cv_df, output_dir)
            joblib.dump(candidate, cache_path, compress=3)
        candidates[spec.name] = candidate
        candidate.threshold_table.to_csv(threshold_dir / f"{spec.name.lower().replace(' ', '_')}_thresholds.csv", index=False)

        cv_row = base.candidate_to_cv_row(candidate)
        cv_row["wall_seconds_including_exports"] = round(time.perf_counter() - start, 2)
        fit_cv_rows_out[spec.name] = cv_row

        for split_name, df in data.items():
            print(f"[custom train] scoring {spec.name} on {split_name}")
            scores = predict_scores_chunked(candidate, df, chunk_size=chunk_size)
            result_rows.append(evaluate_candidate_on_split(candidate, split_name, df, scores))

        pd.DataFrame(result_rows).to_csv(output_dir / "model_results_all_splits_partial.csv", index=False)
        gc.collect()

    all_results = pd.DataFrame(result_rows)
    all_results.to_csv(output_dir / "model_results_all_splits.csv", index=False)

    cv_rank_rows = []
    metadata_cols = ["model_name", "family", "feature_view", "train_rows_used", "fit_seconds", "threshold"]
    metric_exclusions = {"model_name", "family", "feature_view", "split", "rows", "train_rows_used"}
    for row in all_results.loc[all_results["split"] == "cv"].to_dict(orient="records"):
        fit_meta = fit_cv_rows_out.get(str(row["model_name"]), {})
        ranked_row = {
            "model_name": row["model_name"],
            "family": row["family"],
            "feature_view": row["feature_view"],
            "train_rows_used": row["train_rows_used"],
            "fit_seconds": fit_meta.get("fit_seconds", float("nan")),
            "threshold": row["threshold"],
            "blend_members": "",
            "blend_weights": "",
            "fit_cv_rows_used": len(fit_cv_df),
        }
        for key, value in row.items():
            if key not in metric_exclusions:
                ranked_row[f"cv_{key}"] = value
        cv_rank_rows.append(ranked_row)

    cv_ranked = base.sort_cv_results(pd.DataFrame(cv_rank_rows))
    cv_ranked.to_csv(output_dir / "model_results_cv_ranked.csv", index=False)

    winner_name = str(cv_ranked.iloc[0]["model_name"])
    winner = candidates[winner_name]
    winner_results = all_results.loc[all_results["model_name"] == winner_name].copy()
    winner_results["split"] = pd.Categorical(winner_results["split"], categories=SPLIT_ORDER, ordered=True)
    winner_results = winner_results.sort_values("split").reset_index(drop=True)
    winner_results.to_csv(output_dir / "winning_model_split_comparison.csv", index=False)

    for split_name, df in data.items():
        scores = predict_scores_chunked(winner, df, chunk_size=chunk_size)
        save_winner_predictions(winner, split_name, df, scores, output_dir)

    plot_winner_metrics(winner_results, output_dir)
    plot_model_cv_leaderboard(cv_ranked, output_dir)

    bundle = make_single_model_bundle(winner, cv_ranked.iloc[0], all_results)
    joblib.dump(bundle, output_dir / "winning_model_bundle.joblib", compress=3)

    summary = {
        "experiment": "classifier_v1_2_2_original_5m",
        "preset": preset,
        "no_blending": True,
        "split_order": SPLIT_ORDER,
        "test_splits": TEST_SPLITS,
        "model_count": len(model_specs),
        "winning_model": winner_name,
        "winning_threshold": float(winner.threshold),
        "results_dir": str(output_dir),
        "winning_model_bundle": str(output_dir / "winning_model_bundle.joblib"),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def finalize_from_cached_results(output_dir: Path, preset: str, chunk_size: int) -> dict[str, Any]:
    cfg = preset_config(preset)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = output_dir / "model_results_all_splits.csv"
    cv_ranked_path = output_dir / "model_results_cv_ranked.csv"
    if not all_results_path.exists() or not cv_ranked_path.exists():
        raise FileNotFoundError("Expected model_results_all_splits.csv and model_results_cv_ranked.csv before finalizing.")

    all_results = pd.read_csv(all_results_path)
    cv_ranked = pd.read_csv(cv_ranked_path)
    winner_name = str(cv_ranked.iloc[0]["model_name"])
    cache_path = output_dir / "model_cache" / f"{winner_name.lower().replace(' ', '_')}_candidate.joblib"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached winning model: {cache_path}")

    data = load_custom_data(cfg)
    winner = joblib.load(cache_path)

    winner_results = all_results.loc[all_results["model_name"] == winner_name].copy()
    winner_results["split"] = pd.Categorical(winner_results["split"], categories=SPLIT_ORDER, ordered=True)
    winner_results = winner_results.sort_values("split").reset_index(drop=True)
    winner_results.to_csv(output_dir / "winning_model_split_comparison.csv", index=False)

    for split_name, df in data.items():
        scores = predict_scores_chunked(winner, df, chunk_size=chunk_size)
        save_winner_predictions(winner, split_name, df, scores, output_dir)

    plot_winner_metrics(winner_results, output_dir)
    plot_model_cv_leaderboard(cv_ranked, output_dir)
    bundle = make_single_model_bundle(winner, cv_ranked.iloc[0], all_results)
    joblib.dump(bundle, output_dir / "winning_model_bundle.joblib", compress=3)

    summary = {
        "experiment": "classifier_v1_2_2_original_5m" if preset == "full" else "classifier_v1_2_2_original_quick",
        "preset": preset,
        "no_blending": True,
        "split_order": SPLIT_ORDER,
        "test_splits": TEST_SPLITS,
        "model_count": int(cv_ranked["model_name"].nunique()),
        "winning_model": winner_name,
        "winning_threshold": float(winner.threshold),
        "results_dir": str(output_dir),
        "winning_model_bundle": str(output_dir / "winning_model_bundle.joblib"),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train original v1.2.2 base models on the custom full/quick split; no blending.")
    parser.add_argument("--preset", choices=["full", "quick"], default="full")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--include-balanced-rf",
        action="store_true",
        help="Also train the Balanced Random Forest model that v1.2.2 only includes in quick mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_only:
        cfg = preset_config(args.preset)
        output_dir = args.output_dir if args.output_dir is not None else cfg["results_dir"]  # type: ignore[assignment]
        finalize_from_cached_results(output_dir=output_dir, preset=args.preset, chunk_size=args.chunk_size)
        return
    train_and_evaluate(
        preset=args.preset,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        include_balanced_rf=args.include_balanced_rf,
    )


if __name__ == "__main__":
    main()
