"""
@file dcml_hyperparameter_search.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search pipeline for the GUI-adapted DCML module.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from DCML.Core.common import PathLike, ensure_dir, get_logger, resolve_path, utc_now_iso
from DCML.Core.dcml_results import format_seconds
from DCML.Core.dcml_tester import test_dcml
from DCML.Core.dcml_trainer import DEFAULT_MODEL_TYPE, train_dcml, validate_hyperparameters

LOGGER = get_logger(__name__)


class HyperparameterSearchDCMLError(RuntimeError):
    """Raised when DCML hyperparameter search cannot complete."""


def _as_list(values: Sequence[Any] | Any, default: Sequence[Any]) -> list[Any]:
    if values is None:
        return list(default)
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_trials_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "status",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_samples_split",
        "subsample",
        "max_features",
        "loss",
        "RMSE",
        "Pearson",
        "MAE",
        "training_seconds",
        "model_path",
        "prediction_summary_json",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _is_better(candidate: Mapping[str, Any], current: Optional[Mapping[str, Any]]) -> bool:
    if candidate.get("status") != "success":
        return False
    if current is None:
        return True
    c_rmse = float(candidate.get("RMSE", float("inf")))
    b_rmse = float(current.get("RMSE", float("inf")))
    if c_rmse < b_rmse:
        return True
    if c_rmse > b_rmse:
        return False
    return float(candidate.get("Pearson", float("-inf"))) > float(current.get("Pearson", float("-inf")))


def build_search_grid(
    *,
    n_estimators_values: Sequence[int] | None = None,
    max_depth_values: Sequence[int] | None = None,
    learning_rate_values: Sequence[float] | None = None,
    min_samples_split_values: Sequence[int] | None = None,
    subsample_values: Sequence[float] | None = None,
    max_features_values: Sequence[str | int | float | None] | None = None,
    loss_values: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build and validate the Cartesian hyperparameter grid."""
    grid = []
    for values in itertools.product(
        _as_list(n_estimators_values, [100, 300]),
        _as_list(max_depth_values, [3, 6]),
        _as_list(learning_rate_values, [0.01, 0.05]),
        _as_list(min_samples_split_values, [2]),
        _as_list(subsample_values, [0.7, 1.0]),
        _as_list(max_features_values, ["sqrt", None]),
        _as_list(loss_values, ["squared_error"]),
    ):
        raw = {
            "n_estimators": values[0],
            "max_depth": values[1],
            "learning_rate": values[2],
            "min_samples_split": values[3],
            "subsample": values[4],
            "max_features": values[5],
            "loss": values[6],
        }
        grid.append(validate_hyperparameters(raw))
    return grid


def run_hyperparameter_search(
    *,
    train_feature_zip: PathLike,
    train_label_npy: PathLike,
    validation_feature_zip: PathLike,
    validation_label_npy: PathLike,
    models_root: PathLike,
    results_root: PathLike,
    model_type: str = DEFAULT_MODEL_TYPE,
    device: str | None = "cpu",
    seed: int = 42,
    cast_float32: bool = True,
    n_estimators_values: Sequence[int] | None = None,
    max_depth_values: Sequence[int] | None = None,
    learning_rate_values: Sequence[float] | None = None,
    min_samples_split_values: Sequence[int] | None = None,
    subsample_values: Sequence[float] | None = None,
    max_features_values: Sequence[str | int | float | None] | None = None,
    loss_values: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run DCML hyperparameter search.

    Selection rule: minimize validation RMSE, then maximize validation Pearson.
    """
    if model_type != DEFAULT_MODEL_TYPE:
        raise HyperparameterSearchDCMLError(f"Unsupported DCML model type: {model_type}")

    train_feature_zip_path = resolve_path(train_feature_zip)
    train_label_npy_path = resolve_path(train_label_npy)
    validation_feature_zip_path = resolve_path(validation_feature_zip)
    validation_label_npy_path = resolve_path(validation_label_npy)
    models_root_path = ensure_dir(models_root)
    results_root_path = ensure_dir(results_root)

    search_grid = build_search_grid(
        n_estimators_values=n_estimators_values,
        max_depth_values=max_depth_values,
        learning_rate_values=learning_rate_values,
        min_samples_split_values=min_samples_split_values,
        subsample_values=subsample_values,
        max_features_values=max_features_values,
        loss_values=loss_values,
    )
    if not search_grid:
        raise HyperparameterSearchDCMLError("The DCML search grid is empty.")

    started_at = time.perf_counter()
    run_dir = ensure_dir(results_root_path / f"dcml_hpo_{time.strftime('%Y%m%d_%H%M%S')}")
    trials_csv = run_dir / "dcml_hyperparameter_trials.csv"
    best_config_yaml = run_dir / "best_config_dcml.yaml"
    search_config_json = run_dir / "search_config.json"

    search_config = {
        "script": "dcml_hyperparameter_search.py",
        "created_at_utc": utc_now_iso(),
        "selection_rule": {
            "primary_metric": "RMSE",
            "primary_goal": "min",
            "secondary_metric": "Pearson",
            "secondary_goal": "max",
        },
        "inputs": {
            "train_feature_zip": str(train_feature_zip_path),
            "train_label_npy": str(train_label_npy_path),
            "validation_feature_zip": str(validation_feature_zip_path),
            "validation_label_npy": str(validation_label_npy_path),
        },
        "models_root": str(models_root_path),
        "results_root": str(results_root_path),
        "run_dir": str(run_dir),
        "seed": int(seed),
        "cast_float32": bool(cast_float32),
        "n_trials": len(search_grid),
        "search_grid": search_grid,
    }
    _write_json(search_config_json, search_config)

    rows: list[dict[str, Any]] = []
    best_row: Optional[dict[str, Any]] = None

    for index, hparams in enumerate(search_grid, start=1):
        trial_id = f"trial_{index:04d}"
        trial_model_dir = ensure_dir(models_root_path / trial_id)
        trial_result_dir = ensure_dir(run_dir / trial_id)
        model_path = trial_model_dir / "DCML.pt"
        train_output_dir = ensure_dir(trial_result_dir / "train")
        predict_output_dir = ensure_dir(trial_result_dir / "validation")

        row: dict[str, Any] = {
            "trial_id": trial_id,
            "status": "failure",
            **hparams,
            "RMSE": "",
            "Pearson": "",
            "MAE": "",
            "training_seconds": "",
            "model_path": str(model_path),
            "prediction_summary_json": "",
            "error_message": "",
        }

        try:
            LOGGER.info("Starting DCML HPO %s/%s: %s", index, len(search_grid), hparams)
            train_summary = train_dcml(
                train_feature_zip=train_feature_zip_path,
                train_label_npy=train_label_npy_path,
                output_model=model_path,
                output_dir=train_output_dir,
                model_type=model_type,
                hyperparameters=hparams,
                device=device,
                seed=int(seed),
                cast_float32=bool(cast_float32),
            )
            prediction_summary = test_dcml(
                model_pt=model_path,
                feature_zip=validation_feature_zip_path,
                label_npy=validation_label_npy_path,
                output_dir=predict_output_dir,
                device=device,
                split_id=trial_id,
                dataset_name="validation",
                cast_float32=bool(cast_float32),
            )
            metrics = prediction_summary["metrics"]
            row.update(
                {
                    "status": "success",
                    "RMSE": float(metrics["RMSE"]),
                    "Pearson": float(metrics["Pearson"]),
                    "MAE": float(metrics["MAE"]),
                    "training_seconds": float(train_summary["training"]["training_seconds"]),
                    "prediction_summary_json": prediction_summary["outputs"]["summary_json"],
                    "error_message": "",
                }
            )
            if _is_better(row, best_row):
                best_row = dict(row)
        except Exception as exc:
            row["status"] = "failure"
            row["error_message"] = str(exc)
            LOGGER.exception("DCML HPO trial failed: %s", trial_id)
        finally:
            rows.append(row)
            _write_trials_csv(trials_csv, rows)

    elapsed_seconds = time.perf_counter() - started_at
    if best_row is None:
        payload = {
            "model_name": "DCML",
            "status": "failure",
            "message": "All DCML hyperparameter trials failed.",
            "trials_csv": str(trials_csv),
            "search_config_json": str(search_config_json),
            "elapsed_seconds": elapsed_seconds,
        }
        _write_yaml(best_config_yaml, payload)
        raise HyperparameterSearchDCMLError("All DCML hyperparameter trials failed. See trials CSV for errors.")

    best_model_source = Path(str(best_row["model_path"]))
    best_model_dir = ensure_dir(run_dir / "best_model")
    best_model_copy = best_model_dir / "DCML.pt"
    if best_model_source.is_file():
        shutil.copy2(best_model_source, best_model_copy)

    best_payload = {
        "model_name": "DCML",
        "status": "success",
        "selection_rule": {
            "primary_metric": "RMSE",
            "primary_goal": "min",
            "secondary_metric": "Pearson",
            "secondary_goal": "max",
        },
        "best_trial": {
            "trial_id": best_row["trial_id"],
            "hyperparameters": {
                "n_estimators": best_row["n_estimators"],
                "max_depth": best_row["max_depth"],
                "learning_rate": best_row["learning_rate"],
                "min_samples_split": best_row["min_samples_split"],
                "subsample": best_row["subsample"],
                "max_features": best_row["max_features"],
                "loss": best_row["loss"],
            },
        },
        "best_metrics": {
            "RMSE": float(best_row["RMSE"]),
            "Pearson": float(best_row["Pearson"]),
            "MAE": float(best_row["MAE"]),
        },
        "paths": {
            "run_dir": str(run_dir),
            "trials_csv": str(trials_csv),
            "best_config_yaml": str(best_config_yaml),
            "best_model_original": str(best_model_source),
            "best_model_copy": str(best_model_copy),
            "prediction_summary_json": str(best_row["prediction_summary_json"]),
        },
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_time": format_seconds(elapsed_seconds),
        "n_trials": len(rows),
        "n_success": sum(1 for row in rows if row.get("status") == "success"),
        "n_failures": sum(1 for row in rows if row.get("status") != "success"),
    }
    _write_yaml(best_config_yaml, best_payload)

    return {
        "status": "success",
        "message": "DCML hyperparameter search completed.",
        "best_trial": best_row["trial_id"],
        "best_metrics": best_payload["best_metrics"],
        "best_hyperparameters": best_payload["best_trial"]["hyperparameters"],
        "elapsed_time": best_payload["elapsed_time"],
        "elapsed_seconds": float(elapsed_seconds),
        "run_dir": str(run_dir),
        "trials_csv": str(trials_csv),
        "best_config_yaml": str(best_config_yaml),
        "best_model_path": str(best_model_copy),
    }


def _parse_csv_values(raw: str, caster):
    return [caster(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_max_features_values(raw: str) -> list[str | None]:
    values: list[str | None] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(None if text.lower() in {"none", "null"} else text)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DCML hyperparameter search.")
    parser.add_argument("--train-feature-zip", required=True)
    parser.add_argument("--train-label-npy", required=True)
    parser.add_argument("--validation-feature-zip", required=True)
    parser.add_argument("--validation-label-npy", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cast-float32", action="store_true")
    parser.add_argument("--n-estimators-values", default="100,300")
    parser.add_argument("--max-depth-values", default="3,6")
    parser.add_argument("--learning-rate-values", default="0.01,0.05")
    parser.add_argument("--min-samples-split-values", default="2")
    parser.add_argument("--subsample-values", default="0.7,1.0")
    parser.add_argument("--max-features-values", default="sqrt,none")
    parser.add_argument("--loss-values", default="squared_error")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_hyperparameter_search(
            train_feature_zip=args.train_feature_zip,
            train_label_npy=args.train_label_npy,
            validation_feature_zip=args.validation_feature_zip,
            validation_label_npy=args.validation_label_npy,
            models_root=args.models_root,
            results_root=args.results_root,
            device=args.device,
            seed=args.seed,
            cast_float32=args.cast_float32,
            n_estimators_values=_parse_csv_values(args.n_estimators_values, int),
            max_depth_values=_parse_csv_values(args.max_depth_values, int),
            learning_rate_values=_parse_csv_values(args.learning_rate_values, float),
            min_samples_split_values=_parse_csv_values(args.min_samples_split_values, int),
            subsample_values=_parse_csv_values(args.subsample_values, float),
            max_features_values=_parse_max_features_values(args.max_features_values),
            loss_values=_parse_csv_values(args.loss_values, str),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        LOGGER.exception("DCML hyperparameter search failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
