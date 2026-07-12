"""
@file dcml_gui_workflows.py
@brief GUI-facing DCML workflows that resolve prepared feature roots.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from DCML.Core.common import PathLike, ensure_dir, resolve_path, utc_now_iso
from DCML.Core.data_utils import load_dcml_dataset
from DCML.Core.dcml_hyperparameter_search import run_hyperparameter_search
from DCML.Core.dcml_tester import test_dcml
from DCML.Core.dcml_trainer import DEFAULT_MODEL_TYPE, train_dcml

DCMLVariant = Literal["distance_only", "real_charge", "full"]
DCMLSplit = Literal["train", "valid", "test", "all"]

VARIANT_DATASET_DIRS: dict[str, tuple[str, ...]] = {
    "distance_only": ("distance_only", "urv_distance_only"),
    "real_charge": ("real_charge", "urv_full_pqr"),
    "full": ("full", "urv_full_pqr", "urv_full_zero_charge"),
}

SPLIT_FILE_STEMS: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("validation", "valid"),
    "test": ("test",),
    "all": ("all", "trainval"),
}

GENERATION_LIMITATION_MESSAGE = (
    "DCML Generate Data validates/prepares existing feature matrices; "
    "raw feature extraction is not implemented in this module."
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _progress(callback, message: str) -> None:
    if callback:
        callback(message)


def resolve_prepared_feature_root(prepared_feature_root: PathLike, variant: str) -> Path:
    """Resolve a directory containing DCML feature/label files for a variant."""
    root = resolve_path(prepared_feature_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared feature root does not exist: {root}")

    if any((root / f"{stem}_feature.zip").is_file() for stem in ("all", "train", "validation", "test")):
        return root

    for dirname in VARIANT_DATASET_DIRS.get(variant, (variant,)):
        candidate = root / dirname
        if candidate.is_dir() and any(
            (candidate / f"{stem}_feature.zip").is_file()
            for stem in ("all", "train", "validation", "test")
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        f"No DCML feature ZIP files found under {root} for variant {variant!r}."
    )


def resolve_split_files(prepared_feature_root: PathLike, variant: str, split: str) -> tuple[Path, Path, str]:
    """Return feature ZIP, label NPY and canonical split stem."""
    dataset_root = resolve_prepared_feature_root(prepared_feature_root, variant)
    for stem in SPLIT_FILE_STEMS.get(split, (split,)):
        feature_zip = dataset_root / f"{stem}_feature.zip"
        label_npy = dataset_root / f"{stem}_label.npy"
        if feature_zip.is_file() and label_npy.is_file():
            return feature_zip, label_npy, stem
    raise FileNotFoundError(
        f"Missing {split!r} DCML feature/label files in {dataset_root}. "
        "Expected files like train_feature.zip and train_label.npy."
    )


def prepare_dcml_features(
    *,
    prepared_feature_root: PathLike,
    output_feature_dir: PathLike,
    variant: DCMLVariant = "distance_only",
    raw_complex_folder: PathLike | None = None,
    pqr_folder: PathLike | None = None,
    labels_file: PathLike | None = None,
    sample_ids_file: PathLike | None = None,
    overwrite: bool = False,
    cast_float32: bool = True,
) -> dict[str, Any]:
    """Validate and optionally copy already prepared DCML feature matrices."""
    source_root = resolve_prepared_feature_root(prepared_feature_root, variant)
    output_root = ensure_dir(output_feature_dir)

    validation_rows: list[dict[str, Any]] = []
    copied_files: list[str] = []

    for split in ("all", "train", "valid", "test"):
        try:
            feature_zip, label_npy, stem = resolve_split_files(source_root, variant, split)
        except FileNotFoundError:
            continue

        dataset = load_dcml_dataset(
            feature_zip=feature_zip,
            label_npy=label_npy,
            cast_float32=bool(cast_float32),
        )
        validation_rows.append(
            {
                "split": split,
                "source_stem": stem,
                "feature_zip": str(feature_zip),
                "label_npy": str(label_npy),
                "n_samples": int(dataset.n_samples),
                "n_features": int(dataset.n_features),
                "feature_dtype": dataset.report.feature_dtype_final,
                "label_dtype": dataset.report.label_dtype_final,
                "internal_npy_name": dataset.internal_npy_name,
            }
        )

        if source_root.resolve() != output_root.resolve():
            for src in (feature_zip, label_npy):
                dst = output_root / src.name
                if dst.exists() and not overwrite:
                    raise FileExistsError(f"Output file already exists: {dst}")
                shutil.copy2(src, dst)
                copied_files.append(str(dst))

    if not validation_rows:
        raise FileNotFoundError(f"No valid DCML feature/label pairs found in {source_root}.")

    for optional_path in (labels_file, sample_ids_file):
        if optional_path:
            src = resolve_path(optional_path)
            if not src.is_file():
                raise FileNotFoundError(f"Optional input file does not exist: {src}")
            dst = output_root / src.name
            if src.resolve() != dst.resolve():
                if dst.exists() and not overwrite:
                    raise FileExistsError(f"Output file already exists: {dst}")
                shutil.copy2(src, dst)
                copied_files.append(str(dst))

    report = {
        "script": "dcml_gui_workflows.py",
        "timestamp_utc": utc_now_iso(),
        "status": "success",
        "action": "Generate Data",
        "variant": variant,
        "message": GENERATION_LIMITATION_MESSAGE,
        "inputs": {
            "prepared_feature_root": str(source_root),
            "raw_complex_folder": str(raw_complex_folder or ""),
            "pqr_folder": str(pqr_folder or ""),
            "labels_file": str(labels_file or ""),
            "sample_ids_file": str(sample_ids_file or ""),
        },
        "outputs": {
            "output_feature_dir": str(output_root),
            "report_json": str(output_root / "dcml_generate_data_report.json"),
            "copied_files": copied_files,
        },
        "validated_splits": validation_rows,
    }
    _write_json(output_root / "dcml_generate_data_report.json", report)
    return report


def train_dcml_from_prepared_root(
    *,
    prepared_feature_root: PathLike,
    output_dir: PathLike,
    variant: DCMLVariant = "distance_only",
    seed: int = 42,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
    labels_path: PathLike | None = None,
    sample_ids_path: PathLike | None = None,
    output_model: PathLike | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    hyperparameters: Optional[Mapping[str, Any]] = None,
    device: str | None = "cpu",
    cast_float32: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    split = "train" if use_dataset_folds else "all"
    _progress(progress_callback, f"Resolving {split} split files for variant {variant}.")
    feature_zip, label_npy, split_stem = resolve_split_files(prepared_feature_root, variant, split)
    output_dir_path = ensure_dir(output_dir)
    model_path = resolve_path(output_model or (output_dir_path / "DCML.pt"))
    _progress(progress_callback, f"Feature ZIP: {feature_zip}")
    _progress(progress_callback, f"Label NPY: {label_npy}")
    _progress(progress_callback, f"Model checkpoint path: {model_path}")
    _progress(progress_callback, f"Selected hyperparameters: {dict(hyperparameters or {})}")
    summary = train_dcml(
        train_feature_zip=feature_zip,
        train_label_npy=label_npy,
        output_model=model_path,
        output_dir=output_dir_path,
        model_type=model_type,
        hyperparameters=hyperparameters,
        device=device,
        seed=int(seed),
        cast_float32=bool(cast_float32),
        progress_callback=progress_callback,
    )
    summary["gui"] = {
        "variant": variant,
        "fold_index": int(fold_index),
        "use_dataset_folds": bool(use_dataset_folds),
        "resolved_split": split_stem,
        "labels_path": str(labels_path or ""),
        "sample_ids_path": str(sample_ids_path or ""),
    }
    return summary


def search_dcml_from_prepared_root(
    *,
    prepared_feature_root: PathLike,
    output_root: PathLike,
    variant: DCMLVariant = "distance_only",
    seed: int = 42,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
    labels_path: PathLike | None = None,
    sample_ids_path: PathLike | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    device: str | None = "cpu",
    cast_float32: bool = True,
    n_estimators_values=None,
    max_depth_values=None,
    learning_rate_values=None,
    min_samples_split_values=None,
    subsample_values=None,
    max_features_values=None,
    loss_values=None,
    progress_callback=None,
) -> dict[str, Any]:
    _progress(progress_callback, "Resolving train and validation split files.")
    train_feature_zip, train_label_npy, _ = resolve_split_files(prepared_feature_root, variant, "train")
    validation_feature_zip, validation_label_npy, _ = resolve_split_files(prepared_feature_root, variant, "valid")
    output_root_path = ensure_dir(output_root)
    _progress(progress_callback, f"Train feature ZIP: {train_feature_zip}")
    _progress(progress_callback, f"Train label NPY: {train_label_npy}")
    _progress(progress_callback, f"Validation feature ZIP: {validation_feature_zip}")
    _progress(progress_callback, f"Validation label NPY: {validation_label_npy}")
    _progress(progress_callback, f"Labels path: {labels_path or ''}")
    _progress(progress_callback, f"Sample IDs path: {sample_ids_path or ''}")
    results = run_hyperparameter_search(
        train_feature_zip=train_feature_zip,
        train_label_npy=train_label_npy,
        validation_feature_zip=validation_feature_zip,
        validation_label_npy=validation_label_npy,
        models_root=output_root_path / "models",
        results_root=output_root_path / "results",
        model_type=model_type,
        device=device,
        seed=int(seed),
        cast_float32=bool(cast_float32),
        n_estimators_values=n_estimators_values,
        max_depth_values=max_depth_values,
        learning_rate_values=learning_rate_values,
        min_samples_split_values=min_samples_split_values,
        subsample_values=subsample_values,
        max_features_values=max_features_values,
        loss_values=loss_values,
        progress_callback=progress_callback,
    )
    results["gui"] = {
        "variant": variant,
        "fold_index": int(fold_index),
        "use_dataset_folds": bool(use_dataset_folds),
        "labels_path": str(labels_path or ""),
        "sample_ids_path": str(sample_ids_path or ""),
    }
    return results


def evaluate_dcml_from_prepared_root(
    *,
    prepared_feature_root: PathLike,
    model_pt: PathLike,
    output_dir: PathLike,
    variant: DCMLVariant = "distance_only",
    fold_index: int = 0,
    split: DCMLSplit = "test",
    labels_path: PathLike | None = None,
    sample_ids_path: PathLike | None = None,
    device: str | None = "cpu",
    cast_float32: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    _progress(progress_callback, f"Resolving {split} split files for variant {variant}.")
    feature_zip, label_npy, split_stem = resolve_split_files(prepared_feature_root, variant, split)
    output_dir_path = ensure_dir(output_dir)
    _progress(progress_callback, f"Feature ZIP: {feature_zip}")
    _progress(progress_callback, f"Label NPY: {label_npy}")
    _progress(progress_callback, f"Checkpoint path: {model_pt}")
    summary = test_dcml(
        model_pt=model_pt,
        feature_zip=feature_zip,
        label_npy=label_npy,
        output_dir=output_dir_path,
        device=device,
        split_id=split_stem,
        dataset_name=f"{variant}_{split_stem}",
        cast_float32=bool(cast_float32),
        progress_callback=progress_callback,
    )
    gui_payload = {
        "variant": variant,
        "fold_index": int(fold_index),
        "split": split,
        "resolved_split": split_stem,
        "labels_path": str(labels_path or ""),
        "sample_ids_path": str(sample_ids_path or ""),
    }
    summary["gui"] = gui_payload
    metrics_json = _write_json(output_dir_path / "metrics.json", summary.get("metrics", {}))
    run_config_json = _write_json(
        output_dir_path / "run_config.json",
        {
            "action": "Evaluate",
            "prepared_feature_root": str(resolve_prepared_feature_root(prepared_feature_root, variant)),
            "model_pt": str(resolve_path(model_pt)),
            "output_dir": str(output_dir_path),
            "feature_zip": str(feature_zip),
            "label_npy": str(label_npy),
            "gui": gui_payload,
            "cast_float32": bool(cast_float32),
        },
    )
    summary.setdefault("outputs", {})["metrics_json"] = str(metrics_json)
    summary.setdefault("outputs", {})["run_config_json"] = str(run_config_json)
    return summary
