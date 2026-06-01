#!/usr/bin/env python3
"""
Create a ready-to-train DCML dataset folder from all_feature.zip + all_label.npy.

The script:
- validates feature/label compatibility,
- copies all_feature.zip, all_label.npy and sample_ids.csv,
- creates train / validation / test / trainval splits,
- validates distance/charge blocks,
- writes split_indices.json,
- writes dataset_validation_status.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from DCML.Core.data_utils import load_dcml_dataset, save_feature_zip, save_labels_npy


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_sample_ids(path: Path | None, n_samples: int) -> list[str]:
    if path is None or not path.is_file():
        return [str(i) for i in range(n_samples)]

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            raise ValueError(f"sample_ids CSV has no header: {path}")

        preferred = "PDB_ID" if "PDB_ID" in fieldnames else fieldnames[0]
        ids = []
        for row in reader:
            value = str(row.get(preferred, "")).strip()
            if value:
                ids.append(value)

    if len(ids) != n_samples:
        raise ValueError(
            f"sample_ids count does not match features: {len(ids)} != {n_samples}"
        )

    return ids


def generate_splits(n_samples: int, seed: int) -> dict[str, Any]:
    indices = np.arange(n_samples)

    if n_samples == 378:
        test_count = 57
        validation_count = 57
    else:
        test_count = max(1, round(n_samples * 0.15))
        validation_count = max(1, round(n_samples * 0.15))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_count,
        random_state=seed,
        shuffle=True,
    )

    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=validation_count,
        random_state=seed,
        shuffle=True,
    )

    return {
        "method": "random_holdout",
        "seed": seed,
        "n_samples": int(n_samples),
        "train": [int(x) for x in train_idx],
        "validation": [int(x) for x in validation_idx],
        "test": [int(x) for x in test_idx],
    }


def load_existing_splits(path: Path, n_samples: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    for key in ["train", "validation", "test"]:
        if key not in payload:
            raise ValueError(f"Missing split key '{key}' in {path}")
        values = payload[key]
        if not isinstance(values, list):
            raise ValueError(f"Split '{key}' must be a list in {path}")

    all_indices = payload["train"] + payload["validation"] + payload["test"]

    if sorted(all_indices) != list(range(n_samples)):
        raise ValueError(
            "Split indices are not a complete partition of the dataset."
        )

    return payload


def copy_if_needed(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a ready-to-train DCML dataset variant."
    )

    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--all-feature-zip", required=True, type=Path)
    parser.add_argument("--all-label-npy", required=True, type=Path)
    parser.add_argument("--sample-ids-csv", required=True, type=Path)

    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--feature-mode", required=True, choices=["distance_only", "full"])
    parser.add_argument("--charge-mode", required=True, choices=["none", "zeros", "pqr"])

    parser.add_argument("--expected-samples", type=int, default=378)
    parser.add_argument("--expected-features", type=int, required=True)
    parser.add_argument("--distance-features", type=int, default=63360)
    parser.add_argument("--charge-features", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-source", type=Path, default=None)

    parser.add_argument("--require-nonzero-charge", action="store_true")
    parser.add_argument("--expect-zero-charge", action="store_true")

    parser.add_argument("--report", action="append", default=[])

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dcml_dataset(
        feature_zip=args.all_feature_zip,
        label_npy=args.all_label_npy,
        cast_float32=True,
    )

    if dataset.n_samples != args.expected_samples:
        raise ValueError(
            f"Unexpected sample count: {dataset.n_samples} != {args.expected_samples}"
        )

    if dataset.n_features != args.expected_features:
        raise ValueError(
            f"Unexpected feature count: {dataset.n_features} != {args.expected_features}"
        )

    sample_ids = read_sample_ids(args.sample_ids_csv, dataset.n_samples)

    zero_rows = int(np.all(dataset.features == 0, axis=1).sum())
    finite = bool(np.isfinite(dataset.features).all())

    if zero_rows != 0:
        raise ValueError(f"Feature matrix contains {zero_rows} fully zero rows.")

    if not finite:
        raise ValueError("Feature matrix contains NaN or infinite values.")

    charge_validation: dict[str, Any] | None = None

    if args.charge_features > 0:
        start = args.distance_features
        end = args.distance_features + args.charge_features

        if end > dataset.n_features:
            raise ValueError(
                f"Invalid charge block [{start}:{end}] for {dataset.n_features} features."
            )

        charge_block = dataset.features[:, start:end]
        charge_nonzero = int(np.count_nonzero(charge_block))
        charge_sum_abs = float(np.abs(charge_block).sum())

        charge_validation = {
            "charge_block_start": start,
            "charge_block_end": end,
            "charge_nonzero": charge_nonzero,
            "charge_sum_abs": charge_sum_abs,
        }

        if args.require_nonzero_charge and charge_nonzero <= 0:
            raise ValueError("Charge block is expected to be nonzero, but it is zero.")

        if args.expect_zero_charge and charge_nonzero != 0:
            raise ValueError("Charge block is expected to be zero, but it contains nonzero values.")

    if args.split_source:
        splits = load_existing_splits(args.split_source.resolve(), dataset.n_samples)
        split_origin = str(args.split_source.resolve())
    else:
        splits = generate_splits(dataset.n_samples, args.seed)
        split_origin = "generated"

    train_idx = np.asarray(splits["train"], dtype=int)
    validation_idx = np.asarray(splits["validation"], dtype=int)
    test_idx = np.asarray(splits["test"], dtype=int)
    trainval_idx = np.concatenate([train_idx, validation_idx])

    split_payload = dict(splits)
    split_payload.update(
        {
            "created_at_utc": now_utc(),
            "variant_id": args.variant_id,
            "split_origin": split_origin,
            "counts": {
                "train": int(len(train_idx)),
                "validation": int(len(validation_idx)),
                "test": int(len(test_idx)),
                "trainval": int(len(trainval_idx)),
            },
            "sample_ids": {
                "train": [sample_ids[i] for i in train_idx],
                "validation": [sample_ids[i] for i in validation_idx],
                "test": [sample_ids[i] for i in test_idx],
            },
        }
    )

    (dataset_root / "split_indices.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_feature_dst = copy_if_needed(args.all_feature_zip.resolve(), dataset_root / "all_feature.zip")
    all_label_dst = copy_if_needed(args.all_label_npy.resolve(), dataset_root / "all_label.npy")
    sample_ids_dst = copy_if_needed(args.sample_ids_csv.resolve(), dataset_root / "sample_ids.csv")

    save_feature_zip(
        dataset.features[train_idx],
        dataset_root / "train_feature.zip",
        internal_npy_name="train_feature.npy",
    )
    save_labels_npy(dataset.labels[train_idx], dataset_root / "train_label.npy")

    save_feature_zip(
        dataset.features[validation_idx],
        dataset_root / "validation_feature.zip",
        internal_npy_name="validation_feature.npy",
    )
    save_labels_npy(dataset.labels[validation_idx], dataset_root / "validation_label.npy")

    save_feature_zip(
        dataset.features[test_idx],
        dataset_root / "test_feature.zip",
        internal_npy_name="test_feature.npy",
    )
    save_labels_npy(dataset.labels[test_idx], dataset_root / "test_label.npy")

    save_feature_zip(
        dataset.features[trainval_idx],
        dataset_root / "trainval_feature.zip",
        internal_npy_name="trainval_feature.npy",
    )
    save_labels_npy(dataset.labels[trainval_idx], dataset_root / "trainval_label.npy")

    copied_reports = {}
    for raw_report in args.report:
        report_path = Path(raw_report).expanduser().resolve()
        if report_path.is_file():
            dst = dataset_root / report_path.name
            shutil.copy2(report_path, dst)
            copied_reports[report_path.name] = str(dst)

    artifacts = {
        "all_feature_zip": file_info(all_feature_dst),
        "all_label_npy": file_info(all_label_dst),
        "sample_ids_csv": file_info(sample_ids_dst),
        "split_indices_json": file_info(dataset_root / "split_indices.json"),
        "train_feature_zip": file_info(dataset_root / "train_feature.zip"),
        "train_label_npy": file_info(dataset_root / "train_label.npy"),
        "validation_feature_zip": file_info(dataset_root / "validation_feature.zip"),
        "validation_label_npy": file_info(dataset_root / "validation_label.npy"),
        "test_feature_zip": file_info(dataset_root / "test_feature.zip"),
        "test_label_npy": file_info(dataset_root / "test_label.npy"),
        "trainval_feature_zip": file_info(dataset_root / "trainval_feature.zip"),
        "trainval_label_npy": file_info(dataset_root / "trainval_label.npy"),
    }

    ready_for_hpo = all(
        artifacts[key]["exists"]
        for key in [
            "train_feature_zip",
            "train_label_npy",
            "validation_feature_zip",
            "validation_label_npy",
        ]
    )

    ready_for_final_training = all(
        artifacts[key]["exists"]
        for key in ["trainval_feature_zip", "trainval_label_npy"]
    )

    ready_for_final_evaluation = all(
        artifacts[key]["exists"]
        for key in ["test_feature_zip", "test_label_npy"]
    )

    status = {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "variant_id": args.variant_id,
        "dataset_root": str(dataset_root),
        "feature_mode": args.feature_mode,
        "charge_mode": args.charge_mode,
        "validation": {
            "n_samples": int(dataset.n_samples),
            "n_features": int(dataset.n_features),
            "expected_samples": int(args.expected_samples),
            "expected_features": int(args.expected_features),
            "feature_dtype": str(dataset.features.dtype),
            "label_dtype": str(dataset.labels.dtype),
            "finite": finite,
            "zero_rows": zero_rows,
            "label_min": float(np.min(dataset.labels)),
            "label_max": float(np.max(dataset.labels)),
            "label_mean": float(np.mean(dataset.labels)),
            "charge_validation": charge_validation,
        },
        "split": {
            "seed": int(splits.get("seed", args.seed)),
            "origin": split_origin,
            "counts": split_payload["counts"],
        },
        "readiness": {
            "ready_for_hpo": ready_for_hpo,
            "ready_for_final_training": ready_for_final_training,
            "ready_for_final_evaluation": ready_for_final_evaluation,
        },
        "artifacts": artifacts,
        "copied_reports": copied_reports,
    }

    (dataset_root / "dataset_validation_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
