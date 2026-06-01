#!/usr/bin/env python3
"""Create distance-only DCML datasets by slicing the first N descriptor columns.

This script is intended for the current DCML/TFM layout:

- PDBbind combined training features: data/train_feature.zip
- PDBbind combined test features: data/test_feature.zip
- MPro-URV combined baseline features: data_urv/dcml_input/MProURV_all_feature.zip

The baseline URV features produced previously are:
    [real distance descriptors | zero-padded charge descriptors]

For a fairer baseline, this script creates feature.zip files that contain only
real distance descriptors. You then train a new DCML model on the same slice of
PDBbind and predict URV with the same slice.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def read_single_npy_from_zip(path: Path) -> tuple[np.ndarray, str]:
    require_file(path, "feature ZIP")
    with zipfile.ZipFile(path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".npy") and not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one .npy inside {path}; found {len(names)}: {names[:5]}")
        name = names[0]
        with zf.open(name, "r") as fh:
            data = fh.read()
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D in {path}; got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Feature matrix must be numeric in {path}; got dtype {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Feature matrix contains NaN or infinite values: {path}")
    return arr, name


def write_zip_from_array(path: Path, arr: np.ndarray, internal_name: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(internal_name, buf.getvalue())


def slice_distance(
    *,
    input_zip: Path,
    output_zip: Path,
    n_distance_features: int,
    internal_name: str,
    cast_float32: bool,
    overwrite: bool,
) -> dict[str, Any]:
    arr, input_internal_name = read_single_npy_from_zip(input_zip)
    if arr.shape[1] < n_distance_features:
        raise ValueError(
            f"{input_zip} has only {arr.shape[1]} features; cannot slice first {n_distance_features}."
        )

    sliced = arr[:, :n_distance_features]
    if cast_float32 and sliced.dtype != np.float32:
        sliced = sliced.astype(np.float32, copy=False)

    if np.any(np.all(sliced == 0, axis=1)):
        zero_rows = np.where(np.all(sliced == 0, axis=1))[0][:20].tolist()
        raise ValueError(f"Distance slice has completely zero rows in {input_zip}: {zero_rows}")

    write_zip_from_array(output_zip, sliced, internal_name, overwrite=overwrite)
    return {
        "input_zip": str(input_zip),
        "input_internal_name": input_internal_name,
        "input_shape": list(arr.shape),
        "input_dtype": str(arr.dtype),
        "output_zip": str(output_zip),
        "output_internal_name": internal_name,
        "output_shape": list(sliced.shape),
        "output_dtype": str(sliced.dtype),
        "n_distance_features": n_distance_features,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Slice DCML combined feature.zip files into distance-only feature.zip files.")
    p.add_argument("--project-root", type=Path, default=Path("."), help="DCML project root. Default: current directory.")
    p.add_argument("--train-feature-zip", type=Path, default=None)
    p.add_argument("--test-feature-zip", type=Path, default=None)
    p.add_argument("--urv-feature-zip", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--n-distance-features", type=int, default=63360)
    p.add_argument("--cast-float32", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def resolve(root: Path, maybe_path: Path | None, default_rel: str) -> Path:
    path = maybe_path if maybe_path is not None else Path(default_rel)
    return path if path.is_absolute() else root / path


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    out_root = resolve(project_root, args.out_root, "data_urv/distance_only")

    train_in = resolve(project_root, args.train_feature_zip, "data/train_feature.zip")
    test_in = resolve(project_root, args.test_feature_zip, "data/test_feature.zip")
    urv_in = resolve(project_root, args.urv_feature_zip, "data_urv/dcml_input/MProURV_all_feature.zip")

    if args.n_distance_features <= 0:
        raise ValueError("--n-distance-features must be positive")

    outputs = {
        "created_at_utc": now_utc(),
        "project_root": str(project_root),
        "out_root": str(out_root),
        "n_distance_features": args.n_distance_features,
        "datasets": {},
    }

    outputs["datasets"]["train"] = slice_distance(
        input_zip=train_in,
        output_zip=out_root / "train_feature_distance.zip",
        n_distance_features=args.n_distance_features,
        internal_name="train_feature_distance.npy",
        cast_float32=args.cast_float32,
        overwrite=args.overwrite,
    )
    outputs["datasets"]["test"] = slice_distance(
        input_zip=test_in,
        output_zip=out_root / "test_feature_distance.zip",
        n_distance_features=args.n_distance_features,
        internal_name="test_feature_distance.npy",
        cast_float32=args.cast_float32,
        overwrite=args.overwrite,
    )
    outputs["datasets"]["urv_all"] = slice_distance(
        input_zip=urv_in,
        output_zip=out_root / "MProURV_all_feature_distance.zip",
        n_distance_features=args.n_distance_features,
        internal_name="MProURV_all_feature_distance.npy",
        cast_float32=args.cast_float32,
        overwrite=args.overwrite,
    )

    report_path = out_root / "distance_only_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
