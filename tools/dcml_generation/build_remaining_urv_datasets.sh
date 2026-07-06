#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-2}"

INPUT_ROOT="$ROOT/data_urv/dcml_input"
PQR_ROOT="$ROOT/data_urv/pdbbind_like/URV/pqr"
REAL_CHARGE_REPORT_ROOT="$ROOT/data_urv/results/real_charge_features"

DISTANCE_DATASET="$ROOT/DCML/datasets/urv_distance_only"
ZERO_DATASET="$ROOT/DCML/datasets/urv_full_zero_charge"
PQR_DATASET="$ROOT/DCML/datasets/urv_full_pqr"

SCRIPT_CHARGES="$ROOT/tools/dcml_generation/prepare_urv_real_charges.py"
SCRIPT_FEATURES="$ROOT/tools/dcml_generation/generate_urv_dcml_features.py"
SCRIPT_READY="$ROOT/tools/create_dcml_ready_dataset.py"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 1
  fi
}

echo "[0/8] Checking required files"

require_file "$SCRIPT_CHARGES"
require_file "$SCRIPT_FEATURES"
require_file "$SCRIPT_READY"

require_file "$INPUT_ROOT/MProURV_all_feature_distance_only.npy"
require_file "$INPUT_ROOT/MProURV_all_label.npy"
require_file "$INPUT_ROOT/MProURV_all_sample_ids.csv"
require_file "$DISTANCE_DATASET/split_indices.json"

command -v obabel >/dev/null || {
  echo "ERROR: missing obabel" >&2
  exit 1
}

if ! command -v pdb2pqr >/dev/null && ! command -v pdb2pqr30 >/dev/null; then
  echo "ERROR: missing pdb2pqr or pdb2pqr30" >&2
  exit 1
fi

mkdir -p "$ZERO_DATASET" "$PQR_DATASET" "$REAL_CHARGE_REPORT_ROOT"

echo
echo "[1/8] Creating full zero-charge matrix from existing distance-only matrix"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

src = Path("data_urv/dcml_input/MProURV_all_feature_distance_only.npy")
out_npy = Path("data_urv/dcml_input/MProURV_all_feature_zero_charge.npy")
out_zip = Path("data_urv/dcml_input/MProURV_all_feature_zero_charge.zip")
report_path = Path("data_urv/dcml_input/MProURV_all_feature_zero_charge.report.json")

distance_features = 63360
charge_features = 55000
full_features = distance_features + charge_features
expected_samples = 378

X = np.load(src, mmap_mode="r")

if X.shape != (expected_samples, distance_features):
    raise ValueError(
        f"Unexpected distance-only shape: {X.shape} != "
        f"({expected_samples}, {distance_features})"
    )

if not np.isfinite(X).all():
    raise ValueError("Distance-only matrix contains NaN or infinite values.")

if np.any(np.all(X == 0, axis=1)):
    raise ValueError("Distance-only matrix contains completely empty rows.")

out_npy.parent.mkdir(parents=True, exist_ok=True)

Y = np.lib.format.open_memmap(
    out_npy,
    mode="w+",
    dtype=np.float32,
    shape=(expected_samples, full_features),
)

Y[:, :distance_features] = X.astype(np.float32, copy=False)
Y[:, distance_features:] = 0.0
Y.flush()

with zipfile.ZipFile(
    out_zip,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as archive:
    archive.write(out_npy, arcname="MProURV_all_feature_zero_charge.npy")

summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "variant_id": "urv_full_zero_charge",
    "source_distance_npy": str(src),
    "out_npy": str(out_npy),
    "out_zip": str(out_zip),
    "shape": [expected_samples, full_features],
    "dtype": "float32",
    "distance_features": distance_features,
    "charge_features": charge_features,
    "finite": bool(np.isfinite(Y).all()),
    "zero_rows": int(np.all(Y == 0, axis=1).sum()),
    "distance_nonzero": int(np.count_nonzero(Y[:, :distance_features])),
    "charge_nonzero": int(np.count_nonzero(Y[:, distance_features:])),
    "charge_sum_abs": float(np.abs(Y[:, distance_features:]).sum()),
}

report_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))

assert summary["finite"] is True
assert summary["zero_rows"] == 0
assert summary["distance_nonzero"] > 0
assert summary["charge_nonzero"] == 0
assert summary["charge_sum_abs"] == 0.0
PY

echo
echo "[2/8] Creating ready-to-train urv_full_zero_charge dataset"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$SCRIPT_READY" \
  --dataset-root "$ZERO_DATASET" \
  --all-feature-zip "$INPUT_ROOT/MProURV_all_feature_zero_charge.zip" \
  --all-label-npy "$INPUT_ROOT/MProURV_all_label.npy" \
  --sample-ids-csv "$INPUT_ROOT/MProURV_all_sample_ids.csv" \
  --variant-id urv_full_zero_charge \
  --feature-mode full \
  --charge-mode zeros \
  --expected-samples 378 \
  --expected-features 118360 \
  --distance-features 63360 \
  --charge-features 55000 \
  --expect-zero-charge \
  --split-source "$DISTANCE_DATASET/split_indices.json" \
  --report "$INPUT_ROOT/MProURV_all_feature_zero_charge.report.json"

echo
echo "[3/8] Repairing known PDB2PQR edge cases: 8ACL,8Q71"

"$PYTHON_BIN" "$SCRIPT_CHARGES" \
  --work-root "$ROOT/data_urv" \
  --ids 8ACL,8Q71 \
  --pqr-root "$PQR_ROOT" \
  --ff AMBER \
  --ph 7.0 \
  --ligand-charge-method gasteiger \
  --protein-fallback auto \
  --protein-charge-method gasteiger \
  --workers 1 \
  --force-pqr

echo
echo "[4/8] Preparing real PQR and ligand charges for all URV complexes"

"$PYTHON_BIN" "$SCRIPT_CHARGES" \
  --work-root "$ROOT/data_urv" \
  --pqr-root "$PQR_ROOT" \
  --ff AMBER \
  --ph 7.0 \
  --ligand-charge-method gasteiger \
  --protein-fallback auto \
  --protein-charge-method gasteiger \
  --workers "$WORKERS"

echo
echo "[5/8] Generating full PQR feature matrix — this is the long step"

"$PYTHON_BIN" "$SCRIPT_FEATURES" \
  --work-root "$ROOT/data_urv" \
  --out-npy "$INPUT_ROOT/MProURV_all_feature_real_charge.npy" \
  --out-zip "$INPUT_ROOT/MProURV_all_feature_real_charge.zip" \
  --write-zip \
  --output-mode full \
  --charge-mode pqr \
  --pqr-root "$PQR_ROOT" \
  --dtype float32 \
  --workers "$WORKERS" \
  --force

echo
echo "[6/8] Validating full PQR feature matrix"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

feature_npy = Path("data_urv/dcml_input/MProURV_all_feature_real_charge.npy")
validation_path = Path(
    "data_urv/results/real_charge_features/"
    "real_charge_feature_validation.json"
)

X = np.load(feature_npy, mmap_mode="r")

summary = {
    "feature_npy": str(feature_npy),
    "feature_zip": (
        "data_urv/dcml_input/"
        "MProURV_all_feature_real_charge.zip"
    ),
    "shape": list(X.shape),
    "dtype": str(X.dtype),
    "finite": bool(np.isfinite(X).all()),
    "zero_rows": int(np.all(X == 0, axis=1).sum()),
    "distance_nonzero": int(np.count_nonzero(X[:, :63360])),
    "charge_nonzero": int(np.count_nonzero(X[:, 63360:])),
    "charge_sum_abs": float(np.abs(X[:, 63360:]).sum()),
}

validation_path.parent.mkdir(parents=True, exist_ok=True)
validation_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))

assert X.shape == (378, 118360), X.shape
assert summary["finite"] is True
assert summary["zero_rows"] == 0
assert summary["distance_nonzero"] > 0
assert summary["charge_nonzero"] > 0
assert summary["charge_sum_abs"] > 0.0
PY

echo
echo "[7/8] Creating ready-to-train urv_full_pqr dataset"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$SCRIPT_READY" \
  --dataset-root "$PQR_DATASET" \
  --all-feature-zip "$INPUT_ROOT/MProURV_all_feature_real_charge.zip" \
  --all-label-npy "$INPUT_ROOT/MProURV_all_label.npy" \
  --sample-ids-csv "$INPUT_ROOT/MProURV_all_sample_ids.csv" \
  --variant-id urv_full_pqr \
  --feature-mode full \
  --charge-mode pqr \
  --expected-samples 378 \
  --expected-features 118360 \
  --distance-features 63360 \
  --charge-features 55000 \
  --require-nonzero-charge \
  --split-source "$DISTANCE_DATASET/split_indices.json" \
  --report "$INPUT_ROOT/MProURV_all_feature_real_charge.report.json" \
  --report "$REAL_CHARGE_REPORT_ROOT/real_charge_feature_validation.json" \
  --report "$ROOT/data_urv/reports/real_charge_preparation/real_charge_preparation_report.csv" \
  --report "$ROOT/data_urv/reports/real_charge_preparation/real_charge_preparation_report.json"

echo
echo "[8/8] Final readiness summary"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

roots = [
    Path("DCML/datasets/urv_distance_only"),
    Path("DCML/datasets/urv_full_zero_charge"),
    Path("DCML/datasets/urv_full_pqr"),
]

for root in roots:
    status_path = root / "dataset_validation_status.json"

    print()
    print("=" * 80)
    print("DATASET:", root.name)

    if not status_path.is_file():
        print("STATUS: missing", status_path)
        continue

    status = json.loads(status_path.read_text(encoding="utf-8"))

    print("samples:", status["validation"]["n_samples"])
    print("features:", status["validation"]["n_features"])
    print("charge:", status["validation"].get("charge_validation"))
    print("readiness:", status["readiness"])
PY

echo
echo "DONE."
