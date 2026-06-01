#!/usr/bin/env python3

from __future__ import annotations

import pickletools
import re
from pathlib import Path

import sklearn
import torch


MODEL_PATHS = [
    Path("DCML/models/pretrained/DCML_distance_only.pt"),
    Path("DCML/models/pretrained/DCML_full.pt"),
]


def load_bundle(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_sklearn_versions(blob: bytes) -> list[str]:
    versions: set[str] = set()

    try:
        operations = list(pickletools.genops(blob))

        for index, (_, argument, _) in enumerate(operations):
            if argument != "_sklearn_version":
                continue

            for _, nearby_argument, _ in operations[index + 1 : index + 10]:
                if isinstance(nearby_argument, str) and re.fullmatch(
                    r"\d+\.\d+(?:\.\d+)?(?:[a-zA-Z0-9_.+-]*)?",
                    nearby_argument,
                ):
                    versions.add(nearby_argument)
                    break
    except Exception as exc:
        print("pickletools warning:", exc)

    if not versions:
        for match in re.findall(
            rb"(?<!\d)\d+\.\d+(?:\.\d+)?(?:[a-zA-Z0-9_.+-]*)?",
            blob,
        ):
            versions.add(match.decode("utf-8", errors="replace"))

    return sorted(versions)


print("Current scikit-learn version:", sklearn.__version__)

for path in MODEL_PATHS:
    print("\n" + "=" * 80)
    print("MODEL:", path)

    if not path.is_file():
        print("ERROR: file not found")
        continue

    bundle = load_bundle(path)

    if not isinstance(bundle, dict):
        print("ERROR: bundle is not a dictionary")
        continue

    print("Bundle keys:", sorted(bundle.keys()))
    print("model_name:", bundle.get("model_name"))
    print("backend:", bundle.get("backend"))
    print("estimator_type:", bundle.get("estimator_type"))
    print("n_features:", bundle.get("n_features"))
    print("hyperparameters:", bundle.get("hyperparameters"))

    blob = bundle.get("serialized_estimator")

    if not isinstance(blob, (bytes, bytearray)):
        print("ERROR: serialized_estimator is missing or invalid")
        continue

    print("Detected scikit-learn versions inside estimator pickle:")
    print(extract_sklearn_versions(bytes(blob)))
