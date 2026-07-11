"""GUI-facing DEAttentionDTA workflow functions.

The functions in this file adapt GUI dictionaries to the standalone scripts
kept in the maintained DEAttentionDTA package. The archival source tree is not
imported at runtime.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping

from .common import (
    MODULE_ROOT,
    as_absolute_string,
    ensure_dir,
    load_base_runner,
    load_finetune_runner,
    load_prepare_runner,
)

DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_MAX_SMI_LEN = 256


def _text(params: Mapping[str, Any], key: str, default: str = "") -> str:
    value = params.get(key, default)
    return str(value).strip()


def _integer(params: Mapping[str, Any], key: str, default: int) -> int:
    return int(params.get(key, default))


def _floating(params: Mapping[str, Any], key: str, default: float) -> float:
    return float(params.get(key, default))


def _boolean(params: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def prepare_urv_dataset(params: Mapping[str, Any]) -> dict[str, Any]:
    """Generate DEAttentionDTA prepared data from a raw MPro-v2-like root."""
    from .generate_deattentiondta_from_mpro_v2 import generate_deattentiondta_dataset_from_mpro_v2

    raw_root = _text(params, "raw_root", _text(params, "urv_v3b_dir", ""))
    output_root = _text(params, "output_root", _text(params, "out_dir", ""))
    max_smiles_len = params.get("max_smiles_len")
    max_protein_len = params.get("max_protein_len")
    max_smiles_len = int(max_smiles_len) if max_smiles_len not in (None, "", 0, "0") else None
    max_protein_len = int(max_protein_len) if max_protein_len not in (None, "", 0, "0") else None

    return generate_deattentiondta_dataset_from_mpro_v2(
        raw_root=raw_root,
        output_root=output_root,
        overwrite=_boolean(params, "overwrite", False),
        max_smiles_len=max_smiles_len,
        max_protein_len=max_protein_len,
        strict=_boolean(params, "strict", True),
    )

def _base_args(params: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        urv_dir=as_absolute_string(_text(params, "urv_v3b_dir", "DEAttentionDTA/data/urv_dataset_v3b")),
        prepared_dir=as_absolute_string(_text(params, "prepared_dir"), must_exist=True),
        models_dir=as_absolute_string(_text(params, "models_dir", "DEAttentionDTA/models/from_scratch")),
        results_dir=as_absolute_string(_text(params, "results_dir", _text(params, "output_dir", "DEAttentionDTA/outputs/from_scratch"))),
        zero_seq_policy="drop",
        skip_prepare=True,
        splits=_text(params, "splits", "all"),
        device=_text(params, "device", "auto"),
        epochs=_integer(params, "epochs", 50),
        batch_size=_integer(params, "batch_size", 16),
        lr=_floating(params, "lr", 1e-4),
        weight_decay=_floating(params, "weight_decay", 0.0),
        early_stopping_rounds=_integer(params, "early_stopping_rounds", 5),
        seed=_integer(params, "seed", 990721),
        num_workers=_integer(params, "num_workers", 0),
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        max_smi_len=DEFAULT_MAX_SMI_LEN,
    )


def debug_prepared_dataset(params: Mapping[str, Any]) -> dict[str, Any]:
    """Run a short forward pass using prepared URV data."""
    runner = load_base_runner()
    args = _base_args(params)
    report = runner.run_debug(args, str(MODULE_ROOT))
    report_path = ensure_dir(args.results_dir) / "debug_report.json"
    return {
        "status": "success",
        "operation": "debug_prepared_dataset",
        "summary": report,
        "artifacts": {"debug_report_json": str(report_path)},
    }


def train_official_splits(params: Mapping[str, Any]) -> dict[str, Any]:
    """Train independent DEAttentionDTA models on selected official splits."""
    runner = load_base_runner()
    args = _base_args(params)
    summaries = runner.run_all(args, str(MODULE_ROOT))
    results_dir = ensure_dir(args.results_dir)
    return {
        "status": "success",
        "operation": "train_official_splits",
        "summary": {"splits": summaries},
        "artifacts": {
            "summary_csv": str(results_dir / "Summary_5splits.csv"),
            "aggregate_metrics_json": str(results_dir / "Aggregate_metrics.json"),
            "models_dir": args.models_dir,
        },
    }


def _pretrained_args(params: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        prepared_dir=as_absolute_string(_text(params, "prepared_dir"), must_exist=True),
        pretrained_path=as_absolute_string(_text(params, "checkpoint"), must_exist=True),
        pretrained_fold=_text(params, "pretrained_fold", "matching"),
        models_dir=as_absolute_string(_text(params, "models_dir", "DEAttentionDTA/models/finetuned")),
        results_dir=as_absolute_string(_text(params, "results_dir", "DEAttentionDTA/outputs/pretrained_vs_finetuned")),
        splits=_text(params, "splits", "all"),
        device=_text(params, "device", "auto"),
        epochs=_integer(params, "epochs", 100),
        batch_size=_integer(params, "batch_size", 16),
        lr=_floating(params, "lr", 5e-5),
        weight_decay=_floating(params, "weight_decay", 0.0),
        early_stopping_rounds=_integer(params, "early_stopping_rounds", 15),
        seed=_integer(params, "seed", 990721),
        num_workers=_integer(params, "num_workers", 0),
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        max_smi_len=DEFAULT_MAX_SMI_LEN,
        non_strict_pretrained=_boolean(params, "non_strict_pretrained", False),
        eval_split_mode=_text(params, "eval_split_mode", _text(params, "split", "test")),
    )


def debug_pretrained_checkpoint(params: Mapping[str, Any]) -> dict[str, Any]:
    """Check checkpoint loading and execute one pretrained forward pass."""
    runner = load_finetune_runner()
    args = _pretrained_args(params)
    report = runner.run_debug_pretrained(args, str(MODULE_ROOT))
    report_path = ensure_dir(args.results_dir) / "debug_pretrained_report.json"
    return {
        "status": "success",
        "operation": "debug_pretrained_checkpoint",
        "summary": report,
        "artifacts": {"debug_pretrained_report_json": str(report_path)},
    }


def evaluate_checkpoint(params: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a checkpoint on selected official validation and test subsets."""
    runner = load_finetune_runner()
    args = _pretrained_args(params)
    summary_csv, aggregate_json = runner.run_zero_shot(args, str(MODULE_ROOT))
    return {
        "status": "success",
        "operation": "evaluate_checkpoint",
        "artifacts": {
            "summary_csv": str(summary_csv),
            "aggregate_metrics_json": str(aggregate_json),
            "results_dir": args.results_dir,
        },
    }


def finetune_pretrained_checkpoint(params: Mapping[str, Any]) -> dict[str, Any]:
    """Fine-tune the same pretrained checkpoint independently on each split."""
    runner = load_finetune_runner()
    args = _pretrained_args(params)
    summary_csv, aggregate_json = runner.run_finetune(args, str(MODULE_ROOT))
    return {
        "status": "success",
        "operation": "finetune_pretrained_checkpoint",
        "artifacts": {
            "summary_csv": str(summary_csv),
            "aggregate_metrics_json": str(aggregate_json),
            "models_dir": args.models_dir,
            "results_dir": args.results_dir,
        },
    }


def run_hyperparameter_search(params: Mapping[str, Any]) -> dict[str, Any]:
    """Run validation-only HPO without touching official test subsets."""
    from .hyperparameter_search import run_hyperparameter_search as _run

    return _run(params)
