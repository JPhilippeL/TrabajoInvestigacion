"""Validation-only hyperparameter search for DEAttentionDTA.

The official test subsets are intentionally excluded from every trial. They are
reserved for the final experiment executed after a configuration has been
selected.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .common import MODULE_ROOT, as_absolute_string, ensure_dir, load_base_runner, save_json


class DEAttentionDTAHPOError(RuntimeError):
    """Raised when validation-only hyperparameter search cannot complete."""


def _parse_splits(value: str) -> list[int]:
    text = str(value).strip().lower()

    if text in {"all", "*"}:
        return [1, 2, 3, 4, 5]

    split_ids: list[int] = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        split_id = int(item)

        if split_id < 1 or split_id > 5:
            raise ValueError("Split id must be between 1 and 5.")

        split_ids.append(split_id)

    split_ids = sorted(set(split_ids))

    if not split_ids:
        raise ValueError("At least one tuning split is required.")

    return split_ids


def _values(params: Mapping[str, Any], key: str, caster) -> list[Any]:
    raw = params.get(key, [])

    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]

    values = [caster(item) for item in raw]

    if not values:
        raise ValueError(f"At least one {key} value is required.")

    return values


def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except Exception:
        return fallback

    return number if math.isfinite(number) else fallback


def _format_elapsed_ddhhmmss(seconds: float) -> str:
    """Convert elapsed seconds to DD:HH:MM:SS format."""
    total_seconds = int(round(float(seconds)))

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


def _is_better(candidate: Mapping[str, Any], current: Mapping[str, Any] | None) -> bool:
    if candidate.get("status") != "success":
        return False

    if current is None:
        return True

    candidate_rmse = _finite_or(candidate.get("mean_val_RMSE"), float("inf"))
    current_rmse = _finite_or(current.get("mean_val_RMSE"), float("inf"))

    if candidate_rmse != current_rmse:
        return candidate_rmse < current_rmse

    candidate_pearson = _finite_or(candidate.get("mean_val_Pearson"), float("-inf"))
    current_pearson = _finite_or(current.get("mean_val_Pearson"), float("-inf"))

    return candidate_pearson > current_pearson


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        path.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return path


def _write_trials_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "trial_id",
        "status",
        "lr",
        "batch_size",
        "weight_decay",
        "tuning_splits",
        "mean_val_RMSE",
        "mean_val_Pearson",
        "mean_val_MAE",
        "mean_best_epoch",
        "elapsed_seconds",
        "elapsed_time",
        "elapsed_time_format",
        "models_dir",
        "trial_dir",
        "error_message",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    return path


def _build_grid(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    learning_rates = _values(params, "lr_values", float)
    batch_sizes = _values(params, "batch_size_values", int)
    weight_decays = _values(params, "weight_decay_values", float)

    grid: list[dict[str, Any]] = []

    for lr, batch_size, weight_decay in itertools.product(
        learning_rates,
        batch_sizes,
        weight_decays,
    ):
        if lr <= 0:
            raise ValueError("Learning rates must be > 0.")

        if batch_size < 1:
            raise ValueError("Batch sizes must be >= 1.")

        if weight_decay < 0:
            raise ValueError("Weight decays must be >= 0.")

        grid.append(
            {
                "lr": float(lr),
                "batch_size": int(batch_size),
                "weight_decay": float(weight_decay),
            }
        )

    return grid


def _train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm: float) -> float:
    import torch

    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        _, smi, seq, pocket, affinity = batch

        smi = smi.to(device)
        seq = seq.to(device)
        pocket = pocket.to(device)
        affinity = affinity.to(device).reshape(-1)

        optimizer.zero_grad(set_to_none=True)

        output = model(seq, smi, pocket).reshape(-1)
        loss = criterion(output, affinity)

        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        n = int(affinity.numel())
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / float(total_n) if total_n else float("nan")


def _run_one_split(
    *,
    base,
    model_cls,
    split_id: int,
    trial_id: str,
    hparams: Mapping[str, Any],
    prepared_dir: Path,
    trial_result_dir: Path,
    trial_model_dir: Path,
    device,
    epochs: int,
    early_stopping_rounds: int,
    min_epochs_before_stopping: int,
    min_delta: float,
    grad_clip_norm: float,
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    import pandas as pd
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    split_result_dir = ensure_dir(trial_result_dir / f"split_{split_id:02d}")
    split_model_dir = ensure_dir(trial_model_dir / f"split_{split_id:02d}")
    paths = base.split_paths(str(prepared_dir), split_id)

    train_dataset = base.URVDEAttentionDataset(
        paths["train_seq"],
        paths["train_aff"],
        base.DEFAULT_MAX_SEQ_LEN,
        base.DEFAULT_MAX_SMI_LEN,
    )
    valid_dataset = base.URVDEAttentionDataset(
        paths["valid_seq"],
        paths["valid_aff"],
        base.DEFAULT_MAX_SEQ_LEN,
        base.DEFAULT_MAX_SMI_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(hparams["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(hparams["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
    )

    base.set_seed(seed)

    model = model_cls().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    criterion = nn.MSELoss()

    best_state = None
    best_metrics = None
    best_epoch = -1
    best_rmse = float("inf")
    rounds_without_improvement = 0
    history: list[dict[str, Any]] = []
    started = time.time()

    print(
        f"{trial_id} split {split_id:02d} | "
        f"train={len(train_dataset)} valid={len(valid_dataset)} "
        f"lr={hparams['lr']} batch={hparams['batch_size']} "
        f"wd={hparams['weight_decay']} device={device}"
    )

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm,
        )

        val_metrics, _, _, _ = base.evaluate(model, valid_loader, criterion, device)
        current_rmse = float(val_metrics["RMSE"])
        improved = current_rmse < best_rmse - min_delta

        if improved:
            best_rmse = current_rmse
            best_metrics = {name: float(value) for name, value in val_metrics.items()}
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_RMSE": float(val_metrics["RMSE"]),
                "val_MAE": float(val_metrics["MAE"]),
                "val_Pearson": float(val_metrics["Pearson"]),
                "is_best": bool(improved),
            }
        )

        print(
            f"{trial_id} split {split_id:02d} epoch {epoch:03d}/{epochs:03d} "
            f"val_RMSE={val_metrics['RMSE']:.6f} "
            f"val_Pearson={val_metrics['Pearson']:.6f}"
            + (" [best]" if improved else "")
        )

        if (
            epoch >= min_epochs_before_stopping
            and rounds_without_improvement >= early_stopping_rounds
        ):
            print(
                f"{trial_id} split {split_id:02d} early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}"
            )
            break

    if best_state is None or best_metrics is None:
        raise DEAttentionDTAHPOError(
            f"{trial_id} split {split_id} did not produce a valid checkpoint."
        )

    history_csv = split_result_dir / "train_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)

    model_path = split_model_dir / f"DEAttentionDTA_HPO_{trial_id}_split{split_id:02d}.pt"

    torch.save(
        {
            "format_version": "deattentiondta_hpo_v1",
            "model_name": "DEAttentionDTA",
            "workflow": "validation_only_hpo",
            "trial_id": trial_id,
            "split_id": int(split_id),
            "best_epoch": int(best_epoch),
            "state_dict": best_state,
            "hyperparameters": dict(hparams),
            "metrics": {
                "validation": best_metrics,
            },
            "test_rows_used": 0,
        },
        model_path,
    )

    elapsed_seconds = float(time.time() - started)

    summary = {
        "trial_id": trial_id,
        "split": split_id,
        "model_pt": str(model_path),
        "history_csv": str(history_csv),
        "best_epoch": int(best_epoch),
        "val_RMSE": float(best_metrics["RMSE"]),
        "val_MAE": float(best_metrics["MAE"]),
        "val_Pearson": float(best_metrics["Pearson"]),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_time": _format_elapsed_ddhhmmss(elapsed_seconds),
        "elapsed_time_format": "DD:HH:MM:SS",
        "train_rows": int(len(train_dataset)),
        "validation_rows": int(len(valid_dataset)),
        "test_rows_used": 0,
    }

    save_json(summary, split_result_dir / "split_summary.json")

    return summary


def run_hyperparameter_search(params: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the validation-only search and return generated artifacts."""
    import pandas as pd

    base = load_base_runner()

    prepared_dir = Path(
        as_absolute_string(
            str(params.get("prepared_dir", "")),
            must_exist=True,
        )
    )
    models_root = ensure_dir(str(params.get("models_root", "DEAttentionDTA/models/hpo")))
    results_root = ensure_dir(str(params.get("results_root", "DEAttentionDTA/outputs/hpo")))
    tuning_splits = _parse_splits(str(params.get("splits", "1")))
    grid = _build_grid(params)
    device = base.choose_device(str(params.get("device", "auto")))

    epochs = int(params.get("epochs", 50))
    early_stopping_rounds = int(params.get("early_stopping_rounds", 10))
    min_epochs_before_stopping = int(params.get("min_epochs_before_stopping", 5))
    min_delta = float(params.get("min_delta", 0.0))
    grad_clip_norm = float(params.get("grad_clip_norm", 0.0))
    seed = int(params.get("seed", 42))
    num_workers = int(params.get("num_workers", 0))

    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be >= 1.")

    if min_epochs_before_stopping < 1:
        raise ValueError("min_epochs_before_stopping must be >= 1.")

    check_args = SimpleNamespace(prepared_dir=str(prepared_dir))
    base.ensure_prepared_exists(check_args, str(MODULE_ROOT))
    model_cls, model_source_path = base.load_model_class(str(MODULE_ROOT))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(results_root / f"deattentiondta_hpo_{timestamp}")
    trial_models_root = ensure_dir(models_root / f"deattentiondta_hpo_{timestamp}")
    trials_csv = run_dir / "trials.csv"

    save_json(
        {
            "model_name": "DEAttentionDTA",
            "workflow": "validation_only_hpo",
            "methodology": "official train and validation subsets only; official test subsets excluded",
            "selection_rule": "min mean validation RMSE, then max mean validation Pearson",
            "prepared_dir": str(prepared_dir),
            "model_source_path": str(model_source_path),
            "device": str(device),
            "tuning_splits": tuning_splits,
            "search_grid": grid,
            "n_trials": len(grid),
            "test_rows_used": 0,
        },
        run_dir / "search_config.json",
    )

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    started = time.time()

    for trial_number, hparams in enumerate(grid, start=1):
        trial_id = f"trial_{trial_number:04d}"
        trial_result_dir = ensure_dir(run_dir / trial_id)
        trial_model_dir = ensure_dir(trial_models_root / trial_id)

        row: dict[str, Any] = {
            "trial_id": trial_id,
            "status": "failure",
            "lr": hparams["lr"],
            "batch_size": hparams["batch_size"],
            "weight_decay": hparams["weight_decay"],
            "tuning_splits": ",".join(str(item) for item in tuning_splits),
            "models_dir": str(trial_model_dir),
            "trial_dir": str(trial_result_dir),
            "error_message": "",
        }

        try:
            print(f"Starting DEAttentionDTA HPO {trial_number}/{len(grid)} | {hparams}")

            split_rows = []

            for split_id in tuning_splits:
                split_rows.append(
                    _run_one_split(
                        base=base,
                        model_cls=model_cls,
                        split_id=split_id,
                        trial_id=trial_id,
                        hparams=hparams,
                        prepared_dir=prepared_dir,
                        trial_result_dir=trial_result_dir,
                        trial_model_dir=trial_model_dir,
                        device=device,
                        epochs=epochs,
                        early_stopping_rounds=early_stopping_rounds,
                        min_epochs_before_stopping=min_epochs_before_stopping,
                        min_delta=min_delta,
                        grad_clip_norm=grad_clip_norm,
                        seed=seed + trial_number * 1000 + split_id,
                        num_workers=num_workers,
                    )
                )

            pd.DataFrame(split_rows).to_csv(
                trial_result_dir / "split_summaries.csv",
                index=False,
            )

            frame = pd.DataFrame(split_rows)
            trial_elapsed_seconds = float(frame["elapsed_seconds"].sum())

            row.update(
                {
                    "status": "success",
                    "mean_val_RMSE": float(frame["val_RMSE"].mean()),
                    "mean_val_Pearson": float(frame["val_Pearson"].mean()),
                    "mean_val_MAE": float(frame["val_MAE"].mean()),
                    "mean_best_epoch": float(frame["best_epoch"].mean()),
                    "elapsed_seconds": trial_elapsed_seconds,
                    "elapsed_time": _format_elapsed_ddhhmmss(trial_elapsed_seconds),
                    "elapsed_time_format": "DD:HH:MM:SS",
                }
            )

            save_json(row, trial_result_dir / "trial_summary.json")

            if _is_better(row, best_row):
                best_row = dict(row)

        except Exception as exc:
            row["error_message"] = str(exc)
            (trial_result_dir / "error_traceback.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(f"DEAttentionDTA HPO {trial_id} failed: {exc}")

        finally:
            rows.append(row)
            _write_trials_csv(trials_csv, rows)

    if best_row is None:
        raise DEAttentionDTAHPOError(
            "All DEAttentionDTA HPO trials failed. Inspect trials.csv."
        )

    best_models_dir = run_dir / "best_models"

    if best_models_dir.exists():
        shutil.rmtree(best_models_dir)

    shutil.copytree(Path(str(best_row["models_dir"])), best_models_dir)

    total_elapsed_seconds = float(time.time() - started)

    payload = {
        "model_name": "DEAttentionDTA",
        "status": "success",
        "workflow": "validation_only_hpo",
        "methodology": "official train and validation subsets only; official test subsets excluded",
        "selection_rule": {
            "primary_metric": "mean validation RMSE",
            "primary_goal": "min",
            "secondary_metric": "mean validation Pearson",
            "secondary_goal": "max",
        },
        "best_trial": best_row["trial_id"],
        "best_hyperparameters": {
            "lr": float(best_row["lr"]),
            "batch_size": int(best_row["batch_size"]),
            "weight_decay": float(best_row["weight_decay"]),
        },
        "best_metrics": {
            "mean_val_RMSE": float(best_row["mean_val_RMSE"]),
            "mean_val_Pearson": float(best_row["mean_val_Pearson"]),
            "mean_val_MAE": float(best_row["mean_val_MAE"]),
            "mean_best_epoch": float(best_row["mean_best_epoch"]),
        },
        "tuning_splits": tuning_splits,
        "n_trials": len(rows),
        "n_success": sum(1 for row in rows if row.get("status") == "success"),
        "n_failures": sum(1 for row in rows if row.get("status") != "success"),
        "elapsed_seconds": total_elapsed_seconds,
        "elapsed_time": _format_elapsed_ddhhmmss(total_elapsed_seconds),
        "elapsed_time_format": "DD:HH:MM:SS",
        "test_rows_used": 0,
        "paths": {
            "run_dir": str(run_dir),
            "trials_csv": str(trials_csv),
            "best_models_dir": str(best_models_dir),
        },
    }

    best_yaml = _write_yaml(run_dir / "best_config_deattentiondta.yaml", payload)
    best_json = save_json(payload, run_dir / "best_config_deattentiondta.json")
    save_json(payload, results_root / "latest_run.json")

    print("DEAttentionDTA hyperparameter search completed")
    print(f"  best trial:           {payload['best_trial']}")
    print(f"  best hyperparameters: {payload['best_hyperparameters']}")
    print(f"  best mean val RMSE:   {payload['best_metrics']['mean_val_RMSE']:.6f}")
    print(f"  best mean Pearson:    {payload['best_metrics']['mean_val_Pearson']:.6f}")
    print(f"  elapsed time:         {payload['elapsed_time']} ({payload['elapsed_time_format']})")

    return {
        "status": "success",
        "operation": "hyperparameter_search",
        "summary": payload,
        "artifacts": {
            "run_dir": str(run_dir),
            "trials_csv": str(trials_csv),
            "best_config_yaml": str(best_yaml),
            "best_config_json": str(best_json),
            "best_models_dir": str(best_models_dir),
        },
    }
