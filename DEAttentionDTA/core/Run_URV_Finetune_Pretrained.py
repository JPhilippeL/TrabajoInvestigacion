#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune or evaluate a pretrained DEAttentionDTA checkpoint on the
URV official 5-split protocol used in this TFM.

Python 3.6 compatible.

This script reuses the dataset preparation and evaluation utilities from:

    Run_URV_5Splits.py

It does NOT regenerate Position/Pocket. It expects that the prepared dataset was
already created with:

    Prepare_URV_Positions_From_V2_Dataset.py

Default pretrained checkpoint:

    models/pretrained/DEAttentionDTA.pt

Default outputs:

    models/finetuned/
    outputs/pretrained_vs_finetuned/

Examples
--------
Debug checkpoint loading and one forward pass:

    python DEAttentionDTA/core/Run_URV_Finetune_Pretrained.py \
      --mode debug-pretrained --device cuda --batch-size 2

Evaluate the pretrained model directly on the five URV test splits, without
training:

    python DEAttentionDTA/core/Run_URV_Finetune_Pretrained.py \
      --mode zero-shot --device cuda --batch-size 16

Fine-tune from the same pretrained checkpoint independently for each split:

    python DEAttentionDTA/core/Run_URV_Finetune_Pretrained.py \
      --mode finetune --device cuda --epochs 100 --batch-size 16 \
      --early-stopping-rounds 15 --lr 0.00005
"""

from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


SCRIPT_NAME = "Run_URV_Finetune_Pretrained"
MODEL_NAME = "DEAttentionDTA"


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_base_runner():
    base_path = os.path.join(script_dir(), "Run_URV_5Splits.py")
    if not os.path.isfile(base_path):
        raise IOError("Base runner not found: {0}".format(base_path))
    spec = importlib.util.spec_from_file_location("urv_base_runner", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()


def join_root(repo_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def choose_fold_from_fold_state_dicts(fold_state_dicts, pretrained_fold, target_split):
    """Select one state_dict from a checkpoint that stores multiple folds.

    Supports common aggregate formats:
      - list/tuple: fold 1 is index 0
      - dict: keys may be 1, "1", "fold_1", "split_01", etc.

    pretrained_fold can be:
      - "matching": use target_split
      - "first": use the first available fold
      - an integer-like string, e.g. "1"
    """
    if pretrained_fold is None:
        pretrained_fold = "matching"
    pretrained_fold = str(pretrained_fold).strip().lower()

    desired = None
    if pretrained_fold in ["matching", "match", "same"]:
        desired = int(target_split) if target_split is not None else 1
    elif pretrained_fold in ["first", "0"]:
        desired = 1
    else:
        try:
            desired = int(pretrained_fold)
        except Exception:
            raise ValueError("Invalid --pretrained-fold value: {0}".format(pretrained_fold))

    if isinstance(fold_state_dicts, (list, tuple)):
        if desired < 1 or desired > len(fold_state_dicts):
            raise ValueError("Requested pretrained fold {0}, but checkpoint has {1} folds".format(desired, len(fold_state_dicts)))
        return fold_state_dicts[desired - 1], "fold_state_dicts:list:fold_{0}".format(desired)

    if isinstance(fold_state_dicts, dict):
        candidate_keys = [
            desired,
            str(desired),
            "fold_{0}".format(desired),
            "fold{0}".format(desired),
            "split_{0}".format(desired),
            "split_{0:02d}".format(desired),
            "fold_{0:02d}".format(desired),
        ]
        for key in candidate_keys:
            if key in fold_state_dicts:
                return fold_state_dicts[key], "fold_state_dicts:dict:{0}".format(key)

        # Fallback: sort keys and select by 1-based position.
        keys = sorted(list(fold_state_dicts.keys()), key=lambda x: str(x))
        if desired < 1 or desired > len(keys):
            raise ValueError("Requested pretrained fold {0}, but checkpoint fold_state_dicts keys are: {1}".format(
                desired, [str(k) for k in keys]
            ))
        key = keys[desired - 1]
        return fold_state_dicts[key], "fold_state_dicts:dict_sorted:{0}".format(key)

    raise ValueError("Unsupported fold_state_dicts type: {0}".format(type(fold_state_dicts)))


def get_state_dict_from_checkpoint(checkpoint_obj, pretrained_fold=None, target_split=None, _depth=0):
    """Return a state_dict from common PyTorch checkpoint formats.

    Also supports aggregate checkpoints with a top-level `fold_state_dicts`
    field. In that case one fold is selected using --pretrained-fold.
    """
    if _depth > 5:
        raise ValueError("Checkpoint nesting is too deep; could not resolve state_dict")

    if isinstance(checkpoint_obj, torch.nn.Module):
        return checkpoint_obj.state_dict(), "torch_nn_module"

    if not isinstance(checkpoint_obj, dict):
        raise ValueError("Unsupported checkpoint type: {0}".format(type(checkpoint_obj)))

    if "fold_state_dicts" in checkpoint_obj:
        fold_obj, fold_format = choose_fold_from_fold_state_dicts(
            checkpoint_obj["fold_state_dicts"], pretrained_fold, target_split
        )
        raw_state, nested_format = get_state_dict_from_checkpoint(
            fold_obj, pretrained_fold=pretrained_fold, target_split=target_split, _depth=_depth + 1
        )
        return raw_state, "{0}->{1}".format(fold_format, nested_format)

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "module",
    ]
    for key in candidate_keys:
        if key in checkpoint_obj:
            value = checkpoint_obj[key]
            if isinstance(value, torch.nn.Module):
                return value.state_dict(), "dict_key:{0}:module".format(key)
            if isinstance(value, dict):
                # The value may be a raw state_dict, or another nested checkpoint.
                try:
                    nested_state, nested_format = get_state_dict_from_checkpoint(
                        value, pretrained_fold=pretrained_fold, target_split=target_split, _depth=_depth + 1
                    )
                    return nested_state, "dict_key:{0}->{1}".format(key, nested_format)
                except Exception:
                    return value, "dict_key:{0}".format(key)

    # Raw state_dict case: most values are tensors.
    tensor_like = 0
    total = 0
    for value in checkpoint_obj.values():
        total += 1
        if torch.is_tensor(value):
            tensor_like += 1
    if total > 0 and tensor_like >= max(1, int(0.5 * total)):
        return checkpoint_obj, "raw_state_dict"

    raise ValueError("Could not find a model state_dict inside checkpoint. Top-level keys: {0}".format(
        sorted([str(k) for k in checkpoint_obj.keys()])[:50]
    ))


def strip_prefix_if_present(state_dict, prefix):
    keys = list(state_dict.keys())
    if not keys:
        return state_dict, False
    prefixed = [key for key in keys if str(key).startswith(prefix)]
    if len(prefixed) == len(keys):
        out = {}
        for key, value in state_dict.items():
            out[str(key)[len(prefix):]] = value
        return out, True
    return state_dict, False


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[str(key)] = value

    prefixes_removed = []
    for prefix in ["module.", "model.", "net."]:
        cleaned, removed = strip_prefix_if_present(cleaned, prefix)
        if removed:
            prefixes_removed.append(prefix)
    return cleaned, prefixes_removed


def compatible_key_count(model, state_dict):
    own = model.state_dict()
    compatible = 0
    shape_mismatch = []
    unknown = []
    for key, value in state_dict.items():
        if key not in own:
            unknown.append(key)
            continue
        try:
            if tuple(own[key].size()) == tuple(value.size()):
                compatible += 1
            else:
                shape_mismatch.append(key)
        except Exception:
            shape_mismatch.append(key)
    return compatible, shape_mismatch, unknown


def load_pretrained_weights(model, pretrained_path, device, strict, pretrained_fold=None, target_split=None):
    if not os.path.isfile(pretrained_path):
        raise IOError("Pretrained checkpoint not found: {0}".format(pretrained_path))

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    raw_state, checkpoint_format = get_state_dict_from_checkpoint(
        checkpoint, pretrained_fold=pretrained_fold, target_split=target_split
    )
    state, prefixes_removed = clean_state_dict_keys(raw_state)

    compatible, shape_mismatch, unknown = compatible_key_count(model, state)
    report = {
        "pretrained_path": pretrained_path,
        "checkpoint_format": checkpoint_format,
        "pretrained_fold_argument": str(pretrained_fold),
        "target_split": int(target_split) if target_split is not None else None,
        "prefixes_removed": prefixes_removed,
        "strict": bool(strict),
        "checkpoint_keys": int(len(state)),
        "compatible_keys": int(compatible),
        "shape_mismatch_keys": sorted(shape_mismatch),
        "unknown_checkpoint_keys": sorted(unknown),
        "missing_model_keys": [],
        "unexpected_checkpoint_keys": [],
    }

    if strict:
        model.load_state_dict(state, strict=True)
    else:
        incompatible = model.load_state_dict(state, strict=False)
        if hasattr(incompatible, "missing_keys"):
            report["missing_model_keys"] = sorted(list(incompatible.missing_keys))
        if hasattr(incompatible, "unexpected_keys"):
            report["unexpected_checkpoint_keys"] = sorted(list(incompatible.unexpected_keys))

    model.to(device)
    return report


def ensure_prepared_dataset(args, repo_root):
    prepared_dir = join_root(repo_root, args.prepared_dir)
    # Use the base runner's strict checker. It checks all five splits.
    BASE.ensure_prepared_exists(args, repo_root)
    return prepared_dir


def build_datasets(args, prepared_dir, split_id):
    paths = BASE.split_paths(prepared_dir, split_id)
    for key in paths:
        if not os.path.isfile(paths[key]):
            raise IOError("Missing prepared split file {0}: {1}".format(key, paths[key]))

    train_dataset = BASE.URVDEAttentionDataset(paths["train_seq"], paths["train_aff"], args.max_seq_len, args.max_smi_len)
    valid_dataset = BASE.URVDEAttentionDataset(paths["valid_seq"], paths["valid_aff"], args.max_seq_len, args.max_smi_len)
    test_dataset = BASE.URVDEAttentionDataset(paths["test_seq"], paths["test_aff"], args.max_seq_len, args.max_smi_len)
    return paths, train_dataset, valid_dataset, test_dataset


def build_loaders(args, train_dataset, valid_dataset, test_dataset, shuffle_train):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, valid_loader, test_loader


def save_split_eval_outputs(split_result_dir, split_id, prefix, role, metrics, ids, y_true, y_pred):
    metrics_path = os.path.join(split_result_dir, "Metrics_{0}_{1}.csv".format(role, prefix))
    pred_path = os.path.join(split_result_dir, "Predictions_{0}_{1}.csv".format(role, prefix))
    scatter_path = os.path.join(split_result_dir, "Scatter_{0}_{1}.png".format(role, prefix))

    BASE.save_metrics(metrics_path, metrics, split_id, role + "_" + prefix)
    BASE.save_predictions(pred_path, ids, y_true, y_pred, split_id, role + "_" + prefix)
    BASE.save_scatter(scatter_path, y_true, y_pred,
                      "DEAttentionDTA pretrained {0} split {1} {2}".format(prefix, split_id, role))

    return {
        "metrics_csv": metrics_path,
        "predictions_csv": pred_path,
        "scatter_png": scatter_path,
    }


def run_debug_pretrained(args, repo_root):
    prepared_dir = ensure_prepared_dataset(args, repo_root)
    device = BASE.choose_device(args.device)
    model_cls, model_source_path = BASE.load_model_class(repo_root)
    paths, train_dataset, _, _ = build_datasets(args, prepared_dir, 1)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = model_cls().to(device)
    strict = not args.non_strict_pretrained
    load_report = load_pretrained_weights(
        model, join_root(repo_root, args.pretrained_path), device, strict,
        pretrained_fold=args.pretrained_fold, target_split=1
    )

    model.eval()
    first_batch = next(iter(loader))
    _, smi, seq, pocket, affinity = BASE.to_device(first_batch, device)
    with torch.no_grad():
        output = model(seq, smi, pocket).reshape(-1)

    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    report = {
        "script": SCRIPT_NAME,
        "mode": "debug-pretrained",
        "device": str(device),
        "model_source_path": model_source_path,
        "prepared_train_split_1_rows": int(len(train_dataset)),
        "input_paths": paths,
        "batch_size": int(args.batch_size),
        "output_shape": list(output.detach().cpu().size()),
        "target_shape": list(affinity.detach().cpu().size()),
        "output_has_nan": bool(torch.isnan(output.detach().cpu()).any().item()),
        "output_has_inf": bool(torch.isinf(output.detach().cpu()).any().item()),
        "load_pretrained_report": load_report,
    }
    report_path = os.path.join(results_dir, "debug_pretrained_report.json")
    write_json(report_path, report)
    print("Debug pretrained forward pass OK")
    print("  device:          {0}".format(device))
    print("  output_shape:    {0}".format(report["output_shape"]))
    print("  compatible_keys: {0}/{1}".format(load_report["compatible_keys"], load_report["checkpoint_keys"]))
    print("  report:          {0}".format(report_path))
    return report


def evaluate_pretrained_split(args, repo_root, split_id):
    prepared_dir = join_root(repo_root, args.prepared_dir)
    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    split_result_dir = ensure_dir(os.path.join(results_dir, "split_{0:02d}".format(split_id)))

    BASE.set_seed(args.seed + split_id)
    device = BASE.choose_device(args.device)
    model_cls, model_source_path = BASE.load_model_class(repo_root)
    paths, train_dataset, valid_dataset, test_dataset = build_datasets(args, prepared_dir, split_id)
    train_loader, valid_loader, test_loader = build_loaders(args, train_dataset, valid_dataset, test_dataset, shuffle_train=False)

    model = model_cls().to(device)
    strict = not args.non_strict_pretrained
    load_report = load_pretrained_weights(
        model, join_root(repo_root, args.pretrained_path), device, strict,
        pretrained_fold=args.pretrained_fold, target_split=split_id
    )
    criterion = nn.MSELoss()

    requested = getattr(args, "eval_split_mode", "test")
    if requested == "validation":
        requested = "valid"
    if requested not in ["train", "valid", "test", "all"]:
        raise ValueError("Evaluation split must be train, valid, test, or all: {0}".format(requested))

    role_loaders = {
        "train": (train_loader, int(len(train_dataset))),
        "valid": (valid_loader, int(len(valid_dataset))),
        "test": (test_loader, int(len(test_dataset))),
    }
    roles = ["train", "valid", "test"] if requested == "all" else [requested]

    metrics_by_role = {}
    row_counts = {"train": int(len(train_dataset)), "valid": int(len(valid_dataset)), "test": int(len(test_dataset))}
    summary = {
        "split": int(split_id),
        "mode": "zero-shot",
        "eval_split_mode": requested,
        "pretrained_path": join_root(repo_root, args.pretrained_path),
        "model_source_path": model_source_path,
        "row_counts": row_counts,
        "load_pretrained_report": load_report,
    }

    for role in roles:
        loader, _count = role_loaders[role]
        metrics, ids, y_true, y_pred = BASE.evaluate(model, loader, criterion, device)
        metrics_by_role[role] = metrics
        output_role = "validation" if role == "valid" else role
        save_split_eval_outputs(split_result_dir, split_id, "zero_shot", output_role, metrics, ids, y_true, y_pred)
        prefix = "val" if role == "valid" else role
        summary[prefix + "_RMSE"] = metrics["RMSE"]
        summary[prefix + "_MSE"] = metrics.get("MSE")
        summary[prefix + "_MAE"] = metrics["MAE"]
        summary[prefix + "_Pearson"] = metrics["Pearson"]
        summary[prefix + "_N"] = metrics.get("N")

    summary["metrics"] = metrics_by_role
    run_config = {
        "model_name": "DEAttentionDTA",
        "workflow": "evaluate",
        "prepared_dir": prepared_dir,
        "checkpoint_path": join_root(repo_root, args.pretrained_path),
        "results_dir": results_dir,
        "device": str(device),
        "fold_index": int(split_id),
        "split": requested,
        "batch_size": int(args.batch_size),
        "seed": int(args.seed + split_id),
    }
    write_json(os.path.join(split_result_dir, "run_config.json"), run_config)
    write_json(os.path.join(split_result_dir, "zero_shot_summary.json"), summary)
    primary_role = roles[-1]
    primary_metrics = metrics_by_role[primary_role]
    print("Zero-shot split {0} {1}: RMSE={2:.6f} Pearson={3:.6f}".format(
        split_id, primary_role, primary_metrics["RMSE"], primary_metrics["Pearson"]
    ))
    return summary


def finetune_split(args, repo_root, split_id):
    prepared_dir = join_root(repo_root, args.prepared_dir)
    models_dir = ensure_dir(join_root(repo_root, args.models_dir))
    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    split_model_dir = ensure_dir(os.path.join(models_dir, "split_{0:02d}".format(split_id)))
    split_result_dir = ensure_dir(os.path.join(results_dir, "split_{0:02d}".format(split_id)))

    BASE.set_seed(args.seed + split_id)
    device = BASE.choose_device(args.device)
    model_cls, model_source_path = BASE.load_model_class(repo_root)
    paths, train_dataset, valid_dataset, test_dataset = build_datasets(args, prepared_dir, split_id)
    train_loader, valid_loader, test_loader = build_loaders(args, train_dataset, valid_dataset, test_dataset, shuffle_train=True)

    model = model_cls().to(device)
    strict = not args.non_strict_pretrained
    load_report = load_pretrained_weights(
        model, join_root(repo_root, args.pretrained_path), device, strict,
        pretrained_fold=args.pretrained_fold, target_split=split_id
    )

    criterion = nn.MSELoss()

    # Evaluate before fine-tuning so the same run also documents the zero-shot baseline.
    before_val_metrics, before_val_ids, before_val_true, before_val_pred = BASE.evaluate(model, valid_loader, criterion, device)
    before_test_metrics, before_test_ids, before_test_true, before_test_pred = BASE.evaluate(model, test_loader, criterion, device)
    save_split_eval_outputs(split_result_dir, split_id, "before_finetune", "validation",
                            before_val_metrics, before_val_ids, before_val_true, before_val_pred)
    save_split_eval_outputs(split_result_dir, split_id, "before_finetune", "test",
                            before_test_metrics, before_test_ids, before_test_true, before_test_pred)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    rounds_without_improvement = 0
    history_rows = []

    print("Fine-tuning split {0}/5 from pretrained | train={1} valid={2} test={3} | device={4}".format(
        split_id, len(train_dataset), len(valid_dataset), len(test_dataset), device
    ))
    print("  pretrained: {0}".format(join_root(repo_root, args.pretrained_path)))
    print("  lr={0} epochs={1} early_stopping_rounds={2}".format(args.lr, args.epochs, args.early_stopping_rounds))
    print("  before_finetune test_RMSE={0:.6f} test_Pearson={1:.6f}".format(
        before_test_metrics["RMSE"], before_test_metrics["Pearson"]
    ))

    for epoch in range(1, args.epochs + 1):
        train_loss = BASE.train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, split_id)
        val_metrics, _, _, _ = BASE.evaluate(model, valid_loader, criterion, device)
        history_rows.append({
            "split": split_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_Pearson": val_metrics["Pearson"],
        })
        print("split {0:02d} epoch {1:03d} train_loss={2:.6f} val_loss={3:.6f} val_RMSE={4:.6f}".format(
            split_id, epoch, train_loss, val_metrics["loss"], val_metrics["RMSE"]
        ))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if args.early_stopping_rounds > 0 and rounds_without_improvement >= args.early_stopping_rounds:
                print("Early stopping split {0} at epoch {1}".format(split_id, epoch))
                break

    if best_state is None:
        raise RuntimeError("No best state was captured for split {0}".format(split_id))

    model.load_state_dict(best_state)
    after_val_metrics, after_val_ids, after_val_true, after_val_pred = BASE.evaluate(model, valid_loader, criterion, device)
    after_test_metrics, after_test_ids, after_test_true, after_test_pred = BASE.evaluate(model, test_loader, criterion, device)

    history_path = os.path.join(split_result_dir, "finetune_history.csv")
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    save_split_eval_outputs(split_result_dir, split_id, "after_finetune", "validation",
                            after_val_metrics, after_val_ids, after_val_true, after_val_pred)
    save_split_eval_outputs(split_result_dir, split_id, "after_finetune", "test",
                            after_test_metrics, after_test_ids, after_test_true, after_test_pred)

    model_path = os.path.join(split_model_dir, "DEAttentionDTA_URV_v3b_pretrained_finetuned_split{0:02d}.pt".format(split_id))
    bundle = {
        "format_version": "urv_v3b_pretrained_finetuned_v1",
        "model_name": MODEL_NAME,
        "script": SCRIPT_NAME,
        "split_id": int(split_id),
        "pretrained_path": join_root(repo_root, args.pretrained_path),
        "pretrained_load_report": load_report,
        "best_epoch": int(best_epoch),
        "state_dict": best_state,
        "model_source_path": model_source_path,
        "input_paths": paths,
        "max_seq_len": int(args.max_seq_len),
        "max_smi_len": int(args.max_smi_len),
        "vocab": {"SMI_CHAR": BASE.SMI_CHAR, "PROTEIN_CHAR": BASE.PROTEIN_CHAR},
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "seed": int(args.seed + split_id),
        },
        "metrics": {
            "before_finetune": {"validation": before_val_metrics, "test": before_test_metrics},
            "after_finetune": {"validation": after_val_metrics, "test": after_test_metrics},
        },
    }
    torch.save(bundle, model_path)

    summary = {
        "split": int(split_id),
        "mode": "finetune",
        "model_pt": model_path,
        "results_dir": split_result_dir,
        "pretrained_path": join_root(repo_root, args.pretrained_path),
        "best_epoch": int(best_epoch),
        "train_rows": int(len(train_dataset)),
        "validation_rows": int(len(valid_dataset)),
        "test_rows": int(len(test_dataset)),
        "before_test_RMSE": before_test_metrics["RMSE"],
        "before_test_MAE": before_test_metrics["MAE"],
        "before_test_Pearson": before_test_metrics["Pearson"],
        "val_RMSE": after_val_metrics["RMSE"],
        "val_MAE": after_val_metrics["MAE"],
        "val_Pearson": after_val_metrics["Pearson"],
        "test_RMSE": after_test_metrics["RMSE"],
        "test_MAE": after_test_metrics["MAE"],
        "test_Pearson": after_test_metrics["Pearson"],
    }
    write_json(os.path.join(split_result_dir, "finetune_summary.json"), summary)
    print("Finished fine-tune split {0}: before_RMSE={1:.6f} after_RMSE={2:.6f} model={3}".format(
        split_id, before_test_metrics["RMSE"], after_test_metrics["RMSE"], model_path
    ))
    return summary


def aggregate_summaries(summaries, results_dir, mode):
    summary_df = pd.DataFrame(summaries).sort_values("split")
    if mode == "zero-shot":
        summary_csv = os.path.join(results_dir, "Summary_5splits_zero_shot_pretrained.csv")
        aggregate_path = os.path.join(results_dir, "Aggregate_metrics_zero_shot_pretrained.json")
    else:
        summary_csv = os.path.join(results_dir, "Summary_5splits_pretrained_finetuned.csv")
        aggregate_path = os.path.join(results_dir, "Aggregate_metrics_pretrained_finetuned.json")

    summary_df.to_csv(summary_csv, index=False)

    aggregate = {"splits_completed": int(len(summary_df)), "mode": mode}
    if len(summary_df) > 0:
        metric_columns = [
            "train_RMSE", "train_MSE", "train_MAE", "train_Pearson", "train_N",
            "val_RMSE", "val_MSE", "val_MAE", "val_Pearson", "val_N",
            "test_RMSE", "test_MSE", "test_MAE", "test_Pearson", "test_N",
        ]
        if mode == "finetune":
            metric_columns.extend(["before_test_RMSE", "before_test_MAE", "before_test_Pearson"])
        for prefix in metric_columns:
            if prefix in summary_df.columns:
                aggregate[prefix + "_mean"] = float(summary_df[prefix].mean())
                aggregate[prefix + "_std"] = float(summary_df[prefix].std(ddof=1)) if len(summary_df) > 1 else float("nan")

    write_json(aggregate_path, aggregate)
    print("Completed {0}".format(mode))
    print("  summary_csv: {0}".format(summary_csv))
    print("  aggregate:   {0}".format(aggregate_path))
    return summary_csv, aggregate_path


def run_zero_shot(args, repo_root):
    ensure_prepared_dataset(args, repo_root)
    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    summaries = []
    for split_id in BASE.parse_splits_arg(args.splits):
        summaries.append(evaluate_pretrained_split(args, repo_root, split_id))
    return aggregate_summaries(summaries, results_dir, "zero-shot")


def run_finetune(args, repo_root):
    ensure_prepared_dataset(args, repo_root)
    results_dir = ensure_dir(join_root(repo_root, args.results_dir))
    summaries = []
    for split_id in BASE.parse_splits_arg(args.splits):
        summaries.append(finetune_split(args, repo_root, split_id))
    return aggregate_summaries(summaries, results_dir, "finetune")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune/evaluate pretrained DEAttentionDTA on the official URV 5 splits. Python 3.6 compatible."
    )
    parser.add_argument("--mode", choices=["debug-pretrained", "zero-shot", "finetune"], default="finetune")
    parser.add_argument("--prepared-dir", default="data/urv_dataset_v3b_prepared", help="Path to prepared DEAttentionDTA CSVs generated by Prepare_URV_Positions_From_V2_Dataset.py.")
    parser.add_argument("--pretrained-path", default="models/pretrained/DEAttentionDTA.pt", help="Path to pretrained DEAttentionDTA checkpoint weights. Supports simple state_dict checkpoints and aggregate checkpoints with fold_state_dicts.")
    parser.add_argument("--pretrained-fold", default="matching",
                        help="For aggregate checkpoints with fold_state_dicts: matching, first, or a fold id such as 1. Default: matching")
    parser.add_argument("--models-dir", default="models/finetuned", help="Output directory for fine-tuned .pt models, one subfolder per split.")
    parser.add_argument("--results-dir", default="outputs/pretrained_vs_finetuned", help="Output directory for zero-shot/fine-tuning metrics, predictions, plots and histories.")
    parser.add_argument("--splits", default="all", help="Comma-separated split ids, e.g. 1,2,3,4,5 or all")
    parser.add_argument("--device", choices=["auto", "cuda", "cuda:0", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=5e-5, help="Learning rate used by Adam during fine-tuning. Alias: --learning-rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=990721)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=BASE.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-smi-len", type=int, default=BASE.DEFAULT_MAX_SMI_LEN)
    parser.add_argument("--eval-split-mode", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--non-strict-pretrained", action="store_true",
                        help="Load checkpoint with strict=False. Use only if strict loading fails after verifying the report.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = BASE.repo_root_from_script()
    started = time.time()
    try:
        if args.mode == "debug-pretrained":
            run_debug_pretrained(args, repo_root)
        elif args.mode == "zero-shot":
            run_zero_shot(args, repo_root)
        else:
            run_finetune(args, repo_root)
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        traceback.print_exc()
        return 1
    print("Elapsed seconds: {0:.1f}".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
