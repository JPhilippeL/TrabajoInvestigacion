"""Quick validation and dry-run utility for the TFM CAPLA implementation.

This script validates the explicit CAPLA dataset structure, optionally materializes
an on-disk debug subset of at most ``max-debug-samples`` complexes, instantiates
the clean CAPLA model, and performs a short forward-pass sanity check.

Examples
--------
Run from the repository root with explicit dataset paths::

    python TFM_Implementation/CAPLA/Debug_Graphs.py \
        --affinity-csv CAPLA/CAPLA/data/affinity_data.csv \
        --smi-csv CAPLA/CAPLA/data/Test2016_290_smi.csv \
        --global-dir CAPLA/CAPLA/data/Test2016_290/global \
        --pocket-dir CAPLA/CAPLA/data/Test2016_290/pocket

Force CPU and reduce the debug sample size::

    python TFM_Implementation/CAPLA/Debug_Graphs.py \
        --affinity-csv CAPLA/CAPLA/data/affinity_data.csv \
        --smi-csv CAPLA/CAPLA/data/Test2016_290_smi.csv \
        --global-dir CAPLA/CAPLA/data/Test2016_290/global \
        --pocket-dir CAPLA/CAPLA/data/Test2016_290/pocket \
        --device cpu \
        --max-debug-samples 256 \
        --batch-size 4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make `python TFM_Implementation/CAPLA/Debug_Graphs.py` work from the repo root.
_THIS_FILE = Path(__file__).resolve()
_REPO_HINT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
if str(_REPO_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_HINT))

from CAPLA.core.common import (  # noqa: E402
    CAPLAImplementationError,
    CAPLAPathError,
    choose_device,
    ensure_dir,
    find_capla_repo_root,
    get_logger,
    resolve_path,
    save_json,
    seed_everything,
)
from CAPLA.core.data_utils import (  # noqa: E402
    CAPLADataError,
    CAPLADataset,
    PT_FEATURE_SIZE,
    build_dataset_index,
)
from CAPLA.core.model_adapter import build_capla_model  # noqa: E402

DEFAULT_MAX_SEQ_LEN = 1000
DEFAULT_MAX_PKT_LEN = 64
DEFAULT_MAX_SMI_LEN = 150


class DebugGraphsError(CAPLAImplementationError):
    """Raised when `Debug_Graphs.py` cannot complete its validation workflow."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a CAPLA dataset and run a short forward-pass sanity check."
    )
    parser.add_argument("--affinity-csv", required=True, help="Path to affinity_data.csv")
    parser.add_argument("--smi-csv", required=True, help="Path to the CAPLA *_smi.csv file")
    parser.add_argument("--global-dir", required=True, help="Directory with global/*.csv features")
    parser.add_argument("--pocket-dir", required=True, help="Directory with pocket/*.csv features")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device preference (default: auto)",
    )
    parser.add_argument(
        "--max-debug-samples",
        type=int,
        default=2000,
        help="Maximum number of complexes to copy into the debug subset (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for the dry-run forward pass (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for subset selection and deterministic behavior (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for reports and debug subset output. "
            "Defaults to <repo_root>/TFM_Implementation/CAPLA"
        ),
    )
    return parser.parse_args()


def _read_csv_columns(path: Path) -> List[str]:
    return [str(col).strip() for col in pd.read_csv(path, nrows=0).columns]


def _detect_usable_feature_columns(path: Path) -> List[str]:
    df = pd.read_csv(path, index_col=0)
    if "idx" in df.columns:
        df = df.drop(columns=["idx"])
    unnamed_cols = [col for col in df.columns if str(col).lower().startswith("unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return [str(col) for col in df.columns]


def _materialize_debug_subset(
    index_df: pd.DataFrame,
    output_dir: Path,
    max_samples: int,
    seed: int,
) -> Tuple[pd.DataFrame, Path, Path, Path, Path]:
    """Write an on-disk debug dataset using only the validated ids in `index_df`."""
    n_samples = min(max_samples, len(index_df))
    subset_df = (
        index_df.sample(n=n_samples, random_state=seed).sort_values("pdbid").reset_index(drop=True)
        if len(index_df) > n_samples
        else index_df.sort_values("pdbid").reset_index(drop=True)
    )

    debug_dir = ensure_dir(output_dir / "debug_samples")
    global_out = ensure_dir(debug_dir / "global")
    pocket_out = ensure_dir(debug_dir / "pocket")

    affinity_path = debug_dir / "affinity_data_debug.csv"
    smi_path = debug_dir / "dataset_debug_smi.csv"

    subset_df.loc[:, ["pdbid", "affinity"]].to_csv(affinity_path, index=False)
    subset_df.loc[:, ["pdbid", "smiles"]].to_csv(smi_path, index=False)

    for row in subset_df.itertuples(index=False):
        global_src = Path(str(row.global_path))
        pocket_src = Path(str(row.pocket_path))
        (global_out / global_src.name).write_bytes(global_src.read_bytes())
        (pocket_out / pocket_src.name).write_bytes(pocket_src.read_bytes())

    remapped_df = subset_df.copy()
    remapped_df["global_path"] = remapped_df["pdbid"].apply(lambda x: str(global_out / f"{x}.csv"))
    remapped_df["pocket_path"] = remapped_df["pdbid"].apply(lambda x: str(pocket_out / f"{x}.csv"))
    return remapped_df, debug_dir, affinity_path, smi_path, global_out, pocket_out


def _tensor_stats(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "has_nan": bool(torch.isnan(tensor).any().item()) if tensor.is_floating_point() else False,
        "has_inf": bool(torch.isinf(tensor).any().item()) if tensor.is_floating_point() else False,
        "min": float(tensor.min().item()) if tensor.numel() > 0 and tensor.is_floating_point() else None,
        "max": float(tensor.max().item()) if tensor.numel() > 0 and tensor.is_floating_point() else None,
    }


def _run_dry_forward_pass(
    dataset_df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    logger_name: str = "TFM_CAPLA_DEBUG",
) -> Dict[str, Any]:
    logger = get_logger(logger_name)
    dataset = CAPLADataset(
        index_df=dataset_df,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        max_pkt_len=DEFAULT_MAX_PKT_LEN,
        max_smi_len=DEFAULT_MAX_SMI_LEN,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_capla_model().to(device)
    model.eval()

    dry_run: Dict[str, Any] = {
        "forward_pass_ok": False,
        "checked_batches": 0,
        "model_class": model.__class__.__name__,
        "input_batches": [],
        "output_batches": [],
        "dataset_length": len(dataset),
    }

    max_batches = 2 if device.type == "cuda" else 1
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            pdbids, seq_tensor, pkt_tensor, smi_tensor, affinity = batch
            seq_tensor = seq_tensor.to(device)
            pkt_tensor = pkt_tensor.to(device)
            smi_tensor = smi_tensor.to(device)
            affinity = affinity.to(device)

            batch_info = {
                "batch_index": batch_index,
                "batch_size": len(pdbids),
                "pdbids": list(pdbids),
                "seq": _tensor_stats("seq", seq_tensor),
                "pkt": _tensor_stats("pkt", pkt_tensor),
                "smi": _tensor_stats("smi", smi_tensor.float()),
                "affinity": _tensor_stats("affinity", affinity),
            }

            output = model(seq_tensor, pkt_tensor, smi_tensor)
            output_stats = _tensor_stats("output", output)
            expected_batch = len(pdbids)
            valid_shape = output.ndim in {1, 2} and output.shape[0] == expected_batch
            output_has_nan = bool(torch.isnan(output).any().item()) if output.is_floating_point() else False
            output_has_inf = bool(torch.isinf(output).any().item()) if output.is_floating_point() else False
            batch_info["output"] = output_stats
            batch_info["output_shape_valid"] = valid_shape
            batch_info["output_has_nan"] = output_has_nan
            batch_info["output_has_inf"] = output_has_inf
            dry_run["input_batches"].append(batch_info)
            dry_run["output_batches"].append(output_stats)
            dry_run["checked_batches"] += 1

            if not valid_shape:
                raise DebugGraphsError(
                    f"Unexpected model output shape {tuple(output.shape)} for batch size {expected_batch}."
                )
            if output_has_nan or output_has_inf:
                raise DebugGraphsError("Model forward pass produced NaN or Inf values.")

    dry_run["forward_pass_ok"] = dry_run["checked_batches"] > 0
    logger.info("Dry-run completed on %d batch(es).", dry_run["checked_batches"])
    return dry_run


def _format_list_preview(values: Sequence[Any], max_items: int = 8) -> str:
    values = list(values)
    if not values:
        return "[]"
    if len(values) <= max_items:
        return json.dumps(values, ensure_ascii=False)
    preview = values[:max_items]
    return json.dumps(preview, ensure_ascii=False)[:-1] + f", ...] (total={len(values)})"


def _build_summary_text(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    dry_run = report.get("dry_run", {})
    lines = [
        "CAPLA Debug Report",
        "=" * 80,
        f"Repo root               : {report['repo_root']}",
        f"Original model file     : {report['original_model_file']}",
        f"Device                  : {report['device']['selected']}",
        f"CUDA available          : {report['device']['cuda_available']}",
        f"Total dataset size      : {validation['total_dataset_size']}",
        f"Debug subset size       : {validation['debug_subset_size']}",
        f"Affinity column         : {validation['affinity_column']}",
        f"Affinity columns        : {', '.join(validation['columns_detected']['affinity_csv'])}",
        f"SMI columns             : {', '.join(validation['columns_detected']['smi_csv'])}",
        f"Global feature dims     : {validation['feature_dimensions_detected']['global']}",
        f"Pocket feature dims     : {validation['feature_dimensions_detected']['pocket']}",
        f"Common ids count        : {validation['common_ids_count']}",
        f"Common complete ids     : {validation['common_complete_ids_count']}",
        f"Missing ids             : {validation['missing_ids_count']}",
        f"Duplicate affinity ids  : {len(validation['duplicates']['affinity'])}",
        f"Duplicate SMI ids       : {len(validation['duplicates']['smiles'])}",
        f"Null rows (affinity)    : {validation['null_rows']['affinity']}",
        f"Null rows (smiles)      : {validation['null_rows']['smiles']}",
        f"Unknown SMILES ids      : {validation['unknown_smiles_count']}",
        f"Forward pass ok         : {dry_run.get('forward_pass_ok', False)}",
        f"Checked batches         : {dry_run.get('checked_batches', 0)}",
        f"Critical errors         : {len(report['critical_errors'])}",
        "-" * 80,
        f"Missing global ids      : {_format_list_preview(validation['missing_ids']['global'])}",
        f"Missing pocket ids      : {_format_list_preview(validation['missing_ids']['pocket'])}",
        f"Orphan global ids       : {_format_list_preview(validation['orphan_ids']['global'])}",
        f"Orphan pocket ids       : {_format_list_preview(validation['orphan_ids']['pocket'])}",
        f"Unknown SMILES ids      : {_format_list_preview(list(validation['unknown_smiles'].keys()))}",
        f"Critical error details  : {_format_list_preview(report['critical_errors'], max_items=5)}",
    ]
    return "\n".join(lines) + "\n"


def _build_machine_report(
    *,
    repo_root: Path,
    original_model_file: Path,
    affinity_csv: Path,
    smi_csv: Path,
    global_dir: Path,
    pocket_dir: Path,
    report_obj: Any,
    index_df: pd.DataFrame,
    debug_subset_df: pd.DataFrame,
    debug_paths: Mapping[str, str],
    device: torch.device,
    dry_run: Mapping[str, Any],
    critical_errors: List[str],
) -> Dict[str, Any]:
    global_feature_columns: List[str] = []
    pocket_feature_columns: List[str] = []
    global_feature_dims: List[int] = []
    pocket_feature_dims: List[int] = []

    if not debug_subset_df.empty:
        first_global = Path(str(debug_subset_df.iloc[0]["global_path"]))
        first_pocket = Path(str(debug_subset_df.iloc[0]["pocket_path"]))
        global_feature_columns = _detect_usable_feature_columns(first_global)
        pocket_feature_columns = _detect_usable_feature_columns(first_pocket)
        global_feature_dims = [len(global_feature_columns)]
        pocket_feature_dims = [len(pocket_feature_columns)]

    missing_ids = sorted(
        set(report_obj.missing_global_ids)
        | set(report_obj.missing_pocket_ids)
        | set(report_obj.orphan_global_ids)
        | set(report_obj.orphan_pocket_ids)
    )

    machine_report: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "original_model_file": str(original_model_file),
        "inputs": {
            "affinity_csv": str(affinity_csv),
            "smi_csv": str(smi_csv),
            "global_dir": str(global_dir),
            "pocket_dir": str(pocket_dir),
        },
        "device": {
            "selected": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "requested": None,
        },
        "validation": {
            "total_dataset_size": int(report_obj.valid_ids_count),
            "debug_subset_size": int(len(debug_subset_df)),
            "affinity_column": report_obj.affinity_column,
            "columns_detected": {
                "affinity_csv": _read_csv_columns(affinity_csv),
                "smi_csv": _read_csv_columns(smi_csv),
                "global_feature_sample": global_feature_columns,
                "pocket_feature_sample": pocket_feature_columns,
            },
            "feature_dimensions_detected": {
                "global": global_feature_dims,
                "pocket": pocket_feature_dims,
                "expected": PT_FEATURE_SIZE,
            },
            "common_ids_count": int(report_obj.common_ids_count),
            "common_complete_ids_count": int(report_obj.valid_ids_count),
            "missing_ids_count": len(missing_ids),
            "missing_ids": {
                "global": report_obj.missing_global_ids,
                "pocket": report_obj.missing_pocket_ids,
            },
            "orphan_ids": {
                "global": report_obj.orphan_global_ids,
                "pocket": report_obj.orphan_pocket_ids,
            },
            "duplicates": {
                "affinity": report_obj.duplicated_affinity_ids,
                "smiles": report_obj.duplicated_smi_ids,
            },
            "null_rows": {
                "affinity": int(report_obj.affinity_null_rows),
                "smiles": int(report_obj.smi_null_rows),
            },
            "unknown_smiles_count": len(report_obj.unknown_smiles),
            "unknown_smiles": report_obj.unknown_smiles,
            "invalid_feature_files": {
                "global": report_obj.invalid_global_files,
                "pocket": report_obj.invalid_pocket_files,
            },
        },
        "debug_subset_paths": dict(debug_paths),
        "model": {
            "class_name": "CAPLA",
            "max_seq_len": DEFAULT_MAX_SEQ_LEN,
            "max_pkt_len": DEFAULT_MAX_PKT_LEN,
            "max_smi_len": DEFAULT_MAX_SMI_LEN,
        },
        "dry_run": dict(dry_run),
        "critical_errors": list(critical_errors),
        "status": "ok" if not critical_errors and dry_run.get("forward_pass_ok") else "failed",
    }
    return machine_report


def _collect_critical_errors(report_obj: Any, index_df: pd.DataFrame, dry_run_ok: bool) -> List[str]:
    errors: List[str] = []
    if report_obj.duplicated_affinity_ids:
        errors.append(f"Duplicate affinity ids detected: {len(report_obj.duplicated_affinity_ids)}")
    if report_obj.duplicated_smi_ids:
        errors.append(f"Duplicate SMILES ids detected: {len(report_obj.duplicated_smi_ids)}")
    if report_obj.affinity_null_rows:
        errors.append(f"Affinity null rows detected: {report_obj.affinity_null_rows}")
    if report_obj.smi_null_rows:
        errors.append(f"SMILES null rows detected: {report_obj.smi_null_rows}")
    if report_obj.missing_global_ids:
        errors.append(f"Missing global files for {len(report_obj.missing_global_ids)} ids")
    if report_obj.missing_pocket_ids:
        errors.append(f"Missing pocket files for {len(report_obj.missing_pocket_ids)} ids")
    if report_obj.unknown_smiles:
        errors.append(f"Unsupported SMILES characters found for {len(report_obj.unknown_smiles)} ids")
    if report_obj.invalid_global_files:
        errors.append(f"Invalid global feature files for {len(report_obj.invalid_global_files)} ids")
    if report_obj.invalid_pocket_files:
        errors.append(f"Invalid pocket feature files for {len(report_obj.invalid_pocket_files)} ids")
    if index_df.empty:
        errors.append("No valid ids remain after validation; dry-run cannot proceed")
    if not dry_run_ok:
        errors.append("Forward pass sanity check failed")
    return errors


def main() -> int:
    args = parse_args()
    logger = get_logger("TFM_CAPLA_DEBUG")
    seed_everything(int(args.seed))

    try:
        repo_root = find_capla_repo_root(_THIS_FILE)
        output_dir = ensure_dir(
            resolve_path(
                args.output_dir or (repo_root / "CAPLA" / "outputs" / "debug"),
                base_dir=repo_root,
                must_exist=False,
            )
        )
        reports_dir = ensure_dir(output_dir / "reports")
        affinity_csv = resolve_path(args.affinity_csv, base_dir=repo_root)
        smi_csv = resolve_path(args.smi_csv, base_dir=repo_root)
        global_dir = resolve_path(args.global_dir, base_dir=repo_root)
        pocket_dir = resolve_path(args.pocket_dir, base_dir=repo_root)
        original_model_file = resolve_path("CAPLA/original/src/capla.py", base_dir=repo_root)
        device = choose_device(args.device)

        logger.info("Repository root: %s", repo_root)
        logger.info("Using device: %s", device)
        logger.info("Validating dataset structure...")

        index_df, validation_report = build_dataset_index(
            affinity_csv=affinity_csv,
            smi_csv=smi_csv,
            global_dir=global_dir,
            pocket_dir=pocket_dir,
            strict=False,
        )

        debug_subset_df, debug_dir, dbg_affinity, dbg_smi, dbg_global, dbg_pocket = _materialize_debug_subset(
            index_df=index_df,
            output_dir=output_dir,
            max_samples=int(args.max_debug_samples),
            seed=int(args.seed),
        )

        dry_run: Dict[str, Any] = {"forward_pass_ok": False, "checked_batches": 0}
        if not debug_subset_df.empty:
            logger.info("Running dry forward pass on %d sample(s)...", len(debug_subset_df))
            dry_run = _run_dry_forward_pass(
                dataset_df=debug_subset_df,
                batch_size=int(args.batch_size),
                device=device,
            )
        else:
            logger.warning("No valid samples remain after validation; skipping dry-run.")

        critical_errors = _collect_critical_errors(
            report_obj=validation_report,
            index_df=index_df,
            dry_run_ok=bool(dry_run.get("forward_pass_ok", False)),
        )

        machine_report = _build_machine_report(
            repo_root=repo_root,
            original_model_file=original_model_file,
            affinity_csv=affinity_csv,
            smi_csv=smi_csv,
            global_dir=global_dir,
            pocket_dir=pocket_dir,
            report_obj=validation_report,
            index_df=index_df,
            debug_subset_df=debug_subset_df,
            debug_paths={
                "debug_dir": str(debug_dir),
                "affinity_csv": str(dbg_affinity),
                "smi_csv": str(dbg_smi),
                "global_dir": str(dbg_global),
                "pocket_dir": str(dbg_pocket),
            },
            device=device,
            dry_run=dry_run,
            critical_errors=critical_errors,
        )

        json_path = save_json(machine_report, reports_dir / "debug_report.json")
        txt_path = reports_dir / "debug_report.txt"
        txt_path.write_text(_build_summary_text(machine_report), encoding="utf-8")

        print(_build_summary_text(machine_report))
        logger.info("JSON report saved to %s", json_path)
        logger.info("Text report saved to %s", txt_path)

        return 1 if critical_errors else 0

    except (CAPLAPathError, CAPLADataError, DebugGraphsError, ValueError, OSError) as exc:
        logger.error("Debug routine failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
