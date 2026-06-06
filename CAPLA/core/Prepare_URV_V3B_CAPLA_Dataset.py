#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare URV v3b for CAPLA using the official 5 splits.

Python 3.6 compatible.

This script keeps URV v3b as the official source for PDB IDs, SMILES, pIC50 and
predefined splits, while reusing the existing CAPLA feature files already present
in TFM_Implementation/CAPLA/urv_dataset/global and /pocket.

Default output:
  TFM_Implementation/CAPLA/urv_dataset_v3b_prepared/
"""

import argparse
import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = _THIS_FILE.parent
_REPO_HINT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) >= 3 else _THIS_FILE.parent
import sys
if str(_REPO_HINT) not in sys.path:
    sys.path.insert(0, str(_REPO_HINT))

from CAPLA.core.data_utils import build_dataset_index  # noqa: E402


def clean_pdb_id(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\xa0", "").replace(" ", "")
    text = text.replace(",00", "").replace(",", "")
    return "".join(ch for ch in text if ch.isalnum())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value, base=SCRIPT_DIR):
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def read_split_file(path: Path) -> List[List[str]]:
    data = ast.literal_eval(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 5:
        raise ValueError(f"Expected 5 split lists in {path}")
    return [[clean_pdb_id(item) for item in split] for split in data]


def copy_or_link_feature_dir(source_dir: Path, target_dir: Path, mode: str) -> Dict[str, Any]:
    if target_dir.exists():
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        elif target_dir.is_dir():
            shutil.rmtree(target_dir)
    ensure_dir(target_dir.parent)

    if mode == "symlink":
        target_dir.symlink_to(source_dir, target_is_directory=True)
        return {"mode": "symlink", "source": str(source_dir), "target": str(target_dir)}

    ensure_dir(target_dir)
    copied = 0
    for csv_path in sorted(source_dir.glob("*.csv")):
        shutil.copy2(csv_path, target_dir / csv_path.name)
        copied += 1
    return {"mode": "copy", "source": str(source_dir), "target": str(target_dir), "files_copied": copied}


def write_split_csvs(prepared_df: pd.DataFrame, splits: Dict[str, List[List[str]]], split_root: Path) -> List[Dict[str, Any]]:
    report_rows = []
    available = set(prepared_df["pdbid"])
    by_id = prepared_df.set_index("pdbid", drop=False)
    for split_idx in range(5):
        split_id = split_idx + 1
        split_dir = ensure_dir(split_root / f"split_{split_id:02d}")
        for role in ["train", "valid", "test"]:
            official_ids = splits[role][split_idx]
            missing = [pid for pid in official_ids if pid not in available]
            rows = []
            for pid in official_ids:
                if pid in by_id.index:
                    rows.append(by_id.loc[pid][["pdbid", "pic50", "smiles"]].to_dict())
            out_df = pd.DataFrame(rows, columns=["pdbid", "pic50", "smiles"])
            out_path = split_dir / f"{role}.csv"
            out_df.to_csv(out_path, index=False)
            report_rows.append({
                "split": split_id,
                "role": role,
                "official_ids": len(official_ids),
                "exported_rows": len(out_df),
                "missing_ids": missing,
                "csv": str(out_path),
            })
    return report_rows


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    v3b_dir = resolve_path(args.urv_v3b_dir)
    source_dataset_dir = resolve_path(args.source_dataset_dir)
    out_dir = resolve_path(args.out_dir)
    reports_dir = ensure_dir(out_dir / "reports")
    split_root = ensure_dir(out_dir / "splits")

    info_csv = v3b_dir / "Info.csv"
    split_dir = v3b_dir / "Splits"
    if not info_csv.is_file():
        raise FileNotFoundError(f"Missing URV v3b Info.csv: {info_csv}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing URV v3b Splits directory: {split_dir}")

    source_global = source_dataset_dir / "global"
    source_pocket = source_dataset_dir / "pocket"
    if not source_global.is_dir() or not source_pocket.is_dir():
        raise FileNotFoundError("Expected source CAPLA dataset to contain global/ and pocket/ directories")

    info = pd.read_csv(info_csv, sep=";", dtype=str, keep_default_na=False)
    required = ["PDB_ID", "SMILES", "pIC50"]
    missing_cols = [col for col in required if col not in info.columns]
    if missing_cols:
        raise ValueError(f"Info.csv is missing required columns: {missing_cols}")

    prepared_df = pd.DataFrame({
        "pdbid": info["PDB_ID"].map(clean_pdb_id),
        "pic50": pd.to_numeric(info["pIC50"], errors="coerce"),
        "smiles": info["SMILES"].astype(str).str.strip(),
    })
    prepared_df = prepared_df.dropna(subset=["pdbid", "pic50", "smiles"])
    prepared_df = prepared_df.drop_duplicates(subset=["pdbid"], keep="first").sort_values("pdbid").reset_index(drop=True)

    affinity_path = out_dir / "affinity_data.csv"
    smi_path = out_dir / "urv_v3b_smi.csv"
    ensure_dir(out_dir)
    prepared_df[["pdbid", "pic50"]].to_csv(affinity_path, index=False)
    prepared_df[["pdbid", "smiles"]].to_csv(smi_path, index=False)

    global_report = copy_or_link_feature_dir(source_global, out_dir / "global", args.feature_mode)
    pocket_report = copy_or_link_feature_dir(source_pocket, out_dir / "pocket", args.feature_mode)

    splits = {
        "train": read_split_file(split_dir / "train_index_folder.txt"),
        "valid": read_split_file(split_dir / "valid_index_folder.txt"),
        "test": read_split_file(split_dir / "test_index_folder.txt"),
    }
    split_report = write_split_csvs(prepared_df, splits, split_root)
    split_manifest = pd.DataFrame(split_report)
    # Save missing_ids as comma-separated for CSV readability.
    split_manifest_csv = split_manifest.copy()
    split_manifest_csv["missing_ids"] = split_manifest_csv["missing_ids"].map(lambda ids: ",".join(ids))
    split_manifest_path = out_dir / "split_manifest.csv"
    split_manifest_csv.to_csv(split_manifest_path, index=False)

    index_df, validation_report = build_dataset_index(
        affinity_path,
        smi_path,
        out_dir / "global",
        out_dir / "pocket",
        strict=False,
    )

    report = {
        "dataset": "URV v3b prepared for CAPLA",
        "python_target": "3.6",
        "urv_v3b_dir": str(v3b_dir),
        "source_dataset_dir": str(source_dataset_dir),
        "out_dir": str(out_dir),
        "affinity_csv": str(affinity_path),
        "smi_csv": str(smi_path),
        "global_dir": str(out_dir / "global"),
        "pocket_dir": str(out_dir / "pocket"),
        "n_info_rows": int(len(info)),
        "n_prepared_rows": int(len(prepared_df)),
        "n_valid_capla_rows": int(len(index_df)),
        "feature_global": global_report,
        "feature_pocket": pocket_report,
        "validation_report": validation_report.to_dict(),
        "split_manifest_csv": str(split_manifest_path),
        "split_report": split_report,
    }
    report_path = reports_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("CAPLA URV v3b dataset prepared")
    print("  Info rows          :", len(info))
    print("  Prepared rows      :", len(prepared_df))
    print("  Valid CAPLA rows   :", len(index_df))
    print("  Output directory   :", out_dir)
    print("  Report             :", report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare URV v3b official-split dataset for CAPLA.")
    parser.add_argument("--urv-v3b-dir", default="urv_dataset_v3b", help="Directory containing Info.csv and Splits/")
    parser.add_argument("--source-dataset-dir", default="urv_dataset", help="Existing CAPLA dataset with affinity/smi/global/pocket")
    parser.add_argument("--out-dir", default="urv_dataset_v3b_prepared", help="Prepared output directory")
    parser.add_argument("--feature-mode", choices=["copy", "symlink"], default="copy", help="Copy or symlink global/pocket feature directories")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
