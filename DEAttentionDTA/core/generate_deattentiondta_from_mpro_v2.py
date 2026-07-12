"""Generate DEAttentionDTA prepared CSVs from an MPro-URV v2-like raw root."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from .Run_URV_5Splits import (
        DEFAULT_MAX_SEQ_LEN,
        DEFAULT_MAX_SMI_LEN,
        M_PRO_BINDING_SITE_RESIDUES,
        PROTEIN_CHAR,
        SMI_CHAR,
        pocket_string,
    )
except ImportError:  # pragma: no cover - supports python -m core... from package root
    from Run_URV_5Splits import (  # type: ignore
        DEFAULT_MAX_SEQ_LEN,
        DEFAULT_MAX_SMI_LEN,
        M_PRO_BINDING_SITE_RESIDUES,
        PROTEIN_CHAR,
        SMI_CHAR,
        pocket_string,
    )

AA_CODES = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "HID": "H", "HIE": "H", "HIP": "H", "MSE": "M",
}

ID_RE = re.compile(r"[A-Za-z0-9]{4,}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_id(value: Any) -> str:
    text = str(value).strip().upper()
    text = text.replace("\xa0", "").replace(",00", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _pdb_prefix(stem: str) -> str:
    text = _norm_id(stem.split("_", 1)[0])
    match = ID_RE.match(text)
    return match.group(0) if match else text


def _candidate_aliases(path: Path) -> list[str]:
    stem = path.stem
    aliases = {_norm_id(stem), _pdb_prefix(stem)}
    for suffix in ("_LIGAND", "_PROTEIN"):
        up = stem.upper()
        if up.endswith(suffix):
            aliases.add(_norm_id(stem[: -len(suffix)]))
    parts = re.split(r"[_\-.\s]+", stem)
    if parts:
        aliases.add(_norm_id(parts[0]))
    return [item for item in aliases if item]


def _find_first(root: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_pic50(path: Path | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if path is None or not path.is_file():
        return records, ["missing pIC50.txt or equivalent affinity file"]
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [item for item in re.split(r"[\s,;\t]+", line) if item]
            if len(parts) < 2:
                warnings.append(f"pIC50 line {lineno} has fewer than two fields")
                continue
            pdb_id = _norm_id(parts[0])
            try:
                value = float(parts[1])
            except ValueError:
                if lineno == 1:
                    continue
                warnings.append(f"pIC50 line {lineno} has nonnumeric value: {parts[1]}")
                continue
            if not pdb_id:
                warnings.append(f"pIC50 line {lineno} has empty sample ID")
                continue
            records[pdb_id] = {"PDBname": pdb_id, "affinity": value, "source_line": lineno}
    return records, warnings


def _read_smiles(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() in {"smiles", "smile"} and len(parts) > 1:
            return parts[1], parts[2] if len(parts) > 2 else ""
        return parts[0], parts[1] if len(parts) > 1 else ""
    return "", ""


def _discover_smi(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[Path], list[str]]:
    files = sorted(root.rglob("*.smi"))
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    for path in files:
        smiles, ligand_id = _read_smiles(path)
        if not smiles:
            warnings.append(f"empty SMI file: {_safe_rel(path, root)}")
            continue
        if not ligand_id:
            parts = path.stem.split("_", 1)
            ligand_id = parts[1] if len(parts) > 1 else ""
        record = {
            "smiles": smiles,
            "ligand_id": ligand_id,
            "path": str(path),
            "relative_path": _safe_rel(path, root),
            "aliases": _candidate_aliases(path),
        }
        for alias in record["aliases"]:
            by_key[alias].append(record)
    return by_key, files, warnings


def _discover_files_by_alias(root: Path, suffixes: tuple[str, ...]) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name.lower().endswith(suffixes))
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        record = {"path": str(path), "relative_path": _safe_rel(path, root), "aliases": _candidate_aliases(path)}
        for alias in record["aliases"]:
            by_key[alias].append(record)
    return by_key, files


def _extract_pdb_sequence(path: Path) -> tuple[str, list[str]]:
    chains: dict[str, list[tuple[tuple[str, str, str], str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    warnings: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = AA_CODES.get(resname)
            if aa is None:
                warnings.append(f"unsupported_residue:{resname}")
                aa = "X"
            chains[chain].append((key, aa))
    if not chains:
        return "", ["no_atom_residues"]
    chain, residues = max(chains.items(), key=lambda item: len(item[1]))
    sequence = "".join(aa for _key, aa in sorted(residues, key=lambda item: _resseq_sort(item[0])))
    if chain == "_":
        chain = "blank"
    warnings = sorted(set(warnings))
    if len(chains) > 1:
        warnings.append(f"selected_chain:{chain}")
    return sequence, warnings


def _resseq_sort(key: tuple[str, str, str]) -> tuple[str, int, str, str]:
    chain, resseq, icode = key
    match = re.search(r"-?\d+", resseq)
    number = int(match.group(0)) if match else 0
    return chain, number, icode, resseq


def _parse_split_file(path: Path | None) -> tuple[list[list[str]], list[str]]:
    if path is None or not path.is_file():
        return [], ["missing split file"]
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    warnings: list[str] = []
    try:
        value = ast.literal_eval(text)
    except Exception:
        folds = []
        for line in text.splitlines():
            ids = [_norm_id(item) for item in re.split(r"[\s,;\t]+", line.strip()) if item]
            if ids:
                folds.append(ids)
        return folds, warnings
    if isinstance(value, list) and value and all(isinstance(item, (list, tuple, set)) for item in value):
        return [[_norm_id(item) for item in fold if _norm_id(item)] for fold in value], warnings
    if isinstance(value, list):
        return [[_norm_id(item) for item in value if _norm_id(item)]], warnings
    return [], ["unsupported split file format"]


def _length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / float(len(values))}


def _first_keys(mapping: dict[str, Any], n: int = 10) -> list[str]:
    return sorted(mapping.keys())[:n]


def _choose_unique(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(records, key=lambda item: item.get("relative_path", item.get("path", "")))[0]


def _build_audit(raw_root: Path, output_root: Path, folders: dict[str, Any], pic50: dict[str, Any], smi_by_key: dict[str, Any], sdf_by_key: dict[str, Any], pdb_by_key: dict[str, Any], smi_files: list[Path], sdf_files: list[Path], pdb_files: list[Path], candidates: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    skipped_counter = Counter(item["reason"] for item in skipped)
    pic50_ids = set(pic50)
    smi_ids = set(smi_by_key)
    pdb_ids = set(pdb_by_key)
    return {
        "timestamp": _now(),
        "raw_source_root": str(raw_root),
        "output_root": str(output_root),
        "folders_searched": folders,
        "pIC50_records_parsed": len(pic50),
        "SMI_files_found": len(smi_files),
        "SDF_files_found": len(sdf_files),
        "protein_PDB_files_found": len(pdb_files),
        "first_10_normalized_pIC50_IDs": _first_keys(pic50),
        "first_10_normalized_SMI_keys": _first_keys(smi_by_key),
        "first_10_normalized_SDF_keys": _first_keys(sdf_by_key),
        "first_10_normalized_PDB_keys": _first_keys(pdb_by_key),
        "complete_candidate_samples_detected": len(candidates),
        "unmatched_pIC50_examples": sorted(pic50_ids - (smi_ids & pdb_ids))[:10],
        "unmatched_SMI_examples": sorted(smi_ids - pic50_ids)[:10],
        "unmatched_protein_examples": sorted(pdb_ids - pic50_ids)[:10],
        "skipped_reasons": dict(skipped_counter),
        "skipped_examples": skipped[:25],
    }


def generate_deattentiondta_dataset_from_mpro_v2(raw_root: str | Path, output_root: str | Path, overwrite: bool = False, max_smiles_len: int | None = None, max_protein_len: int | None = None, strict: bool = True) -> dict[str, Any]:
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    reports_dir = output_root / "reports"
    audit_path = output_root / "generate_data_matching_audit.json"

    if not raw_root.is_dir():
        raise FileNotFoundError(f"missing raw folder: {raw_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output root already exists and is not empty: {output_root}")
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pic50_path = _find_first(raw_root, ["pIC50.txt", "pic50.txt", "affinity.txt", "affinity.csv"])
    split_paths = {
        "train": _find_first(raw_root, ["train_index_folder.txt", "train.txt"]),
        "valid": _find_first(raw_root, ["valid_index_folder.txt", "validation_index_folder.txt", "valid.txt", "validation.txt"]),
        "test": _find_first(raw_root, ["test_index_folder.txt", "test.txt"]),
    }
    folders = {
        "raw_root": str(raw_root),
        "pIC50": str(pic50_path) if pic50_path else None,
        "train_split": str(split_paths["train"]) if split_paths["train"] else None,
        "valid_split": str(split_paths["valid"]) if split_paths["valid"] else None,
        "test_split": str(split_paths["test"]) if split_paths["test"] else None,
        "smi_search": str(raw_root),
        "sdf_search": str(raw_root),
        "protein_pdb_search": str(raw_root),
    }

    pic50, warnings = _parse_pic50(pic50_path)
    smi_by_key, smi_files, smi_warnings = _discover_smi(raw_root)
    sdf_by_key, sdf_files = _discover_files_by_alias(raw_root, (".sdf",))
    pdb_by_key, pdb_files = _discover_files_by_alias(raw_root, (".pdb",))
    warnings.extend(smi_warnings)

    skipped: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    max_smi = int(max_smiles_len or DEFAULT_MAX_SMI_LEN)
    max_seq = int(max_protein_len or DEFAULT_MAX_SEQ_LEN)

    for pdb_id, affinity_record in sorted(pic50.items()):
        smi_record = _choose_unique(smi_by_key.get(pdb_id, []))
        pdb_record = _choose_unique(pdb_by_key.get(pdb_id, []))
        if smi_record is None:
            skipped.append({"PDBname": pdb_id, "reason": "missing_smiles_file"})
            continue
        if pdb_record is None:
            skipped.append({"PDBname": pdb_id, "reason": "missing_protein_pdb_file"})
            continue
        sequence, pdb_warnings = _extract_pdb_sequence(Path(pdb_record["path"]))
        smiles = str(smi_record["smiles"]).strip()
        reasons = []
        if not smiles:
            reasons.append("empty_smiles")
        if not sequence:
            reasons.append("empty_protein_sequence")
        smi_unknown = sorted(set(smiles) - set(SMI_CHAR.keys()))
        seq_unknown = sorted(set(sequence) - set(PROTEIN_CHAR.keys()))
        if smi_unknown:
            reasons.append("unsupported_smiles_tokens:" + "".join(smi_unknown))
        if seq_unknown:
            reasons.append("unsupported_protein_tokens:" + "".join(seq_unknown))
        if len(smiles) > max_smi:
            reasons.append(f"smiles_too_long:{len(smiles)}>{max_smi}")
        if len(sequence) > max_seq:
            reasons.append(f"protein_too_long:{len(sequence)}>{max_seq}")
        positions = [pos for pos in M_PRO_BINDING_SITE_RESIDUES if 1 <= pos <= len(sequence)]
        if not positions:
            reasons.append("empty_position_list")
        if reasons:
            for reason in reasons:
                skipped.append({"PDBname": pdb_id, "reason": reason, "smiles_path": smi_record.get("relative_path"), "protein_path": pdb_record.get("relative_path")})
            continue
        candidates.append({
            "PDBname": pdb_id,
            "ligand_id": smi_record.get("ligand_id", ""),
            "Smile": smiles,
            "Sequence": sequence,
            "Pocket": pocket_string(sequence, positions),
            "Position": str(list(positions)),
            "affinity": float(affinity_record["affinity"]),
            "smiles_source_path": smi_record.get("relative_path", smi_record.get("path", "")),
            "protein_source_path": pdb_record.get("relative_path", pdb_record.get("path", "")),
            "sdf_source_path": (_choose_unique(sdf_by_key.get(pdb_id, [])) or {}).get("relative_path", ""),
            "pdb_warnings": ";".join(pdb_warnings),
        })

    audit = _build_audit(raw_root, output_root, folders, pic50, smi_by_key, sdf_by_key, pdb_by_key, smi_files, sdf_files, pdb_files, candidates, skipped)
    _write_json(audit_path, audit)

    if not pic50:
        raise RuntimeError(f"missing pIC50 records: {pic50_path or raw_root}")
    if not smi_files:
        raise RuntimeError(f"no SMI files found under raw root: {raw_root}")
    if not pdb_files:
        raise RuntimeError(f"no protein PDB files found under raw root: {raw_root}")
    if not candidates:
        raise RuntimeError(f"no complete matched samples found; audit written to {audit_path}")
    strict_blocking = [item for item in skipped if str(item.get("reason", "")).startswith(("unsupported_smiles_tokens", "unsupported_protein_tokens", "smiles_too_long", "protein_too_long"))]
    if strict and strict_blocking:
        examples = ", ".join(f"{item.get('PDBname')}:{item.get('reason')}" for item in strict_blocking[:5])
        raise RuntimeError(f"unsupported token characters or length limits in {len(strict_blocking)} samples: {examples}; audit written to {audit_path}")

    split_data: dict[str, list[list[str]]] = {}
    split_warnings: list[str] = []
    for role, path in split_paths.items():
        folds, role_warnings = _parse_split_file(path)
        split_data[role] = folds
        split_warnings.extend(f"{role}: {warning}" for warning in role_warnings)
    warnings.extend(split_warnings)
    n_folds = min((len(folds) for folds in split_data.values() if folds), default=0)
    if n_folds < 1:
        raise RuntimeError("split mismatch: no usable train/valid/test folds were found")

    prepared_df = pd.DataFrame(candidates).drop_duplicates(subset=["PDBname"], keep="first")
    prepared_df = prepared_df.sort_values("PDBname").reset_index(drop=True)
    available_ids = set(prepared_df["PDBname"].tolist())

    output_root.mkdir(parents=True, exist_ok=True)
    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)

    seq_cols = ["PDBname", "Smile", "Sequence", "Pocket", "Position"]
    aff_cols = ["PDBname", "affinity"]
    prepared_df[seq_cols].to_csv(output_root / "seq_data_all.csv", index=False)
    prepared_df[aff_cols].to_csv(output_root / "affinity_all.csv", index=False)
    prepared_df.to_csv(output_root / "samples_manifest.csv", index=False)
    pd.DataFrame(skipped).to_csv(reports_dir / "dropped_rows.csv", index=False)

    manifest_rows: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    missing_split_ids: dict[str, list[str]] = {}
    overlap_checks: list[dict[str, Any]] = []

    for fold_idx in range(n_folds):
        fold_id = fold_idx + 1
        split_dir = split_root / f"split_{fold_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        role_sets = {role: set(split_data[role][fold_idx]) for role in ("train", "valid", "test")}
        overlaps = {
            "train_valid": sorted(role_sets["train"] & role_sets["valid"]),
            "train_test": sorted(role_sets["train"] & role_sets["test"]),
            "valid_test": sorted(role_sets["valid"] & role_sets["test"]),
        }
        overlap_checks.append({"fold": fold_id, **{key: len(value) for key, value in overlaps.items()}})
        if strict and any(overlaps.values()):
            raise RuntimeError(f"split mismatch: overlapping IDs in fold {fold_id}")
        split_record: dict[str, Any] = {"split": fold_id}
        for role in ("train", "valid", "test"):
            ids = role_sets[role]
            missing = sorted(ids - available_ids)
            missing_split_ids[f"split_{fold_id:02d}_{role}"] = missing
            role_df = prepared_df[prepared_df["PDBname"].isin(ids)].sort_values("PDBname").reset_index(drop=True)
            seq_path = split_dir / f"seq_{role}.csv"
            aff_path = split_dir / f"affinity_{role}.csv"
            role_df[seq_cols].to_csv(seq_path, index=False)
            role_df[aff_cols].to_csv(aff_path, index=False)
            manifest_rows.append({
                "split": fold_id,
                "role": role,
                "official_ids": len(ids),
                "exported_rows": len(role_df),
                "missing_from_generated": len(missing),
                "missing_ids": ",".join(missing),
                "seq_csv": str(seq_path),
                "affinity_csv": str(aff_path),
            })
            split_record[f"{role}_official"] = len(ids)
            split_record[f"{role}_exported"] = len(role_df)
            split_record[f"{role}_missing_ids"] = missing
        split_record.update({f"overlap_{key}": len(value) for key, value in overlaps.items()})
        split_reports.append(split_record)

    manifest_path = output_root / "split_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    missing_total = sum(len(v) for v in missing_split_ids.values())
    if strict and missing_total:
        _write_json(audit_path, {**audit, "missing_split_ids": missing_split_ids})
        raise RuntimeError(f"split mismatch: {missing_total} split IDs did not resolve to generated samples; audit written to {audit_path}")

    smiles_lengths = [len(item["Smile"]) for item in candidates]
    protein_lengths = [len(item["Sequence"]) for item in candidates]
    skipped_reasons = dict(Counter(item["reason"] for item in skipped))
    output_files = [
        str(output_root / "seq_data_all.csv"),
        str(output_root / "affinity_all.csv"),
        str(output_root / "samples_manifest.csv"),
        str(manifest_path),
        str(reports_dir / "dropped_rows.csv"),
        str(audit_path),
    ]
    output_files.extend(str(path) for path in sorted(split_root.rglob("*.csv")))
    split_sizes = {f"split_{row['split']:02d}_{row['role']}": int(row["exported_rows"]) for row in manifest_rows}

    report = {
        "raw_source_root": str(raw_root),
        "output_root": str(output_root),
        "timestamp": _now(),
        "detected_raw_structure": folders,
        "pIC50_entries_found": len(pic50),
        "smi_files_found": len(smi_files),
        "sdf_files_found": len(sdf_files),
        "protein_pdb_files_found": len(pdb_files),
        "candidate_samples_detected": len(candidates),
        "samples_kept": int(len(prepared_df)),
        "samples_skipped": int(len(skipped)),
        "skipped_reasons": skipped_reasons,
        "SMILES_length_statistics": _length_stats(smiles_lengths),
        "protein_length_statistics": _length_stats(protein_lengths),
        "split_sizes": split_sizes,
        "split_reports": split_reports,
        "missing_split_IDs": missing_split_ids,
        "overlap_checks": overlap_checks,
        "output_files_created": output_files,
        "warnings": warnings,
        "warnings_count": len(warnings),
        "exact_format_generated": "DEAttentionDTA URV prepared CSV format: seq_data_all.csv, affinity_all.csv, split_manifest.csv, splits/split_XX/seq_{train,valid,test}.csv, splits/split_XX/affinity_{train,valid,test}.csv",
        "max_smiles_len": max_smi,
        "max_protein_len": max_seq,
        "strict": bool(strict),
        "sdf_required": False,
        "sdf_requirement_note": "Current DEAttentionDTA loader consumes SMILES, protein sequence, pocket positions, and affinity only; SDF files are audited but not required.",
    }
    metadata = {
        "model_name": "DEAttentionDTA",
        "generated_by": "generate_deattentiondta_from_mpro_v2",
        "timestamp": report["timestamp"],
        "raw_source_root": str(raw_root),
        "output_root": str(output_root),
        "samples_kept": report["samples_kept"],
        "samples_skipped": report["samples_skipped"],
        "format": report["exact_format_generated"],
        "fold_index": "1..5",
        "split_mode": "train/valid/test",
    }
    _write_json(reports_dir / "generate_data_report.json", report)
    _write_json(output_root / "metadata.json", metadata)

    return {
        "status": "success",
        "operation": "generate_deattentiondta_dataset_from_mpro_v2",
        "summary": report,
        "artifacts": {
            "prepared_dataset": str(output_root),
            "metadata_json": str(output_root / "metadata.json"),
            "generate_data_report_json": str(reports_dir / "generate_data_report.json"),
            "matching_audit_json": str(audit_path),
            "seq_data_all_csv": str(output_root / "seq_data_all.csv"),
            "affinity_all_csv": str(output_root / "affinity_all.csv"),
            "split_manifest_csv": str(manifest_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DEAttentionDTA prepared data from an MPro-URV v2-like raw root.")
    parser.add_argument("--raw-root", required=True, help="Raw MPro-v2-like dataset root")
    parser.add_argument("--output-root", required=True, help="Prepared DEAttentionDTA output root")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output root")
    parser.add_argument("--max-smiles-len", type=int, default=None)
    parser.add_argument("--max-protein-len", type=int, default=None)
    parser.add_argument("--non-strict", action="store_true", help="Write reports even when split IDs are missing/overlapping")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = generate_deattentiondta_dataset_from_mpro_v2(
        raw_root=args.raw_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        max_smiles_len=args.max_smiles_len,
        max_protein_len=args.max_protein_len,
        strict=not args.non_strict,
    )
    summary = result["summary"]
    print("DEAttentionDTA Generate Data completed")
    print(f"  output:          {result['artifacts']['prepared_dataset']}")
    print(f"  samples kept:    {summary['samples_kept']}")
    print(f"  samples skipped: {summary['samples_skipped']}")
    print(f"  report:          {result['artifacts']['generate_data_report_json']}")
    print(f"  warnings:        {summary['warnings_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
