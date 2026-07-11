#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate CAPLA-ready data from raw MPro-URV_Version2-like files."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURE_NUMERIC_COLUMNS = [
    "non_polar", "polar", "acidic", "basic",
    "c2_1", "c2_2", "c2_3", "c2_4", "c2_5", "c2_6", "c2_7",
    "s2_B", "s2_C", "s2_E", "s2_G", "s2_H", "s2_I", "s2_S", "s2_T",
    "a_G", "a_A", "a_V", "a_L", "a_I", "a_M", "a_F", "a_P", "a_W",
    "a_S", "a_T", "a_Y", "a_C", "a_Q", "a_N", "a_D", "a_E", "a_K",
    "a_R", "a_H", "a_X",
]
SCHEMA_COLUMNS = FEATURE_NUMERIC_COLUMNS + ["idx"]
CSV_HEADER = [""] + SCHEMA_COLUMNS

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

AA_ORDER = ["G", "A", "V", "L", "I", "M", "F", "P", "W", "S", "T", "Y", "C", "Q", "N", "D", "E", "K", "R", "H", "X"]

C2_MAPPING = {
    "c2_1": ["A", "G", "V"],
    "c2_2": ["I", "L", "F", "P"],
    "c2_3": ["Y", "M", "T", "S"],
    "c2_4": ["H", "N", "Q", "W"],
    "c2_5": ["R", "K"],
    "c2_6": ["D", "E"],
    "c2_7": ["C"],
}
C2_BY_AA = {aa: group for group, aas in C2_MAPPING.items() for aa in aas}

PHYSICOCHEMICAL_MAPPING = {
    "non_polar": ["A", "F", "G", "I", "L", "M", "P", "V", "W"],
    "polar": ["C", "N", "Q", "S", "T", "Y"],
    "acidic": ["D", "E"],
    "basic": ["H", "K", "R"],
}
PHYS_BY_AA = {aa: group for group, aas in PHYSICOCHEMICAL_MAPPING.items() for aa in aas}

DSSP_COLUMNS = ["s2_B", "s2_C", "s2_E", "s2_G", "s2_H", "s2_I", "s2_S", "s2_T"]
DSSP_CODE_TO_COLUMN = {
    "B": "s2_B",
    "-": "s2_C",
    "C": "s2_C",
    " ": "s2_C",
    "E": "s2_E",
    "G": "s2_G",
    "H": "s2_H",
    "I": "s2_I",
    "S": "s2_S",
    "T": "s2_T",
}

FALLBACK_FEATURE_MODES = {"copy_existing", "symlink_existing", "validate_existing"}


class SecondaryStructureError(RuntimeError):
    """Raised when DSSP-backed secondary structure extraction cannot run."""


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


def _clean_id(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\xa0", "").replace(" ", "")
    text = text.replace(",00", "").replace(",", "")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})


def _split_filename_stem(stem: str) -> tuple[str, str, str]:
    raw_id = _clean_id(stem)
    parts = raw_id.split("_", 1)
    pdb_id = parts[0]
    ligand_id = parts[1] if len(parts) > 1 else ""
    return raw_id, pdb_id, ligand_id


def _find_required_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under raw root: {root}")
    exact = root / name
    return exact if exact in matches else matches[0]


def _read_smiles(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text:
            return re.split(r"\s+", text, maxsplit=1)[0]
    raise ValueError(f"SMILES file is empty: {path}")


def _parse_pic50_file(path: Path) -> dict[str, float]:
    entries: dict[str, float] = {}
    impossible_lines: list[tuple[int, str]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = [item.strip() for item in re.split(r"[\s,;\t]+", line) if item.strip()]
        numeric_positions: list[tuple[int, float]] = []
        for idx, token in enumerate(tokens):
            try:
                numeric_positions.append((idx, float(token)))
            except ValueError:
                continue
        if not numeric_positions:
            lowered = {token.lower() for token in tokens}
            if {"pic50", "pdbid", "pdb_id", "id", "complex", "sample"} & lowered:
                continue
            impossible_lines.append((line_no, raw_line))
            continue
        value_idx, value = numeric_positions[-1]
        id_token = next((token for idx, token in enumerate(tokens) if idx != value_idx and _clean_id(token)), "")
        if not id_token:
            impossible_lines.append((line_no, raw_line))
            continue
        entries[_clean_id(id_token)] = value
    if not entries:
        detail = "; ".join(f"line {line_no}: {text}" for line_no, text in impossible_lines[:5])
        raise ValueError(f"Could not parse any pIC50 entries from {path}. {detail}".strip())
    return entries


def _discover_ligand_files(root: Path, folder: str, suffix: str) -> dict[str, Path]:
    base = root / "Ligand" / folder
    if not base.is_dir():
        raise FileNotFoundError(f"Missing raw ligand directory: {base}")
    return {_clean_id(path.stem): path for path in sorted(base.glob(f"*{suffix}"))}


def _discover_protein_files(root: Path) -> dict[str, Path]:
    base = root / "Protein" / "Protein_PDB"
    if not base.is_dir():
        raise FileNotFoundError(f"Missing raw protein directory: {base}")
    proteins: dict[str, Path] = {}
    for path in sorted(base.glob("*_protein.pdb")):
        pdb_id = _clean_id(path.name[: -len("_protein.pdb")])
        proteins[pdb_id] = path
    return proteins


def _pdb_key_from_stem(stem: str, kind: str) -> str:
    cleaned = _clean_id(stem)
    for suffix in ("_protein", "_ligand"):
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    if "_" in cleaned:
        prefix, suffix = cleaned.split("_", 1)
        if kind in {"smi", "sdf"} and suffix in {"ligand", "protein"}:
            return prefix
        if kind in {"smi", "sdf"}:
            return prefix
        if kind == "pdb" and suffix == "protein":
            return prefix
    return cleaned


def _ligand_id_from_stem(stem: str, pdb_key: str) -> str:
    cleaned = _clean_id(stem)
    if "_" not in cleaned:
        return ""
    prefix, suffix = cleaned.split("_", 1)
    if prefix == pdb_key and suffix not in {"ligand", "protein"}:
        return suffix
    return ""


def _aliases_for_id(value: Any, kind: str = "generic") -> list[str]:
    cleaned = _clean_id(value)
    if not cleaned:
        return []
    aliases = [cleaned]
    pdb_key = _pdb_key_from_stem(cleaned, kind)
    aliases.append(pdb_key)
    if "_" in cleaned:
        aliases.append(cleaned.split("_", 1)[0])
    aliases.append(cleaned.replace("_ligand", "").replace("_protein", ""))
    out: list[str] = []
    for alias in aliases:
        if alias and alias not in out:
            out.append(alias)
    return out


def _file_records(paths: list[Path], kind: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        stem = _clean_id(path.stem)
        pdb_key = _pdb_key_from_stem(stem, kind)
        records.append(
            {
                "path": path,
                "stem": stem,
                "pdb_key": pdb_key,
                "ligand_id": _ligand_id_from_stem(stem, pdb_key),
                "aliases": _aliases_for_id(stem, kind) + _aliases_for_id(pdb_key, "generic"),
            }
        )
    return records


def _index_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for alias in record["aliases"]:
            index.setdefault(alias, [])
            if record not in index[alias]:
                index[alias].append(record)
    return index


def _choose_record(index: dict[str, list[dict[str, Any]]], aliases: list[str], preferred_pdb: str) -> dict[str, Any] | None:
    seen: list[dict[str, Any]] = []
    for alias in aliases:
        for record in index.get(alias, []):
            if record not in seen:
                seen.append(record)
    if not seen:
        return None
    exact = [record for record in seen if record["stem"] in aliases]
    if exact:
        return exact[0]
    same_pdb = [record for record in seen if record["pdb_key"] == preferred_pdb]
    if same_pdb:
        return same_pdb[0]
    return seen[0]


def _matching_audit(
    raw_root: Path,
    pic50_entries: dict[str, float],
    smi_records: list[dict[str, Any]],
    sdf_records: list[dict[str, Any]],
    pdb_records: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_pdbs = {sample["pdb_id"] for sample in samples}
    smi_pdbs = {record["pdb_key"] for record in smi_records}
    sdf_pdbs = {record["pdb_key"] for record in sdf_records}
    pdb_pdbs = {record["pdb_key"] for record in pdb_records}
    pic50_ids = set(pic50_entries)
    return {
        "raw_source_root": str(raw_root),
        "folders_searched": {
            "pIC50": str(raw_root / "pIC50.txt"),
            "smi": str(raw_root / "Ligand" / "Ligand_SMI"),
            "sdf": str(raw_root / "Ligand" / "Ligand_SDF"),
            "protein_pdb": str(raw_root / "Protein" / "Protein_PDB"),
            "splits": str(raw_root / "Splits"),
        },
        "pIC50_records_parsed": len(pic50_entries),
        "smi_files_found": len(smi_records),
        "sdf_files_found": len(sdf_records),
        "protein_pdb_files_found": len(pdb_records),
        "complete_samples_found": len(samples),
        "first_10_normalized_pIC50_ids": sorted(pic50_entries)[:10],
        "first_10_normalized_smi_keys": sorted(smi_pdbs)[:10],
        "first_10_normalized_sdf_keys": sorted(sdf_pdbs)[:10],
        "first_10_normalized_pdb_keys": sorted(pdb_pdbs)[:10],
        "first_10_smi_filenames": [record["path"].name for record in smi_records[:10]],
        "first_10_sdf_filenames": [record["path"].name for record in sdf_records[:10]],
        "first_10_pdb_filenames": [record["path"].name for record in pdb_records[:10]],
        "examples_unmatched_pIC50_ids": sorted(pic50_ids - sample_pdbs)[:10],
        "examples_unmatched_smi_ids": sorted(smi_pdbs - sample_pdbs)[:10],
        "examples_unmatched_sdf_ids": sorted(sdf_pdbs - sample_pdbs)[:10],
        "examples_unmatched_pdb_ids": sorted(pdb_pdbs - sample_pdbs)[:10],
        "skipped_examples": skipped[:10],
    }


def _build_candidate_samples(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    pic50_path = _find_required_file(raw_root, "pIC50.txt")
    pic50_entries = _parse_pic50_file(pic50_path)
    smi_dir = raw_root / "Ligand" / "Ligand_SMI"
    sdf_dir = raw_root / "Ligand" / "Ligand_SDF"
    pdb_dir = raw_root / "Protein" / "Protein_PDB"
    if not smi_dir.is_dir():
        raise FileNotFoundError(f"Missing raw ligand directory: {smi_dir}")
    if not sdf_dir.is_dir():
        raise FileNotFoundError(f"Missing raw ligand directory: {sdf_dir}")
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Missing raw protein directory: {pdb_dir}")
    smi_records = _file_records(list(smi_dir.glob("*.smi")), "smi")
    sdf_records = _file_records(list(sdf_dir.glob("*.sdf")), "sdf")
    pdb_records = _file_records(list(pdb_dir.glob("*.pdb")), "pdb")
    smi_index = _index_records(smi_records)
    sdf_index = _index_records(sdf_records)
    pdb_index = _index_records(pdb_records)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for pic50_id, pic50_value in sorted(pic50_entries.items()):
        aliases = _aliases_for_id(pic50_id, "generic")
        pdb_id = _pdb_key_from_stem(pic50_id, "generic")
        reasons: list[str] = []
        smi_record = _choose_record(smi_index, aliases, pdb_id)
        sdf_record = _choose_record(sdf_index, aliases, pdb_id)
        pdb_record = _choose_record(pdb_index, aliases, pdb_id)
        if smi_record is None:
            reasons.append("missing_smi")
        if sdf_record is None:
            reasons.append("missing_sdf")
        if pdb_record is None:
            reasons.append("missing_protein_pdb")
        if reasons:
            skipped.append({"raw_id": pic50_id, "pdb_id": pdb_id, "ligand_id": "", "reason": ";".join(reasons), "aliases": aliases})
            continue
        assert smi_record is not None and sdf_record is not None and pdb_record is not None
        ligand_id = smi_record.get("ligand_id") or sdf_record.get("ligand_id") or ""
        samples.append(
            {
                "raw_id": pic50_id,
                "pdbid": pic50_id,
                "pdb_id": pdb_id,
                "ligand_id": ligand_id,
                "pIC50": float(pic50_value),
                "smiles": _read_smiles(smi_record["path"]),
                "smi_path": str(smi_record["path"]),
                "sdf_path": str(sdf_record["path"]),
                "protein_pdb_path": str(pdb_record["path"]),
            }
        )
    stats = {
        "pIC50_entries_found": len(pic50_entries),
        "smi_files_found": len(smi_records),
        "sdf_files_found": len(sdf_records),
        "protein_pdb_files_found": len(pdb_records),
    }
    samples = sorted(samples, key=lambda row: row["pdbid"])
    audit = _matching_audit(raw_root, pic50_entries, smi_records, sdf_records, pdb_records, samples, skipped)
    return samples, skipped, stats, audit


def _parse_pdb_residues(pdb_path: Path) -> list[dict[str, Any]]:
    residues: "OrderedDict[tuple[str, int, str, str], dict[str, Any]]" = OrderedDict()
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        resname = line[17:20].strip().upper()
        if record == "HETATM" and resname != "MSE":
            continue
        aa = AA3_TO_1.get(resname)
        if aa is None and record != "ATOM":
            continue
        chain = line[21].strip() or "_"
        try:
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        icode = line[26].strip()
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", atom_name)[:1]).upper()
        if element == "H":
            continue
        key = (chain, resseq, icode, resname)
        if key not in residues:
            idx = f"{chain}{resseq}{icode}" if icode else f"{chain}{resseq}"
            residues[key] = {"key": key, "idx": idx, "aa": aa or "X", "atoms": []}
        residues[key]["atoms"].append((x, y, z))
    out = [row for row in residues.values() if row["atoms"]]
    if not out:
        raise ValueError(f"No protein residues with coordinates were parsed from {pdb_path}")
    return out


def _parse_sdf_coordinates(sdf_path: Path) -> list[tuple[float, float, float]]:
    lines = sdf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"SDF file is too short: {sdf_path}")
    atom_count = None
    counts_idx = None
    for idx, line in enumerate(lines[:20]):
        try:
            atom_count = int(line[0:3])
            int(line[3:6])
            counts_idx = idx
            break
        except Exception:
            continue
    if atom_count is None or counts_idx is None:
        raise ValueError(f"Could not parse SDF counts line: {sdf_path}")
    coords: list[tuple[float, float, float]] = []
    for line in lines[counts_idx + 1: counts_idx + 1 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            continue
        element = parts[3].upper()
        if element == "H":
            continue
        try:
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not coords:
        raise ValueError(f"No ligand heavy-atom coordinates were parsed from {sdf_path}")
    return coords


def _dssp_key_variants(chain: str, resseq: int, icode: str) -> list[tuple[str, tuple[str, int, str]]]:
    insertion = icode if icode else " "
    return [
        (chain, (" ", resseq, insertion)),
        (chain, ("H_MSE", resseq, insertion)),
        (chain, ("W", resseq, insertion)),
    ]


def _load_dssp_map(pdb_path: Path) -> tuple[dict[tuple[str, int, str], str], str]:
    try:
        from Bio.PDB import DSSP, PDBParser  # type: ignore
    except ImportError as exc:
        raise SecondaryStructureError(
            "Secondary structure extraction requires DSSP/mkdssp. Install mkdssp or use a documented fallback mode."
        ) from exc
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("capla_mpro", str(pdb_path))
        model = structure[0]
        dssp = DSSP(model, str(pdb_path))
    except Exception as exc:
        raise SecondaryStructureError(
            "Secondary structure extraction requires DSSP/mkdssp. Install mkdssp or use a documented fallback mode."
        ) from exc

    dssp_map: dict[tuple[str, int, str], str] = {}
    for key in dssp.keys():
        chain, residue_id = key
        _, resseq, icode = residue_id
        dssp_map[(str(chain), int(resseq), str(icode).strip())] = str(dssp[key][2] or "C")
    if not dssp_map:
        raise SecondaryStructureError(
            "Secondary structure extraction requires DSSP/mkdssp. Install mkdssp or use a documented fallback mode."
        )
    return dssp_map, "Bio.PDB.DSSP"


def _secondary_structure_map(pdb_path: Path, mode: str, residues: list[dict[str, Any]]) -> tuple[dict[tuple[str, int, str], str], str, list[str]]:
    if mode == "coil_fallback":
        return {(row["key"][0], row["key"][1], row["key"][2]): "C" for row in residues}, "coil_fallback_non_scientific", [
            "secondary_structure_mode=coil_fallback is non-original and non-scientific; all residues were marked as coil."
        ]
    if mode != "dssp":
        raise ValueError("secondary_structure_mode must be one of: dssp, coil_fallback")
    dssp_map, method = _load_dssp_map(pdb_path)
    return dssp_map, method, []


def _feature_row_for_residue(residue: dict[str, Any], ss_code: str) -> dict[str, Any]:
    aa = residue["aa"] if residue["aa"] in AA_ORDER else "X"
    row = {column: 0.0 for column in FEATURE_NUMERIC_COLUMNS}
    phys = PHYS_BY_AA.get(aa)
    if phys:
        row[phys] = 1.0
    c2 = C2_BY_AA.get(aa)
    if c2:
        row[c2] = 1.0
    ss_col = DSSP_CODE_TO_COLUMN.get(ss_code, "s2_C")
    row[ss_col] = 1.0
    row[f"a_{aa}"] = 1.0
    row["idx"] = residue["idx"]
    return row


def _write_feature_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for idx, row in enumerate(rows):
            writer.writerow([idx] + [row[column] for column in FEATURE_NUMERIC_COLUMNS] + [row["idx"]])


def _min_distance_to_ligand(residue: dict[str, Any], ligand_coords: list[tuple[float, float, float]]) -> float:
    min_sq = float("inf")
    for ax, ay, az in residue["atoms"]:
        for lx, ly, lz in ligand_coords:
            dx = ax - lx
            dy = ay - ly
            dz = az - lz
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < min_sq:
                min_sq = dist_sq
    return math.sqrt(min_sq)


def _generate_raw_features(
    samples: list[dict[str, Any]],
    output_root: Path,
    pocket_cutoff: float,
    secondary_structure_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], str]:
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    generated_files: list[str] = []
    warnings: list[str] = []
    secondary_structure_method = ""
    global_dir = output_root / "global"
    pocket_dir = output_root / "pocket"
    shutil.rmtree(global_dir, ignore_errors=True)
    shutil.rmtree(pocket_dir, ignore_errors=True)
    global_dir.mkdir(parents=True, exist_ok=True)
    pocket_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        sample_id = sample["pdbid"]
        try:
            residues = _parse_pdb_residues(Path(sample["protein_pdb_path"]))
            ligand_coords = _parse_sdf_coordinates(Path(sample["sdf_path"]))
            ss_map, method, ss_warnings = _secondary_structure_map(Path(sample["protein_pdb_path"]), secondary_structure_mode, residues)
            secondary_structure_method = secondary_structure_method or method
            warnings.extend([f"{sample_id}: {warning}" for warning in ss_warnings])

            feature_rows: list[dict[str, Any]] = []
            missing_ss = 0
            for residue in residues:
                chain, resseq, icode, _ = residue["key"]
                ss_code = ss_map.get((chain, resseq, icode), "C")
                if (chain, resseq, icode) not in ss_map:
                    missing_ss += 1
                feature_rows.append(_feature_row_for_residue(residue, ss_code))
            if missing_ss and secondary_structure_mode == "dssp":
                warnings.append(f"{sample_id}: DSSP assignments missing for {missing_ss} residues; those residues were marked s2_C.")

            pocket_rows = [
                row
                for residue, row in zip(residues, feature_rows)
                if _min_distance_to_ligand(residue, ligand_coords) <= float(pocket_cutoff)
            ]
            if not pocket_rows:
                raise ValueError(f"no pocket residues within {pocket_cutoff} A")

            global_path = global_dir / f"{sample_id}.csv"
            pocket_path = pocket_dir / f"{sample_id}.csv"
            _write_feature_csv(global_path, feature_rows)
            _write_feature_csv(pocket_path, pocket_rows)
            generated_files.extend([str(global_path), str(pocket_path)])
            kept.append(sample)
        except SecondaryStructureError:
            raise
        except Exception as exc:
            skipped.append({**sample, "reason": f"feature_generation_failed:{exc}"})
    return kept, skipped, warnings, generated_files, secondary_structure_method or secondary_structure_mode


def _prepare_existing_features(
    samples: list[dict[str, Any]],
    output_root: Path,
    feature_source_root: Path | None,
    feature_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    if feature_source_root is None:
        raise RuntimeError("Existing feature fallback modes require feature_source_root containing global/ and pocket/.")
    source_global = feature_source_root / "global"
    source_pocket = feature_source_root / "pocket"
    if not source_global.is_dir() or not source_pocket.is_dir():
        raise FileNotFoundError(f"Feature source root must contain global/ and pocket/: {feature_source_root}")

    link_mode = feature_mode in {"symlink_existing", "validate_existing"}
    global_dir = output_root / "global"
    pocket_dir = output_root / "pocket"
    shutil.rmtree(global_dir, ignore_errors=True)
    shutil.rmtree(pocket_dir, ignore_errors=True)
    global_dir.mkdir(parents=True, exist_ok=True)
    pocket_dir.mkdir(parents=True, exist_ok=True)

    maps = {
        "global": {_clean_id(path.stem): path for path in sorted(source_global.glob("*.csv"))},
        "pocket": {_clean_id(path.stem): path for path in sorted(source_pocket.glob("*.csv"))},
    }
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    generated_files: list[str] = []
    for sample in samples:
        candidates = [_clean_id(sample["pdbid"]), _clean_id(sample["raw_id"]), _clean_id(sample["pdb_id"])]
        found: dict[str, Path] = {}
        for kind in ("global", "pocket"):
            source = next((maps[kind][candidate] for candidate in candidates if candidate in maps[kind]), None)
            if source is not None:
                found[kind] = source
        if set(found) != {"global", "pocket"}:
            skipped.append({**sample, "reason": "missing_existing_feature_csv"})
            continue
        for kind, source in found.items():
            target = output_root / kind / f"{sample['pdbid']}.csv"
            if link_mode:
                target.symlink_to(source.resolve())
            else:
                shutil.copy2(source, target)
            generated_files.append(str(target))
        kept.append(sample)
    return kept, skipped, [], generated_files


def _parse_split_payload(text: str) -> list[list[str]]:
    stripped = text.strip()
    if not stripped:
        return [[]]
    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        if all(isinstance(item, (list, tuple, set)) for item in parsed):
            return [[_clean_id(value) for value in fold] for fold in parsed]
        return [[_clean_id(value) for value in parsed]]
    items: list[str] = []
    for line in stripped.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.extend(item for item in re.split(r"[\s,;\t]+", line) if item)
    return [[_clean_id(value) for value in items]]


def _write_table(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_splits(raw_root: Path, samples: list[dict[str, Any]], output_root: Path) -> tuple[dict[str, dict[str, int]], dict[str, list[str]], list[str], list[str]]:
    split_dir = raw_root / "Splits"
    files = {
        "train": split_dir / "train_index_folder.txt",
        "valid": split_dir / "valid_index_folder.txt",
        "test": split_dir / "test_index_folder.txt",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw split file: {path}")
    parsed = {role: _parse_split_payload(path.read_text(encoding="utf-8", errors="replace")) for role, path in files.items()}
    n_folds = max(len(value) for value in parsed.values())
    for role, folds in parsed.items():
        if len(folds) == 1 and n_folds > 1:
            parsed[role] = folds * n_folds
        elif len(folds) != n_folds:
            raise ValueError(f"Split role {role} has {len(folds)} folds, expected {n_folds}.")

    id_lookup: dict[str, list[str]] = {}
    by_id = {sample["pdbid"]: sample for sample in samples}
    for sample in samples:
        for value in (sample["pdbid"], sample["raw_id"], sample["pdb_id"], sample["ligand_id"]):
            cleaned = _clean_id(value)
            if cleaned:
                id_lookup.setdefault(cleaned, [])
                if sample["pdbid"] not in id_lookup[cleaned]:
                    id_lookup[cleaned].append(sample["pdbid"])

    split_sizes: dict[str, dict[str, int]] = {}
    missing_split_ids: dict[str, list[str]] = {}
    warnings: list[str] = []
    generated_files: list[str] = []
    split_root = output_root / "splits"
    shutil.rmtree(split_root, ignore_errors=True)

    for fold_idx in range(n_folds):
        split_name = f"split_{fold_idx + 1:02d}"
        out_dir = split_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved_by_role: dict[str, list[str]] = {}
        for role in ("train", "valid", "test"):
            resolved: list[str] = []
            missing: list[str] = []
            for raw_item in parsed[role][fold_idx]:
                item = _clean_id(raw_item)
                matches = id_lookup.get(item, [])
                if matches:
                    for match in matches:
                        if match not in resolved:
                            resolved.append(match)
                else:
                    missing.append(item)
            key = f"{split_name}/{role}"
            missing_split_ids[key] = missing
            if missing:
                warnings.append(f"{key} has {len(missing)} ids not present in generated samples.")
            resolved_by_role[role] = resolved

        overlaps = {
            "train_valid": sorted(set(resolved_by_role["train"]) & set(resolved_by_role["valid"])),
            "train_test": sorted(set(resolved_by_role["train"]) & set(resolved_by_role["test"])),
            "valid_test": sorted(set(resolved_by_role["valid"]) & set(resolved_by_role["test"])),
        }
        if any(overlaps.values()):
            raise ValueError(f"Split {split_name} has overlapping train/valid/test ids: {overlaps}")

        split_sizes[split_name] = {}
        for role, ids in resolved_by_role.items():
            rows = [
                {"pdbid": sample_id, "pic50": by_id[sample_id]["pIC50"], "smiles": by_id[sample_id]["smiles"]}
                for sample_id in ids
            ]
            out_path = out_dir / f"{role}.csv"
            _write_table(out_path, ["pdbid", "pic50", "smiles"], rows)
            split_sizes[split_name][role] = len(rows)
            generated_files.append(str(out_path))
    return split_sizes, missing_split_ids, warnings, generated_files


def _validate_feature_schema(feature_dir: Path) -> dict[str, Any]:
    files = sorted(feature_dir.glob("*.csv"))
    invalid: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for path in files:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != CSV_HEADER:
                    raise ValueError(f"unexpected header {reader.fieldnames}")
                count = 0
                for row in reader:
                    count += 1
                    for column in FEATURE_NUMERIC_COLUMNS:
                        float(row[column])
                    if not row["idx"]:
                        raise ValueError("blank idx")
                row_counts[path.stem] = count
        except Exception as exc:
            invalid[path.name] = str(exc)
    return {"files": len(files), "invalid_files": invalid, "row_counts": row_counts}


def _mode_name(feature_mode: str) -> str:
    if feature_mode == "generate":
        return "true_raw_feature_generation"
    return {
        "copy_existing": "copy_existing_features",
        "symlink_existing": "symlink_existing_features",
        "validate_existing": "validate_existing_features",
    }[feature_mode]


def _write_matching_audit(output_root: Path, audit: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "generate_data_matching_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


def _no_complete_samples_error(audit: dict[str, Any], audit_path: Path) -> ValueError:
    return ValueError(
        "No complete candidate samples were found after matching pIC50, SMI, SDF, and protein PDB files.\n"
        f"Matching audit: {audit_path}\n"
        f"pIC50 records parsed: {audit.get('pIC50_records_parsed')}\n"
        f"SMI files found: {audit.get('smi_files_found')}\n"
        f"SDF files found: {audit.get('sdf_files_found')}\n"
        f"Protein PDB files found: {audit.get('protein_pdb_files_found')}\n"
        f"First pIC50 IDs: {audit.get('first_10_normalized_pIC50_ids')}\n"
        f"First SMI keys: {audit.get('first_10_normalized_smi_keys')}\n"
        f"First SDF keys: {audit.get('first_10_normalized_sdf_keys')}\n"
        f"First PDB keys: {audit.get('first_10_normalized_pdb_keys')}\n"
        f"Unmatched pIC50 examples: {audit.get('examples_unmatched_pIC50_ids')}\n"
        f"Unmatched SMI examples: {audit.get('examples_unmatched_smi_ids')}\n"
        f"Unmatched SDF examples: {audit.get('examples_unmatched_sdf_ids')}\n"
        f"Unmatched PDB examples: {audit.get('examples_unmatched_pdb_ids')}\n"
        f"Folders searched: {audit.get('folders_searched')}"
    )


def generate_capla_dataset_from_mpro_v2(
    raw_root,
    output_root,
    overwrite=False,
    pocket_cutoff=4.5,
    secondary_structure_mode="dssp",
    feature_mode="generate",
    feature_source_root=None,
) -> dict:
    raw_root = _resolve_path(raw_root)
    output_root = _resolve_path(output_root)
    feature_source_root = _resolve_path(feature_source_root)
    assert raw_root is not None and output_root is not None

    feature_mode_aliases = {"copy": "copy_existing", "symlink": "symlink_existing", "validate": "validate_existing", "auto": "copy_existing"}
    feature_mode = feature_mode_aliases.get(str(feature_mode), str(feature_mode))
    if feature_mode not in {"generate"} | FALLBACK_FEATURE_MODES:
        raise ValueError("feature_mode must be one of: generate, copy_existing, symlink_existing, validate_existing")
    if float(pocket_cutoff) <= 0:
        raise ValueError("pocket_cutoff must be positive.")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw MPro-v2-like root does not exist: {raw_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output root already exists and is not empty: {output_root}. Use overwrite=True.")
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_samples, skipped, raw_stats, matching_audit = _build_candidate_samples(raw_root)
    audit_path = _write_matching_audit(output_root, matching_audit)
    if not candidate_samples:
        raise _no_complete_samples_error(matching_audit, audit_path)

    warnings: list[str] = []
    generated_files: list[str] = []
    secondary_structure_method = None
    if feature_mode == "generate":
        kept_samples, feature_skipped, feature_warnings, feature_files, secondary_structure_method = _generate_raw_features(
            candidate_samples,
            output_root,
            float(pocket_cutoff),
            str(secondary_structure_mode),
        )
    else:
        kept_samples, feature_skipped, feature_warnings, feature_files = _prepare_existing_features(
            candidate_samples,
            output_root,
            feature_source_root,
            feature_mode,
        )
        secondary_structure_method = "not_performed_existing_features"
    skipped.extend(feature_skipped)
    warnings.extend(feature_warnings)
    generated_files.extend(feature_files)
    kept_samples = sorted(kept_samples, key=lambda row: row["pdbid"])
    if not kept_samples:
        raise ValueError("No samples remained after feature generation/validation. See skipped reasons.")

    affinity_path = output_root / "affinity_data.csv"
    smi_path = output_root / "urv_v3b_smi.csv"
    sample_table_path = output_root / "mpro_v2_samples.csv"
    _write_table(affinity_path, ["pdbid", "pic50"], [{"pdbid": row["pdbid"], "pic50": row["pIC50"]} for row in kept_samples])
    _write_table(smi_path, ["pdbid", "smiles"], [{"pdbid": row["pdbid"], "smiles": row["smiles"]} for row in kept_samples])
    _write_table(
        sample_table_path,
        ["raw_id", "pdb_id", "ligand_id", "pIC50", "smiles", "smi_path", "sdf_path", "protein_pdb_path"],
        kept_samples,
    )
    generated_files.extend([str(affinity_path), str(smi_path), str(sample_table_path)])

    split_sizes, missing_split_ids, split_warnings, split_files = _load_splits(raw_root, kept_samples, output_root)
    warnings.extend(split_warnings)
    generated_files.extend(split_files)

    schema_validation = {
        "global": _validate_feature_schema(output_root / "global"),
        "pocket": _validate_feature_schema(output_root / "pocket"),
    }
    if schema_validation["global"]["invalid_files"] or schema_validation["pocket"]["invalid_files"]:
        raise ValueError(f"Generated feature schema validation failed: {schema_validation}")

    metadata_path = output_root / "metadata.json"
    report_path = output_root / "generate_data_report.json"
    generated_files.extend([str(audit_path), str(metadata_path), str(report_path)])
    skipped_reasons = Counter(item["reason"] for item in skipped)
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {
        "raw_source_root": str(raw_root),
        "output_root": str(output_root),
        "timestamp": timestamp,
        "mode": _mode_name(feature_mode),
        "raw_feature_extraction_performed": feature_mode == "generate",
        "feature_source_root": str(feature_source_root) if feature_source_root else None,
        "feature_mode": feature_mode,
        "pocket_cutoff": float(pocket_cutoff),
        "secondary_structure_mode": str(secondary_structure_mode),
        "secondary_structure_method": secondary_structure_method,
        "c2_mapping": C2_MAPPING,
        "physicochemical_mapping": PHYSICOCHEMICAL_MAPPING,
        "schema_columns": SCHEMA_COLUMNS,
        **raw_stats,
        "samples_kept": len(kept_samples),
        "samples_skipped": len(skipped),
        "skipped_reasons": dict(skipped_reasons),
        "split_sizes": split_sizes,
        "missing_split_ids": missing_split_ids,
        "generated_files": generated_files,
        "warnings": warnings,
        "skipped_samples": skipped,
        "matching_audit_json": str(audit_path),
        "matching_audit": matching_audit,
        "feature_schema_validation": schema_validation,
    }
    metadata = {key: report[key] for key in report if key not in {"skipped_samples", "feature_schema_validation", "matching_audit"}}
    metadata["dataset"] = "MPro-URV_Version2 prepared for CAPLA"
    metadata["affinity_csv"] = str(affinity_path)
    metadata["smi_csv"] = str(smi_path)
    metadata["global_dir"] = str(output_root / "global")
    metadata["pocket_dir"] = str(output_root / "pocket")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["metadata_json"] = str(metadata_path)
    report["report_json"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a CAPLA prepared dataset from raw MPro-URV_Version2.")
    parser.add_argument("--raw-root", required=True, help="Raw MPro-v2-like root")
    parser.add_argument("--output-root", required=True, help="Prepared CAPLA dataset output root")
    parser.add_argument("--overwrite", action="store_true", help="Replace output-root if it already exists")
    parser.add_argument("--pocket-cutoff", type=float, default=4.5, help="Protein-ligand residue cutoff in Angstrom")
    parser.add_argument("--secondary-structure-mode", choices=["dssp", "coil_fallback"], default="dssp")
    parser.add_argument("--feature-mode", choices=["generate", "copy_existing", "symlink_existing", "validate_existing"], default="generate")
    parser.add_argument("--feature-source-root", default=None, help="Existing CAPLA feature root for fallback modes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_capla_dataset_from_mpro_v2(
        raw_root=args.raw_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        pocket_cutoff=args.pocket_cutoff,
        secondary_structure_mode=args.secondary_structure_mode,
        feature_mode=args.feature_mode,
        feature_source_root=args.feature_source_root,
    )
    print("CAPLA MPro-URV_Version2 dataset generated")
    print("  output root :", report["output_root"])
    print("  samples kept:", report["samples_kept"])
    print("  mode        :", report["mode"])
    print("  report      :", report["report_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
