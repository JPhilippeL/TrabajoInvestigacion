"""
@file generate_dcml_from_mpro_v2.py
@brief Generate DCML prepared matrices from a raw MPro-v2-like dataset.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

DISTANCE_FEATURES = 63360
CHARGE_FEATURES = 55000
FULL_FEATURES = DISTANCE_FEATURES + CHARGE_FEATURES
DEFAULT_MAX_LIGAND_ATOMS = 36
DEFAULT_MAX_PROTEIN_ATOMS = 1760
WATER_RESNAMES = {"HOH", "WAT", "H2O"}
SUPPORTED_VARIANTS = {"distance_only", "real_charge", "full"}
PROTEIN_CHARGE_METHODS = {"pdb2pqr"}
LIGAND_CHARGE_METHODS = {"rdkit_gasteiger", "openbabel"}


class DCMLGenerateDataError(RuntimeError):
    """Raised when raw DCML data generation cannot complete honestly."""


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    pdb_id: str
    ligand_id: str
    label: float
    sdf_path: Path
    pdb_path: Path
    smi_path: Path | None
    smiles: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value).strip()).lower()


def _pdb_prefix(stem: str) -> str:
    return str(stem).split("_", 1)[0]


def _aliases(path: Path, suffixes: tuple[str, ...]) -> set[str]:
    stem = path.stem
    values = {stem, stem.lower(), _pdb_prefix(stem), _pdb_prefix(stem).lower()}
    lowered = stem.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            values.add(stem[: -len(suffix)])
            values.add(lowered[: -len(suffix)])
    return {_norm_id(value) for value in values if _norm_id(value)}


def _first_by_alias(files: Iterable[Path], suffixes: tuple[str, ...]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    mapping: dict[str, Path] = {}
    collisions: dict[str, list[str]] = defaultdict(list)
    for path in sorted(files):
        for key in _aliases(path, suffixes):
            if key in mapping and mapping[key] != path:
                collisions[key].append(str(path))
                continue
            mapping[key] = path
    return mapping, dict(collisions)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_pic50(raw_root: Path) -> tuple[list[dict[str, Any]], Path]:
    candidates = sorted(raw_root.rglob("pIC50.txt")) + sorted(raw_root.rglob("*pic50*.csv"))
    if not candidates:
        raise DCMLGenerateDataError(f"missing pIC50: no pIC50.txt or *pic50*.csv found under {raw_root}")
    path = candidates[0]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(_read_text(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in re.split(r"[\s,;\t]+", stripped) if part.strip()]
        if len(parts) < 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        sample_id = parts[0]
        rows.append(
            {
                "line": lineno,
                "sample_id": sample_id,
                "normalized_id": _norm_id(sample_id),
                "pIC50": value,
            }
        )
    if not rows:
        raise DCMLGenerateDataError(f"missing pIC50: {path} contained no parseable sample/value records")
    return rows, path


def _find_files(raw_root: Path, preferred: str, patterns: tuple[str, ...]) -> list[Path]:
    roots = [raw_root / preferred] if (raw_root / preferred).is_dir() else []
    roots.append(raw_root)
    seen: set[Path] = set()
    found: list[Path] = []
    for root in roots:
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    found.append(path)
    return found


def _parse_smi(path: Path | None) -> str:
    if path is None:
        return ""
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.split()[0]
    return ""


def _ligand_id_from_smi(path: Path | None) -> str:
    if path is None:
        return ""
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return ""


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_audit(output_root: Path, audit: Mapping[str, Any]) -> Path:
    return _write_json(output_root / "generate_data_matching_audit.json", audit)


def _discover_samples(raw_root: Path, output_root: Path) -> tuple[list[RawSample], dict[str, Any]]:
    pic50_rows, pic50_path = _parse_pic50(raw_root)
    sdf_files = _find_files(raw_root, "Ligand/Ligand_SDF", ("*.sdf", "*.SDF"))
    smi_files = _find_files(raw_root, "Ligand/Ligand_SMI", ("*.smi", "*.SMI"))
    pdb_files = _find_files(raw_root, "Protein/Protein_PDB", ("*.pdb", "*.PDB"))
    sdf_by_key, sdf_collisions = _first_by_alias(sdf_files, ("_ligand",))
    smi_by_key, smi_collisions = _first_by_alias(smi_files, ())
    pdb_by_key, pdb_collisions = _first_by_alias(pdb_files, ("_protein",))

    skipped: Counter[str] = Counter()
    unmatched_pic50: list[dict[str, Any]] = []
    samples: list[RawSample] = []
    for row in pic50_rows:
        key = row["normalized_id"]
        sdf = sdf_by_key.get(key)
        pdb = pdb_by_key.get(key)
        smi = smi_by_key.get(key)
        missing = []
        if sdf is None:
            missing.append("missing_ligand_sdf")
        if pdb is None:
            missing.append("missing_protein_pdb")
        if missing:
            reason = "+".join(missing)
            skipped[reason] += 1
            unmatched_pic50.append({"sample_id": row["sample_id"], "normalized_id": key, "reason": reason})
            continue
        samples.append(
            RawSample(
                sample_id=str(row["sample_id"]).upper(),
                pdb_id=str(row["sample_id"]).upper(),
                ligand_id=_ligand_id_from_smi(smi),
                label=float(row["pIC50"]),
                sdf_path=sdf,
                pdb_path=pdb,
                smi_path=smi,
                smiles=_parse_smi(smi),
            )
        )

    sample_keys = {_norm_id(sample.sample_id) for sample in samples}
    audit = {
        "raw_source_root": str(raw_root),
        "folders_searched": {
            "pIC50": str(pic50_path),
            "sdf": str(raw_root / "Ligand/Ligand_SDF"),
            "smi": str(raw_root / "Ligand/Ligand_SMI"),
            "pdb": str(raw_root / "Protein/Protein_PDB"),
        },
        "pIC50_records_parsed": len(pic50_rows),
        "SMI_files_found": len(smi_files),
        "SDF_files_found": len(sdf_files),
        "protein_PDB_files_found": len(pdb_files),
        "first_10_normalized_pIC50_IDs": [row["normalized_id"] for row in pic50_rows[:10]],
        "first_10_normalized_SMI_keys": sorted(smi_by_key)[:10],
        "first_10_normalized_SDF_keys": sorted(sdf_by_key)[:10],
        "first_10_normalized_PDB_keys": sorted(pdb_by_key)[:10],
        "complete_candidate_samples_detected": len(samples),
        "unmatched_pIC50_examples": unmatched_pic50[:20],
        "unmatched_SDF_examples": [
            {"path": str(path), "keys": sorted(_aliases(path, ("_ligand",)))[:5]}
            for path in sdf_files
            if not (_aliases(path, ("_ligand",)) & sample_keys)
        ][:20],
        "unmatched_protein_examples": [
            {"path": str(path), "keys": sorted(_aliases(path, ("_protein",)))[:5]}
            for path in pdb_files
            if not (_aliases(path, ("_protein",)) & sample_keys)
        ][:20],
        "skipped_reasons": dict(skipped),
        "alias_collisions": {"sdf": sdf_collisions, "smi": smi_collisions, "pdb": pdb_collisions},
    }
    _write_audit(output_root, audit)
    if not sdf_files:
        raise DCMLGenerateDataError("no SDF files: no ligand SDF files were found under the raw dataset root")
    if not pdb_files:
        raise DCMLGenerateDataError("no protein PDB files: no protein PDB files were found under the raw dataset root")
    if not samples:
        raise DCMLGenerateDataError(
            "no complete matched samples: pIC50 records did not match both ligand SDF and protein PDB files. "
            f"See {output_root / 'generate_data_matching_audit.json'}"
        )
    return samples, audit


@dataclass(frozen=True)
class ProteinAtom:
    x: float
    y: float
    z: float
    element: str
    atom_name: str
    resname: str
    chain: str
    resseq: str
    icode: str
    occurrence: int
    charge: float | None = None

    @property
    def key(self) -> tuple[str, str, str, str, str, str, int]:
        return (
            self.atom_name.strip().upper(),
            self.resname.strip().upper(),
            self.chain.strip(),
            self.resseq.strip(),
            self.icode.strip(),
            self.element.strip().upper(),
            self.occurrence,
        )


@dataclass(frozen=True)
class LigandAtom:
    x: float
    y: float
    z: float
    element: str
    atom_index: int
    charge: float | None = None


def _parse_sdf_coords(path: Path) -> list[LigandAtom]:
    lines = _read_text(path).splitlines()
    if len(lines) < 4:
        raise DCMLGenerateDataError(f"invalid SDF: {path}")
    try:
        natoms = int(lines[3][0:3])
    except ValueError as exc:
        raise DCMLGenerateDataError(f"invalid SDF counts line in {path}") from exc
    coords: list[LigandAtom] = []
    for atom_index, line in enumerate(lines[4 : 4 + natoms], start=1):
        parts = line.split()
        if len(parts) < 4:
            continue
        element = parts[3].upper()
        if element == "H":
            continue
        try:
            coords.append(LigandAtom(float(parts[0]), float(parts[1]), float(parts[2]), element, atom_index))
        except ValueError:
            continue
    if not coords:
        raise DCMLGenerateDataError(f"ligand SDF has no parseable heavy-atom coordinates: {path}")
    return coords


def _parse_pdb_coords(path: Path) -> list[ProteinAtom]:
    coords: list[ProteinAtom] = []
    occurrences: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for line in _read_text(path).splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        if resname in WATER_RESNAMES:
            continue
        atom_name = line[12:16].strip()
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        icode = line[26:27].strip()
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if element == "H":
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        base_key = (atom_name.upper(), resname.upper(), chain, resseq, icode, element.upper())
        occurrences[base_key] += 1
        coords.append(
            ProteinAtom(
                x=x,
                y=y,
                z=z,
                element=element,
                atom_name=atom_name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                icode=icode,
                occurrence=occurrences[base_key],
            )
        )
    if not coords:
        raise DCMLGenerateDataError(f"protein file has no parseable heavy-atom coordinates: {path}")
    return coords


def _parse_pqr_coords(path: Path) -> list[ProteinAtom]:
    coords: list[ProteinAtom] = []
    occurrences: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for line in _read_text(path).splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        resname = line[17:20].strip().upper() or parts[3].upper()
        if resname in WATER_RESNAMES:
            continue
        atom_name = line[12:16].strip() or parts[2]
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        icode = line[26:27].strip()
        if not chain and len(parts) >= 11:
            chain = parts[4]
            resseq = parts[5]
        elif not resseq and len(parts) >= 10:
            resseq = parts[4]
        element = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", atom_name)[:1]).upper()
        if element == "H":
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            try:
                x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
            except (ValueError, IndexError):
                continue
        try:
            charge = float(parts[-2])
        except (ValueError, IndexError):
            raise DCMLGenerateDataError(f"PQR atom line has no parseable charge in {path}")
        base_key = (atom_name.upper(), resname.upper(), chain, resseq, icode, element.upper())
        occurrences[base_key] += 1
        coords.append(
            ProteinAtom(
                x=x,
                y=y,
                z=z,
                element=element,
                atom_name=atom_name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                icode=icode,
                occurrence=occurrences[base_key],
                charge=charge,
            )
        )
    if not coords:
        raise DCMLGenerateDataError(f"PQR file has no parseable heavy-atom charges: {path}")
    return coords


def _distance(a: LigandAtom, b: ProteinAtom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _resolve_executable(explicit_path: str | None, candidates: tuple[str, ...], error_message: str) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(str(explicit_path))
        if found:
            return found
        raise DCMLGenerateDataError(error_message)
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    raise DCMLGenerateDataError(error_message)


def _dependency_status(
    *,
    variant: str,
    pdb2pqr_executable: str | None,
    ligand_charge_method: str,
    openbabel_executable: str | None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "pdb2pqr": shutil.which(pdb2pqr_executable or "pdb2pqr")
        or shutil.which("pdb2pqr30"),
        "obabel": shutil.which(openbabel_executable or "obabel"),
        "rdkit": False,
    }
    try:
        import rdkit  # type: ignore  # noqa: F401

        status["rdkit"] = True
    except Exception:
        status["rdkit"] = False
    status["missing"] = []
    if variant in {"real_charge", "full"}:
        if not status["pdb2pqr"]:
            status["missing"].append("pdb2pqr")
        if ligand_charge_method == "rdkit_gasteiger" and not status["rdkit"]:
            status["missing"].append("rdkit")
        if ligand_charge_method == "openbabel" and not status["obabel"]:
            status["missing"].append("obabel")
    return status


def _run_pdb2pqr(input_pdb: Path, output_pqr: Path, executable: str, forcefield: str) -> None:
    output_pqr.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [executable, "--ff", forcefield, str(input_pdb), str(output_pqr)],
        [executable, f"--ff={forcefield}", str(input_pdb), str(output_pqr)],
    ]
    last_error = ""
    for cmd in commands:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and output_pqr.is_file() and output_pqr.stat().st_size > 0:
            return
        last_error = f"command={' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    raise DCMLGenerateDataError(f"PDB2PQR failed for {input_pdb.name}: {last_error}")


def _protein_charges_for_sample(
    sample: RawSample,
    *,
    output_root: Path,
    pdb2pqr_executable: str,
    protein_charge_forcefield: str,
) -> tuple[list[ProteinAtom], dict[str, Any]]:
    protein_atoms = _parse_pdb_coords(sample.pdb_path)
    pqr_path = output_root / "intermediate" / "protein_pqr" / f"{sample.sample_id}.pqr"
    if not pqr_path.is_file():
        _run_pdb2pqr(sample.pdb_path, pqr_path, pdb2pqr_executable, protein_charge_forcefield)
    pqr_atoms = _parse_pqr_coords(pqr_path)
    charges_by_key = {atom.key: atom.charge for atom in pqr_atoms}
    charged_atoms: list[ProteinAtom] = []
    missing = []
    for atom in protein_atoms:
        charge = charges_by_key.get(atom.key)
        if charge is None:
            missing.append(atom.key)
            continue
        charged_atoms.append(
            ProteinAtom(
                x=atom.x,
                y=atom.y,
                z=atom.z,
                element=atom.element,
                atom_name=atom.atom_name,
                resname=atom.resname,
                chain=atom.chain,
                resseq=atom.resseq,
                icode=atom.icode,
                occurrence=atom.occurrence,
                charge=float(charge),
            )
        )
    if missing:
        if len(pqr_atoms) >= len(protein_atoms) and all(
            pqr_atoms[idx].element == protein_atoms[idx].element for idx in range(len(protein_atoms))
        ):
            charged_atoms = [
                ProteinAtom(
                    x=atom.x,
                    y=atom.y,
                    z=atom.z,
                    element=atom.element,
                    atom_name=atom.atom_name,
                    resname=atom.resname,
                    chain=atom.chain,
                    resseq=atom.resseq,
                    icode=atom.icode,
                    occurrence=atom.occurrence,
                    charge=float(pqr_atoms[idx].charge),
                )
                for idx, atom in enumerate(protein_atoms)
            ]
        else:
            raise DCMLGenerateDataError(
                f"missing PQR charges for {len(missing)} protein heavy atoms in {sample.sample_id}; "
                f"first missing key: {missing[0]}"
            )
    audit_path = output_root / "intermediate" / "charge_audit" / f"{sample.sample_id}_protein_charge.json"
    _write_json(
        audit_path,
        {
            "sample_id": sample.sample_id,
            "source_pdb": str(sample.pdb_path),
            "generated_pqr": str(pqr_path),
            "protein_atoms": len(protein_atoms),
            "pqr_heavy_atoms": len(pqr_atoms),
            "charges_matched": len(charged_atoms),
            "sum_abs_charge": sum(abs(atom.charge or 0.0) for atom in charged_atoms),
        },
    )
    return charged_atoms, {"protein_pqr_path": str(pqr_path), "protein_charge_audit": str(audit_path)}


def _ligand_charges_rdkit(sample: RawSample, output_root: Path) -> tuple[list[LigandAtom], dict[str, Any]]:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception as exc:
        raise DCMLGenerateDataError("real_charge/full generation requires RDKit or OpenBabel for ligand partial charges.") from exc

    mol = Chem.MolFromMolFile(str(sample.sdf_path), removeHs=False, sanitize=True)
    if mol is None:
        raise DCMLGenerateDataError(f"RDKit could not parse ligand SDF: {sample.sdf_path}")
    AllChem.ComputeGasteigerCharges(mol)
    conformer = mol.GetConformer()
    atoms: list[LigandAtom] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol().upper() == "H":
            continue
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception as exc:
            raise DCMLGenerateDataError(f"missing RDKit Gasteiger charge for ligand atom {idx + 1} in {sample.sample_id}") from exc
        if math.isnan(charge) or math.isinf(charge):
            raise DCMLGenerateDataError(f"invalid RDKit Gasteiger charge for ligand atom {idx + 1} in {sample.sample_id}")
        atoms.append(LigandAtom(pos.x, pos.y, pos.z, atom.GetSymbol().upper(), idx + 1, charge))
    return _write_ligand_charge_audit(sample, output_root, atoms, "rdkit_gasteiger", "")


def _ligand_charges_openbabel(sample: RawSample, output_root: Path, executable: str) -> tuple[list[LigandAtom], dict[str, Any]]:
    mol2_path = output_root / "intermediate" / "ligand_charges" / f"{sample.sample_id}.mol2"
    mol2_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [executable, str(sample.sdf_path), "-O", str(mol2_path), "--partialcharge", "gasteiger"]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not mol2_path.is_file() or mol2_path.stat().st_size == 0:
        raise DCMLGenerateDataError(
            f"OpenBabel ligand charge generation failed for {sample.sample_id}: "
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    atoms: list[LigandAtom] = []
    in_atoms = False
    for line in _read_text(mol2_path).splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        atom_type = parts[5].split(".", 1)[0].upper()
        element = re.sub(r"[^A-Za-z]", "", atom_type)[:1].upper()
        if element == "H":
            continue
        try:
            atoms.append(
                LigandAtom(
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    element=element,
                    atom_index=int(parts[0]),
                    charge=float(parts[-1]),
                )
            )
        except ValueError:
            continue
    if not atoms:
        raise DCMLGenerateDataError(f"OpenBabel produced no parseable heavy-atom charges for {sample.sample_id}")
    return _write_ligand_charge_audit(sample, output_root, atoms, "openbabel_gasteiger", str(mol2_path))


def _write_ligand_charge_audit(
    sample: RawSample,
    output_root: Path,
    atoms: list[LigandAtom],
    method: str,
    generated_file: str,
) -> tuple[list[LigandAtom], dict[str, Any]]:
    if any(atom.charge is None for atom in atoms):
        raise DCMLGenerateDataError(f"missing ligand charges for real atoms in {sample.sample_id}")
    csv_path = output_root / "intermediate" / "ligand_charges" / f"{sample.sample_id}_charges.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["atom_index", "element", "x", "y", "z", "charge"])
        writer.writeheader()
        for atom in atoms:
            writer.writerow(
                {
                    "atom_index": atom.atom_index,
                    "element": atom.element,
                    "x": atom.x,
                    "y": atom.y,
                    "z": atom.z,
                    "charge": atom.charge,
                }
            )
    audit_path = output_root / "intermediate" / "charge_audit" / f"{sample.sample_id}_ligand_charge.json"
    _write_json(
        audit_path,
        {
            "sample_id": sample.sample_id,
            "source_sdf": str(sample.sdf_path),
            "method": method,
            "generated_file": generated_file,
            "charge_csv": str(csv_path),
            "ligand_atoms": len(atoms),
            "nonzero_charges": sum(1 for atom in atoms if abs(atom.charge or 0.0) > 0.0),
            "sum_abs_charge": sum(abs(atom.charge or 0.0) for atom in atoms),
        },
    )
    return atoms, {"ligand_charge_csv": str(csv_path), "ligand_charge_audit": str(audit_path)}


def _ligand_charges_for_sample(
    sample: RawSample,
    *,
    output_root: Path,
    ligand_charge_method: str,
    openbabel_executable: str | None,
) -> tuple[list[LigandAtom], dict[str, Any]]:
    if ligand_charge_method == "rdkit_gasteiger":
        return _ligand_charges_rdkit(sample, output_root)
    if ligand_charge_method == "openbabel":
        executable = _resolve_executable(
            openbabel_executable,
            ("obabel",),
            "real_charge/full generation requires RDKit or OpenBabel for ligand partial charges.",
        )
        return _ligand_charges_openbabel(sample, output_root, executable)
    raise DCMLGenerateDataError(
        f"unsupported ligand charge method: {ligand_charge_method}. Supported: {', '.join(sorted(LIGAND_CHARGE_METHODS))}"
    )


def _distance_features(
    sample: RawSample,
    *,
    max_ligand_atoms: int,
    max_protein_atoms: int,
    distance_cutoff: float | None,
) -> tuple[list[float], dict[str, Any]]:
    lig = _parse_sdf_coords(sample.sdf_path)
    prot = _parse_pdb_coords(sample.pdb_path)
    if distance_cutoff is not None:
        prot = [atom for atom in prot if any(_distance(latom, atom) <= distance_cutoff for latom in lig)]
        if not prot:
            raise DCMLGenerateDataError(f"distance cutoff removed all protein atoms for {sample.sample_id}")

    lig_used = lig[:max_ligand_atoms]
    prot_used = prot[:max_protein_atoms]
    values = [0.0] * (max_ligand_atoms * max_protein_atoms)
    for li, latom in enumerate(lig_used):
        row_offset = li * max_protein_atoms
        for pi, patom in enumerate(prot_used):
            values[row_offset + pi] = _distance(latom, patom)
    stats = {
        "ligand_atoms": len(lig),
        "protein_atoms": len(prot),
        "ligand_atoms_used": len(lig_used),
        "protein_atoms_used": len(prot_used),
        "ligand_atoms_truncated": max(0, len(lig) - len(lig_used)),
        "protein_atoms_truncated": max(0, len(prot) - len(prot_used)),
        "padding_values": (max_ligand_atoms - len(lig_used)) * max_protein_atoms
        + max(0, max_protein_atoms - len(prot_used)) * len(lig_used),
    }
    return values, stats


def _discover_charge_files(charge_root: Path | None) -> tuple[dict[str, Path], int]:
    if charge_root is None:
        return {}, 0
    files = [path for pattern in ("*.pqr", "*.PQR") for path in charge_root.rglob(pattern) if path.is_file()]
    by_key, _ = _first_by_alias(files, ("_pocket", "_protein"))
    return by_key, len(files)


def _charge_features(
    sample: RawSample,
    *,
    output_root: Path,
    max_ligand_atoms: int,
    max_protein_atoms: int,
    pdb2pqr_executable: str,
    protein_charge_forcefield: str,
    ligand_charge_method: str,
    openbabel_executable: str | None,
    distance_cutoff: float | None,
) -> tuple[list[float], dict[str, Any]]:
    ligand_atoms, ligand_audit = _ligand_charges_for_sample(
        sample,
        output_root=output_root,
        ligand_charge_method=ligand_charge_method,
        openbabel_executable=openbabel_executable,
    )
    protein_atoms, protein_audit = _protein_charges_for_sample(
        sample,
        output_root=output_root,
        pdb2pqr_executable=pdb2pqr_executable,
        protein_charge_forcefield=protein_charge_forcefield,
    )
    if distance_cutoff is not None:
        protein_atoms = [
            atom for atom in protein_atoms if any(_distance(lig_atom, atom) <= distance_cutoff for lig_atom in ligand_atoms)
        ]
        if not protein_atoms:
            raise DCMLGenerateDataError(f"distance cutoff removed all charged protein atoms for {sample.sample_id}")

    ligand_used = ligand_atoms[:max_ligand_atoms]
    protein_used = protein_atoms[:max_protein_atoms]
    if any(atom.charge is None for atom in ligand_used):
        raise DCMLGenerateDataError(f"missing ligand charges for real atoms in {sample.sample_id}")
    if any(atom.charge is None for atom in protein_used):
        raise DCMLGenerateDataError(f"missing protein charges for real atoms in {sample.sample_id}")

    full_grid: list[float] = []
    for ligand_atom in ligand_used:
        ligand_charge = float(ligand_atom.charge)
        for protein_atom in protein_used:
            full_grid.append(ligand_charge * float(protein_atom.charge))

    if len(full_grid) < CHARGE_FEATURES:
        full_grid.extend([0.0] * (CHARGE_FEATURES - len(full_grid)))
    values = full_grid[:CHARGE_FEATURES]
    stats = {
        "charge_formula": "row_major_first_55000_of_ligand_charge_times_protein_charge_grid",
        "charge_features": CHARGE_FEATURES,
        "ligand_charge_atoms": len(ligand_atoms),
        "protein_charge_atoms": len(protein_atoms),
        "ligand_charge_atoms_used": len(ligand_used),
        "protein_charge_atoms_used": len(protein_used),
        "charge_grid_values_before_truncation": len(ligand_used) * len(protein_used),
        "charge_values_truncated": max(0, len(ligand_used) * len(protein_used) - CHARGE_FEATURES),
        "charge_padding_values": max(0, CHARGE_FEATURES - len(ligand_used) * len(protein_used)),
        "ligand_charge_atoms_truncated": max(0, len(ligand_atoms) - len(ligand_used)),
        "protein_charge_atoms_truncated": max(0, len(protein_atoms) - len(protein_used)),
        **ligand_audit,
        **protein_audit,
    }
    return values, stats


def _split_lists(raw_root: Path) -> dict[str, list[list[str]]]:
    split_dir = raw_root / "Splits"
    result: dict[str, list[list[str]]] = {}
    for split, filenames in {
        "train": ("train_index_folder.txt", "train.txt"),
        "validation": ("valid_index_folder.txt", "validation_index_folder.txt", "valid.txt", "validation.txt"),
        "test": ("test_index_folder.txt", "test.txt"),
    }.items():
        path = next((split_dir / name for name in filenames if (split_dir / name).is_file()), None)
        if path is None:
            result[split] = []
            continue
        text = _read_text(path).strip()
        folds: list[list[str]] = []
        try:
            parsed = ast.literal_eval(text)
            if parsed and all(isinstance(item, (list, tuple, set)) for item in parsed):
                folds = [[str(value) for value in fold] for fold in parsed]
            elif isinstance(parsed, (list, tuple, set)):
                folds = [[str(value) for value in parsed]]
        except Exception:
            rows = [re.split(r"[\s,;\t]+", line.strip()) for line in text.splitlines() if line.strip()]
            folds = [[value for value in row if value] for row in rows]
        result[split] = folds
    return result


def _indices_for_fold(samples: list[RawSample], split_lists: dict[str, list[list[str]]], fold_index: int) -> tuple[dict[str, list[int]], dict[str, Any]]:
    key_to_index = {_norm_id(sample.sample_id): idx for idx, sample in enumerate(samples)}
    indices: dict[str, list[int]] = {}
    missing: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        folds = split_lists.get(split) or []
        ids = folds[min(fold_index, len(folds) - 1)] if folds else []
        split_indices = []
        for value in ids:
            key = _norm_id(value)
            if key in key_to_index:
                split_indices.append(key_to_index[key])
            else:
                missing.setdefault(split, []).append(str(value))
        indices[split] = split_indices
    if not any(indices.values()):
        n = len(samples)
        train_end = int(n * 0.7)
        valid_end = train_end + int(n * 0.15)
        indices = {
            "train": list(range(0, train_end)),
            "validation": list(range(train_end, valid_end)),
            "test": list(range(valid_end, n)),
        }
    indices["trainval"] = sorted(set(indices["train"]) | set(indices["validation"]))

    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}"] = len(set(indices[left]) & set(indices[right]))
    return indices, {"missing_split_ids": missing, "overlap_checks": overlaps}


def _save_feature_zip(features: Any, path: Path, internal_name: str) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with BytesIO() as buffer:
        np.save(buffer, np.asarray(features, dtype=np.float32), allow_pickle=False)
        payload = buffer.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(internal_name, payload)


def _save_labels(labels: Any, path: Path) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(labels, dtype=np.float64), allow_pickle=False)


def _write_sample_ids(samples: list[RawSample], output_root: Path) -> Path:
    path = output_root / "sample_ids.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "PDB_ID", "pIC50", "Ligand", "SMILES", "sdf_path", "pdb_path", "smi_path"],
        )
        writer.writeheader()
        for idx, sample in enumerate(samples):
            writer.writerow(
                {
                    "row_index": idx,
                    "PDB_ID": sample.pdb_id,
                    "pIC50": sample.label,
                    "Ligand": sample.ligand_id,
                    "SMILES": sample.smiles,
                    "sdf_path": str(sample.sdf_path),
                    "pdb_path": str(sample.pdb_path),
                    "smi_path": str(sample.smi_path or ""),
                }
            )
    return path


def _write_dropped_rows(rows: list[Mapping[str, Any]], output_root: Path) -> Path | None:
    if not rows:
        return None
    path = output_root / "reports" / "dropped_rows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_dataset_info(output_root: Path, report: Mapping[str, Any]) -> Path:
    path = output_root / "dataset_info.yaml"
    lines = [
        "schema_version: 1",
        "dataset:",
        "  name: MPro-URV",
        f"  variant_id: generated_{report['variant']}",
        "generation:",
        "  generator_script: Core/generate_dcml_from_mpro_v2.py",
        f"  feature_mode: {report['variant']}",
        "features:",
        f"  n_samples: {report['samples_kept']}",
        f"  n_features: {report['matrix_shape_summary']['n_features']}",
        "  dtype: float32",
        "  format: zip_with_single_npy",
        "labels:",
        f"  n_samples: {report['samples_kept']}",
        "  target: pIC50",
        "files:",
        "  all:",
        "    feature_zip: all_feature.zip",
        "    label_npy: all_label.npy",
        "    sample_ids_csv: sample_ids.csv",
        "  splits:",
        "    train_feature_zip: train_feature.zip",
        "    train_label.npy: train_label.npy",
        "    validation_feature_zip: validation_feature.zip",
        "    validation_label_npy: validation_label.npy",
        "    test_feature_zip: test_feature.zip",
        "    test_label_npy: test_label.npy",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_dcml_dataset_from_mpro_v2(
    raw_root,
    output_root,
    overwrite: bool = False,
    variant: str = "distance_only",
    distance_cutoff: float | None = None,
    max_ligand_atoms: int | None = None,
    max_protein_atoms: int | None = None,
    charge_source_root=None,
    protein_charge_method: str = "pdb2pqr",
    pdb2pqr_executable: str | None = None,
    protein_charge_forcefield: str = "AMBER",
    ligand_charge_method: str = "rdkit_gasteiger",
    openbabel_executable: str | None = None,
    strict: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    raw_root_path = Path(raw_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    variant = str(variant).strip()
    if variant not in SUPPORTED_VARIANTS:
        raise DCMLGenerateDataError(f"unsupported variant: {variant}. Supported variants: {', '.join(sorted(SUPPORTED_VARIANTS))}")
    protein_charge_method = str(protein_charge_method or "pdb2pqr").strip()
    ligand_charge_method = str(ligand_charge_method or "rdkit_gasteiger").strip()
    if protein_charge_method not in PROTEIN_CHARGE_METHODS:
        raise DCMLGenerateDataError(
            f"unsupported protein charge method: {protein_charge_method}. Supported: {', '.join(sorted(PROTEIN_CHARGE_METHODS))}"
        )
    if ligand_charge_method not in LIGAND_CHARGE_METHODS:
        raise DCMLGenerateDataError(
            f"unsupported ligand charge method: {ligand_charge_method}. Supported: {', '.join(sorted(LIGAND_CHARGE_METHODS))}"
        )
    if not raw_root_path.is_dir():
        raise DCMLGenerateDataError(f"missing raw folder: {raw_root_path}")
    if progress_callback:
        progress_callback(f"Raw dataset root: {raw_root_path}")
        progress_callback(f"Output prepared dataset root: {output_root_path}")
        progress_callback(f"Variant: {variant}")
    if output_root_path.exists():
        if not overwrite and any(output_root_path.iterdir()):
            raise DCMLGenerateDataError(f"output root already exists and is not empty: {output_root_path}")
        if overwrite:
            shutil.rmtree(output_root_path)
    output_root_path.mkdir(parents=True, exist_ok=True)

    max_ligand_atoms = int(max_ligand_atoms or DEFAULT_MAX_LIGAND_ATOMS)
    max_protein_atoms = int(max_protein_atoms or DEFAULT_MAX_PROTEIN_ATOMS)
    if max_ligand_atoms * max_protein_atoms != DISTANCE_FEATURES:
        raise DCMLGenerateDataError(
            "unsupported matrix shape: current DCML expects 63,360 distance columns "
            f"(36 x 1760), got {max_ligand_atoms} x {max_protein_atoms}"
        )

    samples, audit = _discover_samples(raw_root_path, output_root_path)
    if progress_callback:
        progress_callback(f"pIC50 records found: {audit['pIC50_records_parsed']}")
        progress_callback(f"SMI files found: {audit['SMI_files_found']}")
        progress_callback(f"SDF files found: {audit['SDF_files_found']}")
        progress_callback(f"Protein PDB files found: {audit['protein_PDB_files_found']}")
        progress_callback(f"Complete samples detected: {audit['complete_candidate_samples_detected']}")
        progress_callback(f"Matching audit written: {output_root_path / 'generate_data_matching_audit.json'}")
    dependency_status = _dependency_status(
        variant=variant,
        pdb2pqr_executable=pdb2pqr_executable,
        ligand_charge_method=ligand_charge_method,
        openbabel_executable=openbabel_executable,
    )
    resolved_pdb2pqr = ""
    charge_layout = (
        "distance_only has 63,360 distance columns. real_charge and full use the existing 118,360-column "
        "DCML-compatible layout: 63,360 distance columns followed by 55,000 real charge-interaction columns. "
        "The charge block is the first 55,000 values of the row-major heavy-atom "
        "ligand_charge * protein_charge grid using the same raw SDF/PDB heavy-atom ordering as distance generation."
    )
    if variant in {"real_charge", "full"}:
        if progress_callback:
            progress_callback(f"Charge tool detection: {dependency_status}")
        try:
            variant_name = "real_charge" if variant == "real_charge" else "full"
            resolved_pdb2pqr = _resolve_executable(
                pdb2pqr_executable,
                ("pdb2pqr", "pdb2pqr30"),
                f"{variant_name} requires PDB2PQR to generate protein charges from raw PDB files. "
                "Install pdb2pqr or make it available in PATH.",
            )
            if ligand_charge_method == "rdkit_gasteiger" and not dependency_status["rdkit"]:
                raise DCMLGenerateDataError(
                    f"{variant_name} requires RDKit to generate ligand partial charges from raw SDF files."
                )
            if ligand_charge_method == "openbabel":
                _resolve_executable(
                    openbabel_executable,
                    ("obabel",),
                    "real_charge/full generation requires RDKit or OpenBabel for ligand partial charges.",
                )
        except Exception as exc:
            _write_json(
                output_root_path / "generate_data_report.json",
                {
                    "raw_source_root": str(raw_root_path),
                    "output_root": str(output_root_path),
                    "timestamp": _utc_now_iso(),
                    "variant": variant,
                    "status": "failed",
                    "charge_generation_enabled": True,
                    "protein_charge_method": protein_charge_method,
                    "protein_charge_tool": pdb2pqr_executable or "pdb2pqr/pdb2pqr30",
                    "protein_charge_forcefield": protein_charge_forcefield,
                    "ligand_charge_method": ligand_charge_method,
                    "ligand_charge_tool": "rdkit" if ligand_charge_method == "rdkit_gasteiger" else (openbabel_executable or "obabel"),
                    "dependencies_found_missing": dependency_status,
                    "candidate_samples_detected": audit["complete_candidate_samples_detected"],
                    "error": str(exc),
                },
            )
            raise

    charge_summary = {
        "charge_generation_enabled": variant in {"real_charge", "full"},
        "protein_charge_method": protein_charge_method if variant in {"real_charge", "full"} else "",
        "protein_charge_tool": resolved_pdb2pqr,
        "protein_charge_forcefield": protein_charge_forcefield if variant in {"real_charge", "full"} else "",
        "ligand_charge_method": ligand_charge_method if variant in {"real_charge", "full"} else "",
        "ligand_charge_tool": "rdkit" if ligand_charge_method == "rdkit_gasteiger" else (openbabel_executable or "obabel"),
        "charge_layout": charge_layout if variant in {"real_charge", "full"} else "",
        "dependencies_found_missing": dependency_status,
        "ignored_charge_source_root": str(charge_source_root or ""),
    }

    rows: list[list[float]] = []
    labels: list[float] = []
    kept: list[RawSample] = []
    dropped: list[dict[str, Any]] = []
    shape_stats: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped_reasons: Counter[str] = Counter()
    generated_pqr_count = 0
    generated_ligand_charge_count = 0
    failed_charge_samples: list[dict[str, Any]] = []

    for sample in samples:
        try:
            dist, stats = _distance_features(
                sample,
                max_ligand_atoms=max_ligand_atoms,
                max_protein_atoms=max_protein_atoms,
                distance_cutoff=distance_cutoff,
            )
            features = dist
            if variant in {"real_charge", "full"}:
                charges, charge_stats = _charge_features(
                    sample,
                    output_root=output_root_path,
                    max_ligand_atoms=max_ligand_atoms,
                    max_protein_atoms=max_protein_atoms,
                    pdb2pqr_executable=resolved_pdb2pqr,
                    protein_charge_forcefield=protein_charge_forcefield,
                    ligand_charge_method=ligand_charge_method,
                    openbabel_executable=openbabel_executable,
                    distance_cutoff=distance_cutoff,
                )
                features = dist + charges
                stats.update(charge_stats)
                generated_pqr_count += 1
                generated_ligand_charge_count += 1
            rows.append(features)
            labels.append(sample.label)
            kept.append(sample)
            shape_stats.append({"sample_id": sample.sample_id, **stats})
            if stats["ligand_atoms_truncated"] or stats["protein_atoms_truncated"]:
                warnings.append(
                    f"{sample.sample_id}: truncated ligand {stats['ligand_atoms_truncated']} atoms, "
                    f"protein {stats['protein_atoms_truncated']} atoms"
                )
            if progress_callback and len(kept) % 10 == 0:
                progress_callback(f"Generated matrices for {len(kept)}/{len(samples)} samples.")
        except Exception as exc:
            reason = str(exc)
            if variant in {"real_charge", "full"}:
                failed_charge_samples.append({"sample_id": sample.sample_id, "reason": reason})
            if strict:
                dropped.append({"sample_id": sample.sample_id, "reason": reason})
                skipped_reasons[reason] += 1
                continue
            dropped.append({"sample_id": sample.sample_id, "reason": reason})
            skipped_reasons[reason] += 1

    if strict and failed_charge_samples:
        _write_dropped_rows(dropped, output_root_path)
        raise DCMLGenerateDataError(
            "charge generation failed for one or more samples; see reports/dropped_rows.csv. "
            f"First failure: {failed_charge_samples[0]}"
        )

    if not kept:
        _write_dropped_rows(dropped, output_root_path)
        raise DCMLGenerateDataError("no generated samples remained after coordinate parsing; see reports/dropped_rows.csv")

    split_lists = _split_lists(raw_root_path)
    indices, split_report = _indices_for_fold(kept, split_lists, 0)
    if any(split_report["overlap_checks"].values()):
        raise DCMLGenerateDataError(f"split mismatch: non-zero split overlaps {split_report['overlap_checks']}")
    if split_report["missing_split_ids"] and strict:
        raise DCMLGenerateDataError(f"split mismatch: missing split IDs {split_report['missing_split_ids']}")
    if progress_callback:
        progress_callback(f"Samples kept: {len(kept)}")
        progress_callback(f"Samples skipped: {len(dropped) + (audit['pIC50_records_parsed'] - audit['complete_candidate_samples_detected'])}")
        progress_callback(f"Split sizes: {{'train': {len(indices['train'])}, 'validation': {len(indices['validation'])}, 'test': {len(indices['test'])}, 'trainval': {len(indices['trainval'])}}}")

    output_files: list[str] = []
    _save_feature_zip(rows, output_root_path / "all_feature.zip", "urv_feature.npy")
    _save_labels(labels, output_root_path / "all_label.npy")
    output_files.extend(["all_feature.zip", "all_label.npy"])
    for split, split_indices in indices.items():
        split_rows = [rows[idx] for idx in split_indices]
        split_labels = [labels[idx] for idx in split_indices]
        _save_feature_zip(split_rows, output_root_path / f"{split}_feature.zip", f"{split}_feature.npy")
        _save_labels(split_labels, output_root_path / f"{split}_label.npy")
        output_files.extend([f"{split}_feature.zip", f"{split}_label.npy"])
    sample_ids_csv = _write_sample_ids(kept, output_root_path)
    dropped_csv = _write_dropped_rows(dropped, output_root_path)

    split_indices_json = output_root_path / "split_indices.json"
    split_payload = {
        "method": "mpro_v2_splits_first_fold",
        "n_samples": len(kept),
        "train": indices["train"],
        "validation": indices["validation"],
        "test": indices["test"],
        "trainval": indices["trainval"],
        "folds_available": {key: len(value) for key, value in split_lists.items()},
        **split_report,
    }
    _write_json(split_indices_json, split_payload)

    n_features = FULL_FEATURES if variant in {"real_charge", "full"} else DISTANCE_FEATURES
    ligand_counts = [row["ligand_atoms"] for row in shape_stats]
    protein_counts = [row["protein_atoms"] for row in shape_stats]
    report = {
        "raw_source_root": str(raw_root_path),
        "output_root": str(output_root_path),
        "timestamp": _utc_now_iso(),
        "detected_raw_structure": {
            "pIC50_records_found": audit["pIC50_records_parsed"],
            "smi_files_found": audit["SMI_files_found"],
            "sdf_files_found": audit["SDF_files_found"],
            "protein_pdb_files_found": audit["protein_PDB_files_found"],
        },
        "variant": variant,
        "pIC50_entries_found": audit["pIC50_records_parsed"],
        "smi_files_found": audit["SMI_files_found"],
        "sdf_files_found": audit["SDF_files_found"],
        "protein_pdb_files_found": audit["protein_PDB_files_found"],
        **charge_summary,
        "generated_pqr_count": generated_pqr_count,
        "generated_ligand_charge_count": generated_ligand_charge_count,
        "failed_charge_samples": failed_charge_samples[:100],
        "candidate_samples_detected": audit["complete_candidate_samples_detected"],
        "samples_kept": len(kept),
        "samples_skipped": len(dropped) + (audit["pIC50_records_parsed"] - audit["complete_candidate_samples_detected"]),
        "skipped_reasons": {**audit["skipped_reasons"], **dict(skipped_reasons)},
        "matrix_shape_summary": {
            "shape": [len(kept), n_features],
            "n_features": n_features,
            "distance_features": DISTANCE_FEATURES,
            "charge_features": CHARGE_FEATURES if variant in {"real_charge", "full"} else 0,
            "dtype": "float32",
            "max_ligand_atoms": max_ligand_atoms,
            "max_protein_atoms": max_protein_atoms,
            "ligand_atoms_min": min(ligand_counts),
            "ligand_atoms_max": max(ligand_counts),
            "protein_atoms_min": min(protein_counts),
            "protein_atoms_max": max(protein_counts),
        },
        "split_sizes": {key: len(value) for key, value in indices.items()},
        "missing_split_ids": split_report["missing_split_ids"],
        "overlap_checks": split_report["overlap_checks"],
        "output_files_created": output_files + ["sample_ids.csv", "split_indices.json"],
        "generated_intermediate_files": {
            "protein_pqr_dir": str(output_root_path / "intermediate" / "protein_pqr"),
            "ligand_charges_dir": str(output_root_path / "intermediate" / "ligand_charges"),
            "charge_audit_dir": str(output_root_path / "intermediate" / "charge_audit"),
        },
        "warnings": warnings[:200],
        "warnings_count": len(warnings),
        "exact_format_generated": "DCML zip_with_single_npy feature matrices plus 1D label npy files",
        "reports": {
            "matching_audit_json": str(output_root_path / "generate_data_matching_audit.json"),
            "dropped_rows_csv": str(dropped_csv or ""),
        },
    }
    _write_json(output_root_path / "metadata.json", report)
    _write_json(output_root_path / "generate_data_report.json", report)
    dataset_info = _write_dataset_info(output_root_path, report)
    report["reports"]["metadata_json"] = str(output_root_path / "metadata.json")
    report["reports"]["generate_data_report_json"] = str(output_root_path / "generate_data_report.json")
    report["reports"]["dataset_info_yaml"] = str(dataset_info)
    report["outputs"] = {
        "output_feature_dir": str(output_root_path),
        "report_json": str(output_root_path / "generate_data_report.json"),
        "sample_ids_csv": str(sample_ids_csv),
    }
    if progress_callback:
        progress_callback(f"Feature shape: {report['matrix_shape_summary']['shape']}")
        progress_callback(f"Metadata written: {output_root_path / 'metadata.json'}")
        progress_callback(f"Generate Data report written: {output_root_path / 'generate_data_report.json'}")
        progress_callback(f"Dataset info written: {dataset_info}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DCML prepared data from a raw MPro-v2-like dataset.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=sorted(SUPPORTED_VARIANTS), default="distance_only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--distance-cutoff", type=float, default=None)
    parser.add_argument("--max-ligand-atoms", type=int, default=DEFAULT_MAX_LIGAND_ATOMS)
    parser.add_argument("--max-protein-atoms", type=int, default=DEFAULT_MAX_PROTEIN_ATOMS)
    parser.add_argument("--protein-charge-method", choices=sorted(PROTEIN_CHARGE_METHODS), default="pdb2pqr")
    parser.add_argument("--pdb2pqr-executable", default=None)
    parser.add_argument("--protein-charge-forcefield", default="AMBER")
    parser.add_argument("--ligand-charge-method", choices=sorted(LIGAND_CHARGE_METHODS), default="rdkit_gasteiger")
    parser.add_argument("--openbabel-executable", default=None)
    parser.add_argument("--non-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = generate_dcml_dataset_from_mpro_v2(
            raw_root=args.raw_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
            variant=args.variant,
            distance_cutoff=args.distance_cutoff,
            max_ligand_atoms=args.max_ligand_atoms,
            max_protein_atoms=args.max_protein_atoms,
            protein_charge_method=args.protein_charge_method,
            pdb2pqr_executable=args.pdb2pqr_executable,
            protein_charge_forcefield=args.protein_charge_forcefield,
            ligand_charge_method=args.ligand_charge_method,
            openbabel_executable=args.openbabel_executable,
            strict=args.strict,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
