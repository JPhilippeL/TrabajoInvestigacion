import csv
import json
import pickle
import random
from pathlib import Path

import pandas as pd

from data_pipeline.common import (
    copy_file,
    ensure_directory,
    get_input_files,
    save_json,
    write_split_csv,
)
from data_pipeline.pic50_utils import load_pic50
from job_config.planet.PlanetDataConfig import PlanetDataConfig

try:
    from data_pipeline.chemutils import ComplexPocket
except ImportError:
    from data_pipeline.chemutils import ComplexPocket


class PlanetGraph:
    def __init__(self, config: PlanetDataConfig):
        self.config = config

        self.output_path = Path(config.output_path)
        self.protein_path = Path(config.protein_path)
        self.ligand_path = Path(config.ligand_path)
        self.pic50_path = Path(config.pic50_path)

        self.raw_dir = self.output_path / "raw"
        self.metadata_dir = self.output_path / "metadata"
        self.pkl_dir = self.metadata_dir / "pkl"
        self.pocket_root = self.pkl_dir / "pockets"

    def prepare_raw_and_labels(self, overwrite=False):
        pic50 = load_pic50(self.pic50_path)

        ensure_directory(self.raw_dir)
        ensure_directory(self.metadata_dir)

        prepared = []
        ok_ids = []

        for pdb_id, value in pic50.items():
            pdb_id = str(pdb_id).strip().upper()

            problems = []
            protein_src = None
            ligand_src = None

            try:
                protein_src, ligand_src = get_input_files(
                    self.protein_path,
                    self.ligand_path,
                    pdb_id,
                )
            except Exception as exc:
                problems.append(str(exc))

            complex_dir = self.raw_dir / pdb_id

            protein_dst = complex_dir / f"{pdb_id}.pdb"
            ligand_dst = complex_dir / f"{pdb_id}.sdf"

            if not problems:
                ensure_directory(complex_dir)
                copy_file(protein_src, protein_dst, overwrite=overwrite)
                copy_file(ligand_src, ligand_dst, overwrite=overwrite)
                ok_ids.append(pdb_id)

            prepared.append(
                {
                    "complex_id": pdb_id,
                    "pIC50": value,
                    "source_protein_pdb": "" if protein_src is None else str(protein_src),
                    "source_ligand_sdf": "" if ligand_src is None else str(ligand_src),
                    "raw_protein_pdb": str(protein_dst) if protein_dst.exists() else "",
                    "raw_ligand_sdf": str(ligand_dst) if ligand_dst.exists() else "",
                    "ok": not problems,
                    "problems": "; ".join(problems),
                }
            )

        labels_csv = self.metadata_dir / "labels.csv"
        prepared_csv = self.metadata_dir / "prepared.csv"

        with labels_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["complex_id", "PDB_code", "affinity", "pK", "pIC50"],
            )
            writer.writeheader()

            for pdb_id in ok_ids:
                writer.writerow(
                    {
                        "complex_id": pdb_id,
                        "PDB_code": pdb_id,
                        "affinity": pic50[pdb_id],
                        "pK": pic50[pdb_id],
                        "pIC50": pic50[pdb_id],
                    }
                )

        with prepared_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "complex_id",
                "pIC50",
                "source_protein_pdb",
                "source_ligand_sdf",
                "raw_protein_pdb",
                "raw_ligand_sdf",
                "ok",
                "problems",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prepared)

        return ok_ids, pic50, labels_csv, prepared_csv

    def split_ids(self, ids):
        ids = list(ids)

        random.seed(self.config.seed)
        random.shuffle(ids)

        n_total = len(ids)
        n_train = int(n_total * self.config.train_ratio)
        n_valid = int(n_total * self.config.valid_ratio)

        train_ids = ids[:n_train]
        valid_ids = ids[n_train : n_train + n_valid]
        core_ids = ids[n_train + n_valid :]

        if len(core_ids) == 0 and len(valid_ids) > 1:
            core_ids.append(valid_ids.pop())

        if len(valid_ids) == 0 and len(train_ids) > 1:
            valid_ids.append(train_ids.pop())

        return train_ids, valid_ids, core_ids

    def build_metadata_file(self, ids, pic50):
        train_ids, valid_ids, core_ids = self.split_ids(ids)

        train_csv = write_split_csv(
            self.metadata_dir / "train.csv",
            train_ids,
            pic50,
        )

        valid_csv = write_split_csv(
            self.metadata_dir / "valid.csv",
            valid_ids,
            pic50,
        )

        core_csv = write_split_csv(
            self.metadata_dir / "core.csv",
            core_ids,
            pic50,
        )

        return {
            "train": train_ids,
            "valid": valid_ids,
            "core": core_ids,
            "train_csv": train_csv,
            "valid_csv": valid_csv,
            "core_csv": core_csv,
        }

    def build_one_pocket(self, pdb_id, affinity, overwrite=False):
        pdb_id = str(pdb_id).strip().upper()

        protein_path = self.raw_dir / pdb_id / f"{pdb_id}.pdb"
        ligand_sdf_path = self.raw_dir / pdb_id / f"{pdb_id}.sdf"

        if not protein_path.exists():
            raise FileNotFoundError(f"{protein_path} does not exist")

        if not ligand_sdf_path.exists():
            raise FileNotFoundError(f"{ligand_sdf_path} does not exist")

        pocket_dir = self.pocket_root / pdb_id
        ensure_directory(pocket_dir)

        pocket_path = pocket_dir / f"{pdb_id}_pocket.pkl"

        if pocket_path.exists() and not overwrite:
            return pocket_path

        pocket = ComplexPocket(
            str(protein_path),
            str(ligand_sdf_path),
            float(affinity),
            None,
        )

        if getattr(pocket, "res_count", 0) <= 0:
            raise ValueError(f"Pocket construction produced zero residues for {pdb_id}")

        with pocket_path.open("wb") as f:
            pickle.dump(pocket, f, pickle.HIGHEST_PROTOCOL)

        return pocket_path

    def read_split_csv(self, csv_path):
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, index_col=0)

        required_columns = ["PDB_code", "pK", "type"]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"{column} not in {csv_path}")

        return df

    def build_split_pocket(
        self, csv_path, output_pickle, overwrite_pockets=False, log_callback=None
    ):
        csv_path = Path(csv_path)
        output_pickle = Path(output_pickle)

        df = self.read_split_csv(csv_path)

        records = []
        skipped = []

        for _, row in df.iterrows():
            pdb_id = str(row["PDB_code"]).strip().upper()
            affinity = float(row["pK"])

            try:
                pocket_path = self.build_one_pocket(
                    pdb_id=pdb_id,
                    affinity=affinity,
                    overwrite=overwrite_pockets,
                )

                records.append([str(pocket_path), affinity])

            except Exception as exc:
                skipped.append(
                    {
                        "complex_id": pdb_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                log_callback(f"Skipping {pdb_id}: {exc}")

        ensure_directory(output_pickle.parent)

        with output_pickle.open("wb") as f:
            pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)

        return records, skipped

    def build_all_pickle(self, log_callback=None, overwrite_pockets=False):
        ensure_directory(self.pkl_dir)
        ensure_directory(self.pocket_root)

        split_files = {
            "train": self.metadata_dir / "train.csv",
            "valid": self.metadata_dir / "valid.csv",
            "core": self.metadata_dir / "core.csv",
        }

        summary = {
            "pkl_dir": str(self.pkl_dir),
            "pocket_root": str(self.pocket_root),
            "splits": {},
            "skipped": {},
        }

        for split_name, csv_path in split_files.items():
            output_pickle = self.pkl_dir / f"{split_name}.pkl"

            records, skipped = self.build_split_pocket(
                csv_path=csv_path,
                output_pickle=output_pickle,
                overwrite_pockets=overwrite_pockets,
                log_callback=log_callback,
            )

            summary["splits"][split_name] = {
                "csv_path": str(csv_path),
                "pickle_path": str(output_pickle),
                "count_built": len(records),
            }

            summary["skipped"][split_name] = skipped

            log_callback(f"{split_name} built {len(records)} records -> {output_pickle}")

        save_json(summary, self.pkl_dir / "build_pkl_summary.json")

        return summary

    def build_data(
        self,
        overwrite=False,
        overwrite_pockets=False,
        log_callback=None,
    ):
        ensure_directory(self.output_path)

        ok_ids, pic50, labels_csv, prepared_csv = self.prepare_raw_and_labels(
            overwrite=overwrite,
        )

        split_info = self.build_metadata_file(
            ids=ok_ids,
            pic50=pic50,
        )

        pkl_summary = self.build_all_pickle(
            overwrite_pockets=overwrite_pockets, log_callback=log_callback
        )

        manifest = {
            "protein_path": str(self.protein_path),
            "ligand_path": str(self.ligand_path),
            "pic50_path": str(self.pic50_path),
            "output_path": str(self.output_path),
            "raw_dir": str(self.raw_dir),
            "metadata_dir": str(self.metadata_dir),
            "labels_csv": str(labels_csv),
            "prepared_csv": str(prepared_csv),
            "n_labels": len(pic50),
            "n_prepared_ok": len(ok_ids),
            "split_counts": {
                "train": len(split_info["train"]),
                "valid": len(split_info["valid"]),
                "core": len(split_info["core"]),
            },
            "split_csv": {
                "train": str(split_info["train_csv"]),
                "valid": str(split_info["valid_csv"]),
                "core": str(split_info["core_csv"]),
            },
            "pkl_summary": pkl_summary,
        }

        manifest_path = self.metadata_dir / "build_data_manifest.json"
        save_json(manifest, manifest_path)

        if log_callback:
            if hasattr(log_callback, "info"):
                log_callback.info(json.dumps(manifest, indent=2, ensure_ascii=False))
            else:
                log_callback(json.dumps(manifest, indent=2, ensure_ascii=False))

        return manifest
