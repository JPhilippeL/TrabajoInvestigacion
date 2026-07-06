"""
@file widedta_mpro_urv_converter.py
@author Mohamed EL BOUKHIARI
@brief Convert MPro-URV data to the WideDTA input format.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_WIDEDTA_FILES = ("ligands_can.txt", "proteins.txt", "motif2.txt", "Y")


def read_sequence_from_arg(value: str | None, file_path: str | None) -> str | None:
    if value:
        return value.strip().replace("\n", "").replace(" ", "")

    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            return "".join(line.strip() for line in file if not line.startswith(">"))

    return None


def write_json(path: str, data: OrderedDict[str, str] | dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def write_affinity(path: str, values: np.ndarray) -> None:
    with open(path, "wb") as file:
        pickle.dump(values.astype(np.float32), file)


def convert_from_csv(
    input_csv: str,
    output_dir: str,
    smiles_column: str,
    pic50_column: str,
    protein_sequence: str,
    motif_sequence: str,
    id_column: str | None = None,
    sep: str | None = None,
) -> dict:
    """
    @brief Convert a CSV/TSV file with SMILES and pIC50 columns to WideDTA format.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataframe = pd.read_csv(input_csv, sep=sep)

    for column in (smiles_column, pic50_column):
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' was not found in {input_csv}.")

    dataframe = dataframe[[smiles_column, pic50_column] + ([id_column] if id_column else [])].copy()
    dataframe = dataframe.dropna(subset=[smiles_column, pic50_column])
    dataframe[pic50_column] = dataframe[pic50_column].astype(float)
    dataframe = dataframe.reset_index(drop=True)

    ligands: OrderedDict[str, str] = OrderedDict()

    for row_index, row in dataframe.iterrows():
        if id_column and pd.notna(row[id_column]):
            key = str(row[id_column])
        else:
            key = str(row_index)

        ligands[key] = str(row[smiles_column])

    proteins = OrderedDict({"0": protein_sequence})
    motifs = OrderedDict({"0": motif_sequence})
    affinity = dataframe[pic50_column].to_numpy(dtype=np.float32).reshape(-1, 1)

    write_json(os.path.join(output_dir, "ligands_can.txt"), ligands)
    write_json(os.path.join(output_dir, "proteins.txt"), proteins)
    write_json(os.path.join(output_dir, "motif2.txt"), motifs)
    write_affinity(os.path.join(output_dir, "Y"), affinity)

    return {
        "status": "success",
        "mode": "csv",
        "output_dir": output_dir,
        "num_ligands": len(ligands),
        "num_proteins": 1,
        "Y_shape": tuple(affinity.shape),
    }


def convert_from_deepdta_format(
    source_deepdta_dir: str,
    output_dir: str,
    motif_sequence: str | None = None,
    use_protein_as_motif: bool = False,
) -> dict:
    """
    @brief Convert an existing DeepDTA-style MPro folder to WideDTA by adding motif2.txt.
    @details
    This copies ligands_can.txt, proteins.txt and Y. It creates motif2.txt.
    If use_protein_as_motif=True, motif2.txt duplicates proteins.txt. That is a runnable baseline,
    not the original biological motif representation from the WideDTA paper.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in ("ligands_can.txt", "proteins.txt", "Y"):
        source_path = os.path.join(source_deepdta_dir, filename)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Required DeepDTA source file not found: {source_path}")

        shutil.copy2(source_path, os.path.join(output_dir, filename))

    with open(os.path.join(source_deepdta_dir, "proteins.txt"), "r", encoding="utf-8") as file:
        proteins = json.load(file, object_pairs_hook=OrderedDict)

    if motif_sequence:
        motifs = OrderedDict((str(key), motif_sequence) for key in proteins.keys())
    elif use_protein_as_motif:
        motifs = OrderedDict((str(key), str(value)) for key, value in proteins.items())
    else:
        raise ValueError(
            "WideDTA requires motif2.txt. Provide --motif-sequence/--motif-sequence-file "
            "or use --use-protein-as-motif for a technical baseline."
        )

    write_json(os.path.join(output_dir, "motif2.txt"), motifs)

    return {
        "status": "success",
        "mode": "deepdta_dir",
        "output_dir": output_dir,
        "num_proteins": len(proteins),
        "motif_source": "protein_sequence" if use_protein_as_motif and not motif_sequence else "provided_motif_sequence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MPro-URV data to WideDTA format.")

    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--input-csv")
    parser.add_argument("--sep", default=None)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--pic50-column", default="pIC50")
    parser.add_argument("--id-column", default=None)

    parser.add_argument("--source-deepdta-dir", default=None)

    parser.add_argument("--protein-sequence", default=None)
    parser.add_argument("--protein-sequence-file", default=None)
    parser.add_argument("--motif-sequence", default=None)
    parser.add_argument("--motif-sequence-file", default=None)
    parser.add_argument("--use-protein-as-motif", action="store_true")

    args = parser.parse_args()

    motif_sequence = read_sequence_from_arg(args.motif_sequence, args.motif_sequence_file)

    if args.source_deepdta_dir:
        result = convert_from_deepdta_format(
            source_deepdta_dir=args.source_deepdta_dir,
            output_dir=args.output_dir,
            motif_sequence=motif_sequence,
            use_protein_as_motif=args.use_protein_as_motif,
        )
    else:
        if not args.input_csv:
            raise ValueError("Provide either --input-csv or --source-deepdta-dir.")

        protein_sequence = read_sequence_from_arg(args.protein_sequence, args.protein_sequence_file)

        if not protein_sequence:
            raise ValueError("CSV conversion requires --protein-sequence or --protein-sequence-file.")

        if not motif_sequence:
            if args.use_protein_as_motif:
                motif_sequence = protein_sequence
            else:
                raise ValueError(
                    "WideDTA requires a motif sequence. Provide --motif-sequence/--motif-sequence-file "
                    "or use --use-protein-as-motif for a runnable baseline."
                )

        result = convert_from_csv(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            smiles_column=args.smiles_column,
            pic50_column=args.pic50_column,
            protein_sequence=protein_sequence,
            motif_sequence=motif_sequence,
            id_column=args.id_column,
            sep=args.sep,
        )

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
