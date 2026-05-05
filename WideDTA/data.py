"""
@file data.py
@author Mohamed EL BOUKHIARI
@brief Dataset utilities for the WideDTA module.
"""

from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import deepsmiles
except ImportError as exc:  # pragma: no cover
    deepsmiles = None
    DEEPSMILES_IMPORT_ERROR = exc
else:
    DEEPSMILES_IMPORT_ERROR = None


JsonMapping = Dict[str, str]
TokenMapping = Dict[int, Tuple[str, ...]]
TensorMapping = Dict[int, torch.Tensor]


def load_json_mapping(path: str) -> OrderedDict[str, str]:
    """
    @brief Load a JSON dictionary while preserving item order.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file, object_pairs_hook=OrderedDict)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object/dictionary: {path}")

    return OrderedDict((str(key), str(value)) for key, value in data.items())


def filter_by_length(values: JsonMapping, max_length: int | None) -> OrderedDict[int, str]:
    """
    @brief Keep entries whose sequence length is lower than or equal to max_length.
    @details New integer keys are positional indices, matching matrix Y indexing.
    """
    filtered: OrderedDict[int, str] = OrderedDict()

    for positional_index, sequence in enumerate(values.values()):
        if max_length is None or len(sequence) <= max_length:
            filtered[positional_index] = sequence

    return filtered


def smiles_to_deepsmiles(ligands: OrderedDict[int, str]) -> OrderedDict[int, str]:
    """
    @brief Convert canonical SMILES strings into DeepSMILES strings.
    """
    if deepsmiles is None:
        raise ImportError(
            "The 'deepsmiles' package is required by WideDTA. "
            "Install it with: pip install deepsmiles"
        ) from DEEPSMILES_IMPORT_ERROR

    converter = deepsmiles.Converter(rings=True, branches=True)
    converted: OrderedDict[int, str] = OrderedDict()

    for index, smiles in ligands.items():
        converted[index] = converter.encode(smiles)

    return converted


def make_words(values: OrderedDict[int, str], word_len: int) -> TokenMapping:
    """
    @brief Convert sequences into WideDTA word tokens.
    @details This reproduces the original WideDTA tokenization logic.
    """
    if word_len <= 0:
        raise ValueError("word_len must be strictly positive.")

    tokenized: TokenMapping = {}

    for key, sequence in values.items():
        tokens: list[str] = []

        for offset in range(word_len):
            sequence_length = len(sequence)

            for position in range(offset, sequence_length, word_len):
                token = sequence[position:position + word_len]

                if len(token) == word_len:
                    tokens.append(token)

        tokenized[key] = tuple(tokens)

    return tokenized


def build_vocabulary(tokenized_values: TokenMapping) -> list[str]:
    """
    @brief Build a deterministic vocabulary from tokenized sequences.
    """
    vocabulary = sorted({token for tokens in tokenized_values.values() for token in tokens})

    if not vocabulary:
        raise ValueError("Cannot build a vocabulary from empty token sequences.")

    return vocabulary


def one_hot_encode(tokenized_values: TokenMapping) -> TensorMapping:
    """
    @brief One-hot encode tokenized sequences as tensors shaped [vocab_size, max_tokens].
    """
    vocabulary = build_vocabulary(tokenized_values)
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    max_length = max(len(tokens) for tokens in tokenized_values.values())

    encoded: TensorMapping = {}

    for key, tokens in tokenized_values.items():
        matrix = np.zeros((len(vocabulary), max_length), dtype=np.float32)

        for position, token in enumerate(tokens):
            matrix[token_to_index[token], position] = 1.0

        encoded[key] = torch.from_numpy(matrix)

    return encoded


def load_affinity_matrix(path: str) -> np.ndarray:
    """
    @brief Load a WideDTA affinity matrix from a pickle file.
    """
    with open(path, "rb") as file:
        matrix = pickle.load(file, encoding="latin1")

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    if matrix.ndim != 2:
        raise ValueError(f"Affinity matrix must be 2D. Got shape={matrix.shape}")

    return matrix


class WideDTADataset(Dataset):
    """
    @brief WideDTA dataset supporting Davis, KIBA and MPro-URV formatted folders.
    """

    def __init__(
        self,
        ligand_path: str,
        protein_path: str,
        motif_path: str,
        affinity_path: str,
        ligand_max_len: int | None = 50,
        protein_max_len: int | None = 600,
        ligand_word_len: int = 8,
        protein_word_len: int = 3,
        motif_word_len: int = 3,
        skip_nan_targets: bool = True,
    ) -> None:
        self.ligand_path = ligand_path
        self.protein_path = protein_path
        self.motif_path = motif_path
        self.affinity_path = affinity_path

        raw_ligands = load_json_mapping(ligand_path)
        raw_proteins = load_json_mapping(protein_path)
        raw_motifs = load_json_mapping(motif_path)

        filtered_ligands = filter_by_length(raw_ligands, ligand_max_len)
        filtered_proteins = filter_by_length(raw_proteins, protein_max_len)
        filtered_motifs = filter_by_length(raw_motifs, None)

        if not filtered_ligands:
            raise ValueError("No ligands left after filtering. Increase ligand_max_len.")

        if not filtered_proteins:
            raise ValueError("No proteins left after filtering. Increase protein_max_len.")

        if not filtered_motifs:
            raise ValueError("No motifs available.")

        deep_ligands = smiles_to_deepsmiles(filtered_ligands)

        self.ligand_tokens = make_words(deep_ligands, ligand_word_len)
        self.protein_tokens = make_words(filtered_proteins, protein_word_len)
        self.motif_tokens = make_words(filtered_motifs, motif_word_len)

        self.ligands = one_hot_encode(self.ligand_tokens)
        self.proteins = one_hot_encode(self.protein_tokens)
        self.motifs = one_hot_encode(self.motif_tokens)
        self.affinity = load_affinity_matrix(affinity_path)

        self.pairs: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = []
        self._build_pairs(skip_nan_targets=skip_nan_targets)

        if not self.pairs:
            raise ValueError("No usable ligand-protein pairs were created.")

    def _build_pairs(self, skip_nan_targets: bool) -> None:
        """
        @brief Build all ligand-protein-motif-target samples.
        """
        motif_items = list(self.motifs.items())

        for ligand_index, ligand_tensor in self.ligands.items():
            for protein_position, (protein_index, protein_tensor) in enumerate(self.proteins.items()):
                if ligand_index >= self.affinity.shape[0]:
                    continue

                if protein_position >= self.affinity.shape[1]:
                    continue

                target_value = self.affinity[ligand_index, protein_position]

                if skip_nan_targets and np.isnan(target_value):
                    continue

                if np.isnan(target_value):
                    target_value = 0.0

                if protein_position < len(motif_items):
                    motif_tensor = motif_items[protein_position][1]
                else:
                    motif_tensor = motif_items[0][1]

                target_tensor = torch.tensor([float(target_value)], dtype=torch.float32)
                self.pairs.append(((ligand_tensor, protein_tensor, motif_tensor), target_tensor))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        return self.pairs[index]

    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        """
        @brief Return one sample shape per WideDTA input branch.
        """
        (ligand, protein, motif), _target = self.pairs[0]

        return {
            "ligand": tuple(ligand.shape),
            "protein": tuple(protein.shape),
            "motif": tuple(motif.shape),
        }
