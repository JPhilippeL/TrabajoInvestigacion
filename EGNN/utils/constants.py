"""
@file constants.py
@author Mohamed EL BOUKHIARI
@brief Constants used by the EGNN module.
"""

from __future__ import annotations

import os

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATASET_DIR = os.path.join(MODULE_ROOT, "MPro-URV_Version2")
DEFAULT_PIC50_FILE = os.path.join(DEFAULT_DATASET_DIR, "pIC50.txt")
DEFAULT_LIGAND_SDF_DIR = os.path.join(DEFAULT_DATASET_DIR, "Ligand", "Ligand_SDF")
DEFAULT_PROTEIN_PDB_DIR = os.path.join(DEFAULT_DATASET_DIR, "Protein", "Protein_PDB")

DEFAULT_GRAPHS_DIR = os.path.join(MODULE_ROOT, "Graphs_EGNN")
DEFAULT_MODELS_DIR = os.path.join(MODULE_ROOT, "Models_EGNN")
DEFAULT_RESULTS_DIR = os.path.join(MODULE_ROOT, "Results_EGNN")
DEFAULT_TEMP_RUNS_DIR = os.path.join(MODULE_ROOT, "temp_runs")

DEFAULT_TRAIN_SPLIT_FILE = os.path.join(DEFAULT_DATASET_DIR, "train_index_folder.txt")
DEFAULT_VAL_SPLIT_FILE = os.path.join(DEFAULT_DATASET_DIR, "valid_index_folder.txt")
DEFAULT_TEST_SPLIT_FILE = os.path.join(DEFAULT_DATASET_DIR, "test_index_folder.txt")

DEFAULT_DEVICE = "auto"

DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-4
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42

DEFAULT_CUTOFF_EDGES = 5.0
DEFAULT_CUTOFF_PROT = 6.0

DEFAULT_LR_VALUES = "5e-5,1e-4,5e-4,1e-3"
DEFAULT_HIDDEN_DIM_VALUES = "32,64,128"
DEFAULT_BATCH_SIZE_VALUES = "2,4,8"
