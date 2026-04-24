"""
@file constants.py
@author Mohamed EL BOUKHIARI
@brief Constants used by the EGNN module.
"""

import os

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_GRAPHS_DIR = os.path.join(MODULE_ROOT, "Graphs_EGNN")
DEFAULT_MODELS_DIR = os.path.join(MODULE_ROOT, "Models_EGNN")
DEFAULT_RESULTS_DIR = os.path.join(MODULE_ROOT, "Results_EGNN")
DEFAULT_TEMP_RUNS_DIR = os.path.join(MODULE_ROOT, "temp_runs")

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
