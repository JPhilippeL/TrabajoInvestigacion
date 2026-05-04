"""
@file constants.py
@author Mohamed EL BOUKHIARI
@brief Default constants for the DeepDTA module.
"""

from __future__ import annotations

import os


MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_DATASET = "mpro_urv"

DEFAULT_OUTPUT_ROOT = os.path.join(
    MODULE_ROOT,
    "results",
    "deepdta_hpo",
    "runs",
)

DEFAULT_DEVICE = "auto"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 3

DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.2

DEFAULT_LR_VALUES = "0.003,0.001,0.0005"
DEFAULT_BATCH_SIZE_VALUES = "4,8"

# 0 means no limit. Useful only for quick debug runs from the GUI.
DEFAULT_MAX_TRAIN_BATCHES = 0
