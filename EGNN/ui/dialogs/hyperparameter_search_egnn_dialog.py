"""
@file hyperparameter_search_egnn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Backward-compatible alias for the EGNN hyperparameter search dialog.
@details
The active dialog is BatchTrainDialog because, in this module, batch training
is the hyperparameter search workflow.
"""

from __future__ import annotations

from EGNN.ui.dialogs.batch_train_egnn_dialog import BatchTrainDialog


class HyperparameterSearchEGNNDialog(BatchTrainDialog):
    """
    @brief Alias kept for compatibility with previous imports.
    """
    pass
