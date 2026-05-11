"""
@file hyperparameter_search_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search dialog for the EDNN module.
@details
Kept as a compatibility wrapper. The active EDNN hyperparameter-search dialog
is BatchTrainDialog, because batch training is implemented as a search over
multiple learning-rate, hidden-dimension and batch-size values.
"""

from __future__ import annotations

from EDNN.ui.dialogs.batch_train_ednn_dialog import BatchTrainDialog


class HyperparameterSearchEDNNDialog(BatchTrainDialog):
    """
    @brief Backward-compatible EDNN hyperparameter search dialog.
    """

    def __init__(self, parent=None) -> None:
        """
        @brief Initialize the compatibility wrapper.

        @param parent Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Hyperparameter Search Configuration - EDNN")
