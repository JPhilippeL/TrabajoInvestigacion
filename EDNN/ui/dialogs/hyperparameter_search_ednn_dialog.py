"""
@file hyperparameter_search_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search dialog for the EDNN module.
@details
Kept as a compatibility wrapper. The active EDNN hyperparameter-search dialog
is BatchTrainDialog, because batch training is implemented as search over
multiple lr/hidden_dim/batch_size values.
"""

from EDNN.ui.dialogs.batch_train_ednn_dialog import BatchTrainDialog


class HyperparameterSearchEDNNDialog(BatchTrainDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hyperparameter Search Configuration - EDNN")
