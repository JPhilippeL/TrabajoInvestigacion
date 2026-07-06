import logging

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class HyperparameterSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Hyperparameter Tuning")
        self.resize(700, 500)

        self.settings = QSettings("Investigacion", "HyperparameterSearch")

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("Train split file (.txt)")
        self.train_split_input.setText(
            str(self.settings.value("hpgign/train_split_file", ""))
        )

        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("Validation split file (.txt)")
        self.val_split_input.setText(
            str(self.settings.value("hpgign/val_split_file", ""))
        )

        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("Test split file (.txt)")
        self.test_split_input.setText(
            str(self.settings.value("hpgign/test_split_file", ""))
        )

        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.save_directory_input = QLineEdit()
        self.save_directory_input.setPlaceholderText("Save directory")
        self.save_directory_input.setText(
            str(self.settings.value("hpgign/save_directory", ""))
        )

        self.save_directory_btn = QPushButton("Select...")
        self.save_directory_btn.clicked.connect(self.browse_save_directory)

        self.graph_directory_input = QLineEdit()
        self.graph_directory_input.setPlaceholderText("Graph directory")
        self.graph_directory_input.setText(
            str(self.settings.value("hpgign/graph_directory", ""))
        )

        self.graph_directory_btn = QPushButton("Select...")
        self.graph_directory_btn.clicked.connect(self.browse_graph_directory)

        self.cpu_per_trials_input = QSpinBox()
        self.cpu_per_trials_input.setRange(1, 100)
        self.cpu_per_trials_input.setValue(
            int(self.settings.value("hpgign/cpu_per_trials", 4))
        )
        self.cpu_per_trials_input.setToolTip(
            "Number of cpu used for one trial."
        )

        self.gpu_per_trials_input = QSpinBox()
        self.gpu_per_trials_input.setRange(0, 100)
        self.gpu_per_trials_input.setValue(
            int(self.settings.value("hpgign/gpu_per_trials", 0))
        )
        self.gpu_per_trials_input.setToolTip(
            "Number of gpu used for a trial."
        )

        self.number_of_trials = QSpinBox()
        self.number_of_trials.setRange(1, 1000)
        self.number_of_trials.setValue(
            int(self.settings.value("hpgign/number_of_trials", 20))
        )
        self.number_of_trials.setToolTip(
            "Number of random combinaison to test."
        )

        self.node_dim = QSpinBox()
        self.node_dim.setValue(14)
        self.node_dim.setReadOnly(True)
        self.node_dim.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.node_dim.setToolTip("Feature dimension fixed to 14.")

        self.hidden_dim_min = QSpinBox()
        self.hidden_dim_min.setRange(16, 1024)
        self.hidden_dim_min.setSingleStep(16)
        self.hidden_dim_min.setValue(
            int(self.settings.value("hpgign/hidden_dim_min", 32))
        )

        self.hidden_dim_max = QSpinBox()
        self.hidden_dim_max.setRange(16, 1024)
        self.hidden_dim_max.setSingleStep(16)
        self.hidden_dim_max.setValue(
            int(self.settings.value("hpgign/hidden_dim_max", 256))
        )

        self.batch_size_min = QSpinBox()
        self.batch_size_min.setRange(1, 512)
        self.batch_size_min.setSingleStep(1)
        self.batch_size_min.setValue(
            int(self.settings.value("hpgign/batch_size_min", 4))
        )

        self.batch_size_max = QSpinBox()
        self.batch_size_max.setRange(1, 512)
        self.batch_size_max.setSingleStep(1)
        self.batch_size_max.setValue(
            int(self.settings.value("hpgign/batch_size_max", 8))
        )

        self.lr_min = QDoubleSpinBox()
        self.lr_min.setDecimals(8)
        self.lr_min.setRange(1e-8, 1.0)
        self.lr_min.setSingleStep(1e-4)
        self.lr_min.setValue(
            float(self.settings.value("hpgign/lr_min", 1e-4))
        )
        self.lr_min.setToolTip("Learning rate minimale. Exemple : 1e-4.")

        self.lr_max = QDoubleSpinBox()
        self.lr_max.setDecimals(8)
        self.lr_max.setRange(1e-8, 1.0)
        self.lr_max.setSingleStep(1e-4)
        self.lr_max.setValue(
            float(self.settings.value("hpgign/lr_max", 1e-3))
        )
        self.lr_max.setToolTip("Learning rate maximale. Exemple : 1e-3.")

        self.weight_decay_min = QDoubleSpinBox()
        self.weight_decay_min.setDecimals(10)
        self.weight_decay_min.setRange(1e-10, 1.0)
        self.weight_decay_min.setSingleStep(1e-6)
        self.weight_decay_min.setValue(
            float(self.settings.value("hpgign/weight_decay_min", 1e-6))
        )
        self.weight_decay_min.setToolTip("Weight decay minimal. Exemple : 1e-6.")

        self.weight_decay_max = QDoubleSpinBox()
        self.weight_decay_max.setDecimals(10)
        self.weight_decay_max.setRange(1e-10, 1.0)
        self.weight_decay_max.setSingleStep(1e-6)
        self.weight_decay_max.setValue(
            float(self.settings.value("hpgign/weight_decay_max", 1e-3))
        )
        self.weight_decay_max.setToolTip("Weight decay maximal. Exemple : 1e-3.")

        self.drop_out_min = QDoubleSpinBox()
        self.drop_out_min.setDecimals(2)
        self.drop_out_min.setRange(0.0, 1.0)
        self.drop_out_min.setSingleStep(0.05)
        self.drop_out_min.setValue(
            float(self.settings.value("hpgign/drop_out_min", 0.0))
        )

        self.drop_out_max = QDoubleSpinBox()
        self.drop_out_max.setDecimals(2)
        self.drop_out_max.setRange(0.0, 1.0)
        self.drop_out_max.setSingleStep(0.05)
        self.drop_out_max.setValue(
            float(self.settings.value("hpgign/drop_out_max", 0.1))
        )

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(
            int(self.settings.value("hpgign/epochs", 50))
        )

        self.patience = QSpinBox()
        self.patience.setValue(15)
        self.patience.setReadOnly(True)
        self.patience.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.patience.setToolTip("Patience fixe pour l'early stopping : 15.")

        self.hidden_dim_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.hidden_dim_max)
        )

        self.batch_size_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.batch_size_max)
        )

        self.lr_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.lr_max)
        )

        self.weight_decay_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.weight_decay_max)
        )

        self.drop_out_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.drop_out_max)
        )

        main_layout = QVBoxLayout()
        form_layout = QGridLayout()

        row = 0

        form_layout.addWidget(QLabel("Train split:"), row, 0)
        form_layout.addWidget(self.train_split_input, row, 1, 1, 2)
        form_layout.addWidget(self.train_split_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Validation split:"), row, 0)
        form_layout.addWidget(self.val_split_input, row, 1, 1, 2)
        form_layout.addWidget(self.val_split_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Test split:"), row, 0)
        form_layout.addWidget(self.test_split_input, row, 1, 1, 2)
        form_layout.addWidget(self.test_split_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Save directory:"), row, 0)
        form_layout.addWidget(self.save_directory_input, row, 1, 1, 2)
        form_layout.addWidget(self.save_directory_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Graph directory:"), row, 0)
        form_layout.addWidget(self.graph_directory_input, row, 1, 1, 2)
        form_layout.addWidget(self.graph_directory_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("CPU per trial:"), row, 0)
        form_layout.addWidget(self.cpu_per_trials_input, row, 1)

        form_layout.addWidget(QLabel("GPU per trial:"), row, 2)
        form_layout.addWidget(self.gpu_per_trials_input, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Number of trials:"), row, 0)
        form_layout.addWidget(self.number_of_trials, row, 1)

        row += 1
        form_layout.addWidget(QLabel("Node dim:"), row, 0)
        form_layout.addWidget(self.node_dim, row, 1)

        row += 1
        form_layout.addWidget(QLabel("Hidden dim:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.hidden_dim_min, self.hidden_dim_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Batch size:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.batch_size_min, self.batch_size_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Learning rate:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.lr_min, self.lr_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Weight decay:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.weight_decay_min, self.weight_decay_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Dropout:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.drop_out_min, self.drop_out_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Epochs:"), row, 0)
        form_layout.addWidget(self.epochs, row, 1)

        form_layout.addWidget(QLabel("Patience:"), row, 2)
        form_layout.addWidget(self.patience, row, 3)

        main_layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def _min_max_layout(self, min_widget, max_widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Min:"))
        layout.addWidget(min_widget)
        layout.addWidget(QLabel("Max:"))
        layout.addWidget(max_widget)
        return layout

    def _ensure_min_max(self, min_value, max_widget):
        if min_value > max_widget.value():
            max_widget.setValue(min_value)

    def browse_train_split(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select train split file",
            "",
            "Text files (*.txt);;All files (*)",
        )

        if file_path:
            self.train_split_input.setText(file_path)

    def browse_val_split(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select validation split file",
            "",
            "Text files (*.txt);;All files (*)",
        )

        if file_path:
            self.val_split_input.setText(file_path)

    def browse_test_split(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select test split file",
            "",
            "Text files (*.txt);;All files (*)",
        )

        if file_path:
            self.test_split_input.setText(file_path)

    def browse_save_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select save directory",
            "",
        )

        if directory:
            self.save_directory_input.setText(directory)

    def browse_graph_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select graph directory",
            "",
        )

        if directory:
            self.graph_directory_input.setText(directory)

    def accept(self):
        if not self._validate_inputs():
            return

        self.save_settings()
        super().accept()

    def _validate_inputs(self):
        required_fields = [
            (self.train_split_input, "Train split file"),
            (self.val_split_input, "Validation split file"),
            (self.test_split_input, "Test split file"),
            (self.save_directory_input, "Save directory"),
            (self.graph_directory_input, "Graph directory"),
        ]

        for widget, name in required_fields:
            if not widget.text().strip():
                QMessageBox.warning(
                    self,
                    "Missing field",
                    f"{name} is required.",
                )
                return False

        ranges = [
            (
                self.hidden_dim_min.value(),
                self.hidden_dim_max.value(),
                "hidden_dim",
            ),
            (
                self.batch_size_min.value(),
                self.batch_size_max.value(),
                "batch_size",
            ),
            (
                self.lr_min.value(),
                self.lr_max.value(),
                "learning rate",
            ),
            (
                self.weight_decay_min.value(),
                self.weight_decay_max.value(),
                "weight decay",
            ),
            (
                self.drop_out_min.value(),
                self.drop_out_max.value(),
                "dropout",
            ),
        ]

        for min_value, max_value, name in ranges:
            if min_value > max_value:
                QMessageBox.warning(
                    self,
                    "Invalid range",
                    f"{name} min must be <= {name} max.",
                )
                return False

        return True

    def save_settings(self):
        self.settings.setValue(
            "hpgign/train_split_file",
            self.train_split_input.text().strip(),
        )
        self.settings.setValue(
            "hpgign/val_split_file",
            self.val_split_input.text().strip(),
        )
        self.settings.setValue(
            "hpgign/test_split_file",
            self.test_split_input.text().strip(),
        )
        self.settings.setValue(
            "hpgign/save_directory",
            self.save_directory_input.text().strip(),
        )
        self.settings.setValue(
            "hpgign/graph_directory",
            self.graph_directory_input.text().strip(),
        )

        self.settings.setValue(
            "hpgign/cpu_per_trials",
            self.cpu_per_trials_input.value(),
        )
        self.settings.setValue(
            "hpgign/gpu_per_trials",
            self.gpu_per_trials_input.value(),
        )
        self.settings.setValue(
            "hpgign/number_of_trials",
            self.number_of_trials.value(),
        )

        self.settings.setValue("hpgign/hidden_dim_min", self.hidden_dim_min.value())
        self.settings.setValue("hpgign/hidden_dim_max", self.hidden_dim_max.value())

        self.settings.setValue("hpgign/batch_size_min", self.batch_size_min.value())
        self.settings.setValue("hpgign/batch_size_max", self.batch_size_max.value())

        self.settings.setValue("hpgign/lr_min", self.lr_min.value())
        self.settings.setValue("hpgign/lr_max", self.lr_max.value())

        self.settings.setValue(
            "hpgign/weight_decay_min",
            self.weight_decay_min.value(),
        )
        self.settings.setValue(
            "hpgign/weight_decay_max",
            self.weight_decay_max.value(),
        )

        self.settings.setValue("hpgign/drop_out_min", self.drop_out_min.value())
        self.settings.setValue("hpgign/drop_out_max", self.drop_out_max.value())

        self.settings.setValue("hpgign/epochs", self.epochs.value())

    def get_inputs(self):
        return {
            "train_file": self.train_split_input.text().strip(),
            "val_file": self.val_split_input.text().strip(),
            "test_file": self.test_split_input.text().strip(),
            "save_directory": self.save_directory_input.text().strip(),
            "graph_directory": self.graph_directory_input.text().strip(),

            "cpu_per_trials": self.cpu_per_trials_input.value(),
            "gpu_per_trials": self.gpu_per_trials_input.value(),
            "number_of_trials": self.number_of_trials.value(),

            "NODE_DIM": self.node_dim.value(),

            "hidden_dim_min": self.hidden_dim_min.value(),
            "hidden_dim_max": self.hidden_dim_max.value(),

            "batch_size_min": self.batch_size_min.value(),
            "batch_size_max": self.batch_size_max.value(),

            "lr_min": self.lr_min.value(),
            "lr_max": self.lr_max.value(),

            "weight_decay_min": self.weight_decay_min.value(),
            "weight_decay_max": self.weight_decay_max.value(),

            "drop_out_min": self.drop_out_min.value(),
            "drop_out_max": self.drop_out_max.value(),

            "EPOCHS": self.epochs.value(),
            "PATIENCE": self.patience.value(),
        }