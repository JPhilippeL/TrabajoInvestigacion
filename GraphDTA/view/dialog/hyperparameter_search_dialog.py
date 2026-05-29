import logging

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
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


class GraphDTAHyperparameterSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("GraphDTA Hyperparameter Tuning")
        self.resize(700, 500)

        self.settings = QSettings("Investigacion", "GraphDTAHyperparameterSearch")

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("Train split file (.txt)")
        self.train_split_input.setText(
            str(self.settings.value("hpgraphdta/train_split_file", ""))
        )

        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("Validation split file (.txt)")
        self.val_split_input.setText(
            str(self.settings.value("hpgraphdta/val_split_file", ""))
        )

        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("Test split file (.txt)")
        self.test_split_input.setText(
            str(self.settings.value("hpgraphdta/test_split_file", ""))
        )

        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.save_directory_input = QLineEdit()
        self.save_directory_input.setPlaceholderText("Save directory")
        self.save_directory_input.setText(
            str(self.settings.value("hpgraphdta/save_directory", ""))
        )

        self.save_directory_btn = QPushButton("Select...")
        self.save_directory_btn.clicked.connect(self.browse_save_directory)

        self.graph_directory_input = QLineEdit()
        self.graph_directory_input.setPlaceholderText("Graph directory")
        self.graph_directory_input.setText(
            str(self.settings.value("hpgraphdta/graph_directory", ""))
        )

        self.graph_directory_btn = QPushButton("Select...")
        self.graph_directory_btn.clicked.connect(self.browse_graph_directory)

        self.model_name_input = QComboBox()
        self.model_name_input.addItems(
            [
                "GINConvNet",
                "GAT",
                "GCN",
                "GAT_GCN",
            ]
        )

        saved_model_name = str(
            self.settings.value("hpgraphdta/model_name", "GINConvNet")
        )
        index = self.model_name_input.findText(saved_model_name)
        if index >= 0:
            self.model_name_input.setCurrentIndex(index)

        self.cpu_per_trials_input = QSpinBox()
        self.cpu_per_trials_input.setRange(1, 100)
        self.cpu_per_trials_input.setValue(
            int(self.settings.value("hpgraphdta/cpu_per_trials", 5))
        )
        self.cpu_per_trials_input.setToolTip(
            "Number of CPUs used for one trial."
        )

        self.gpu_per_trials_input = QDoubleSpinBox()
        self.gpu_per_trials_input.setDecimals(2)
        self.gpu_per_trials_input.setRange(0.0, 100.0)
        self.gpu_per_trials_input.setSingleStep(0.25)
        self.gpu_per_trials_input.setValue(
            float(self.settings.value("hpgraphdta/gpu_per_trials", 0.0))
        )
        self.gpu_per_trials_input.setToolTip(
            "Number of GPUs used for one trial. Example: 0, 0.5, 1."
        )

        self.number_of_trials = QSpinBox()
        self.number_of_trials.setRange(1, 1000)
        self.number_of_trials.setValue(
            int(self.settings.value("hpgraphdta/number_of_trials", 40))
        )
        self.number_of_trials.setToolTip(
            "Number of random hyperparameter combinations to test."
        )

        self.n_filters_min = QSpinBox()
        self.n_filters_min.setRange(8, 1024)
        self.n_filters_min.setSingleStep(8)
        self.n_filters_min.setValue(
            int(self.settings.value("hpgraphdta/n_filters_min", 16))
        )

        self.n_filters_max = QSpinBox()
        self.n_filters_max.setRange(8, 1024)
        self.n_filters_max.setSingleStep(8)
        self.n_filters_max.setValue(
            int(self.settings.value("hpgraphdta/n_filters_max", 128))
        )

        self.batch_size_min = QSpinBox()
        self.batch_size_min.setRange(1, 512)
        self.batch_size_min.setValue(
            int(self.settings.value("hpgraphdta/batch_size_min", 4))
        )

        self.batch_size_max = QSpinBox()
        self.batch_size_max.setRange(1, 512)
        self.batch_size_max.setValue(
            int(self.settings.value("hpgraphdta/batch_size_max", 64))
        )

        self.lr_min = QDoubleSpinBox()
        self.lr_min.setDecimals(8)
        self.lr_min.setRange(1e-8, 1.0)
        self.lr_min.setSingleStep(1e-4)
        self.lr_min.setValue(
            float(self.settings.value("hpgraphdta/lr_min", 1e-4))
        )
        self.lr_min.setToolTip("Learning rate minimum. Example: 1e-4.")

        self.lr_max = QDoubleSpinBox()
        self.lr_max.setDecimals(8)
        self.lr_max.setRange(1e-8, 1.0)
        self.lr_max.setSingleStep(1e-4)
        self.lr_max.setValue(
            float(self.settings.value("hpgraphdta/lr_max", 1e-2))
        )
        self.lr_max.setToolTip("Learning rate maximum. Example: 1e-2.")

        self.weight_decay_min = QDoubleSpinBox()
        self.weight_decay_min.setDecimals(10)
        self.weight_decay_min.setRange(1e-10, 1.0)
        self.weight_decay_min.setSingleStep(1e-6)
        self.weight_decay_min.setValue(
            float(self.settings.value("hpgraphdta/weight_decay_min", 1e-5))
        )
        self.weight_decay_min.setToolTip("Weight decay minimum. Example: 1e-5.")

        self.weight_decay_max = QDoubleSpinBox()
        self.weight_decay_max.setDecimals(10)
        self.weight_decay_max.setRange(1e-10, 1.0)
        self.weight_decay_max.setSingleStep(1e-6)
        self.weight_decay_max.setValue(
            float(self.settings.value("hpgraphdta/weight_decay_max", 1e-3))
        )
        self.weight_decay_max.setToolTip("Weight decay maximum. Example: 1e-3.")

        self.dropout_min = QDoubleSpinBox()
        self.dropout_min.setDecimals(2)
        self.dropout_min.setRange(0.0, 1.0)
        self.dropout_min.setSingleStep(0.05)
        self.dropout_min.setValue(
            float(self.settings.value("hpgraphdta/dropout_min", 0.0))
        )

        self.dropout_max = QDoubleSpinBox()
        self.dropout_max.setDecimals(2)
        self.dropout_max.setRange(0.0, 1.0)
        self.dropout_max.setSingleStep(0.05)
        self.dropout_max.setValue(
            float(self.settings.value("hpgraphdta/dropout_max", 0.1))
        )

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(
            int(self.settings.value("hpgraphdta/epochs", 50))
        )

        self.patience = QSpinBox()
        self.patience.setValue(15)
        self.patience.setReadOnly(True)
        self.patience.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.patience.setToolTip(
            "Patience is fixed to 15 in train_dta."
        )

        self.n_filters_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.n_filters_max)
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

        self.dropout_min.valueChanged.connect(
            lambda value: self._ensure_min_max(value, self.dropout_max)
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
        form_layout.addWidget(QLabel("Model:"), row, 0)
        form_layout.addWidget(self.model_name_input, row, 1, 1, 3)

        row += 1
        form_layout.addWidget(QLabel("CPU per trial:"), row, 0)
        form_layout.addWidget(self.cpu_per_trials_input, row, 1)

        form_layout.addWidget(QLabel("GPU per trial:"), row, 2)
        form_layout.addWidget(self.gpu_per_trials_input, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Number of trials:"), row, 0)
        form_layout.addWidget(self.number_of_trials, row, 1)

        row += 1
        form_layout.addWidget(QLabel("number of filters:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.n_filters_min, self.n_filters_max),
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
            self._min_max_layout(self.dropout_min, self.dropout_max),
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
                self.n_filters_min.value(),
                self.n_filters_max.value(),
                "n_filters",
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
                self.dropout_min.value(),
                self.dropout_max.value(),
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
            "hpgraphdta/train_split_file",
            self.train_split_input.text().strip(),
        )
        self.settings.setValue(
            "hpgraphdta/val_split_file",
            self.val_split_input.text().strip(),
        )
        self.settings.setValue(
            "hpgraphdta/test_split_file",
            self.test_split_input.text().strip(),
        )
        self.settings.setValue(
            "hpgraphdta/save_directory",
            self.save_directory_input.text().strip(),
        )
        self.settings.setValue(
            "hpgraphdta/graph_directory",
            self.graph_directory_input.text().strip(),
        )

        self.settings.setValue(
            "hpgraphdta/model_name",
            self.model_name_input.currentText(),
        )

        self.settings.setValue(
            "hpgraphdta/cpu_per_trials",
            self.cpu_per_trials_input.value(),
        )
        self.settings.setValue(
            "hpgraphdta/gpu_per_trials",
            self.gpu_per_trials_input.value(),
        )
        self.settings.setValue(
            "hpgraphdta/number_of_trials",
            self.number_of_trials.value(),
        )

        self.settings.setValue(
            "hpgraphdta/n_filters_min",
            self.n_filters_min.value(),
        )
        self.settings.setValue(
            "hpgraphdta/n_filters_max",
            self.n_filters_max.value(),
        )

        self.settings.setValue(
            "hpgraphdta/batch_size_min",
            self.batch_size_min.value(),
        )
        self.settings.setValue(
            "hpgraphdta/batch_size_max",
            self.batch_size_max.value(),
        )

        self.settings.setValue("hpgraphdta/lr_min", self.lr_min.value())
        self.settings.setValue("hpgraphdta/lr_max", self.lr_max.value())

        self.settings.setValue(
            "hpgraphdta/weight_decay_min",
            self.weight_decay_min.value(),
        )
        self.settings.setValue(
            "hpgraphdta/weight_decay_max",
            self.weight_decay_max.value(),
        )

        self.settings.setValue(
            "hpgraphdta/dropout_min",
            self.dropout_min.value(),
        )
        self.settings.setValue(
            "hpgraphdta/dropout_max",
            self.dropout_max.value(),
        )

        self.settings.setValue("hpgraphdta/epochs", self.epochs.value())

    def get_values(self):
        return {
            "train_split_file": self.train_split_input.text().strip(),
            "val_split_file": self.val_split_input.text().strip(),
            "test_split_file": self.test_split_input.text().strip(),
            "output_dir": self.save_directory_input.text().strip(),
            "graph_directory": self.graph_directory_input.text().strip(),

            "model_name": self.model_name_input.currentText(),

            "cpu_per_trial": self.cpu_per_trials_input.value(),
            "gpu_per_trial": self.gpu_per_trials_input.value(),
            "num_samples": self.number_of_trials.value(),

            "n_filters_min": self.n_filters_min.value(),
            "n_filters_max": self.n_filters_max.value(),

            "batch_size_min": self.batch_size_min.value(),
            "batch_size_max": self.batch_size_max.value(),

            "lr_min": self.lr_min.value(),
            "lr_max": self.lr_max.value(),

            "weight_decay_min": self.weight_decay_min.value(),
            "weight_decay_max": self.weight_decay_max.value(),

            "dropout_min": self.dropout_min.value(),
            "dropout_max": self.dropout_max.value(),

            "EPOCHS": self.epochs.value(),
            "PATIENCE": self.patience.value(),
        }
