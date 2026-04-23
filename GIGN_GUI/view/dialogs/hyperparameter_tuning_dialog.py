from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget,
    QHBoxLayout, QLabel, QSpinBox
)
from PySide6.QtCore import QSettings
import logging

logger = logging.getLogger(__name__)


class HyperParameterTuningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hyperparameter Tuning")
        self.resize(700, 560)

        self.settings = QSettings("Investigacion", "Hyperparameter_Tuning")

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("Train split file (.txt)")
        self.train_split_input.setText(self.settings.value("tuning/last_train_split_file", ""))
        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("Test split file (.txt)")
        self.test_split_input.setText(self.settings.value("tuning/last_test_split_file", ""))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("Validation split file (.txt)")
        self.val_split_input.setText(self.settings.value("tuning/last_val_split_file", ""))
        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.graph_dir_input = QLineEdit()
        self.graph_dir_input.setPlaceholderText("Directory containing graph .pt files")
        self.graph_dir_input.setText(self.settings.value("tuning/last_graph_dir", ""))
        self.graph_dir_btn = QPushButton("Select...")
        self.graph_dir_btn.clicked.connect(self.browse_graph_dir)

        self.out_dir_input = QLineEdit()
        self.out_dir_input.setPlaceholderText("Directory to save hyperparameter tuning outputs")
        self.out_dir_input.setText(self.settings.value("tuning/last_out_dir", ""))
        self.out_dir_btn = QPushButton("Select...")
        self.out_dir_btn.clicked.connect(self.browse_out_dir)

        self.cpu_per_trials_input = QSpinBox()
        self.cpu_per_trials_input.setRange(1, 128)
        self.cpu_per_trials_input.setValue(int(self.settings.value("tuning/cpu_per_trials", 6)))

        self.gpu_per_trials_input = QSpinBox()
        self.gpu_per_trials_input.setRange(0, 8)
        self.gpu_per_trials_input.setValue(int(self.settings.value("tuning/gpu_per_trials", 1)))

        self.num_trials_input = QSpinBox()
        self.num_trials_input.setRange(1, 10000)
        self.num_trials_input.setValue(int(self.settings.value("tuning/num_trials", 10)))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>Required files/directories</b>"))
        form_layout.addRow("Train split:", self._with_button(self.train_split_input, self.train_split_btn))
        form_layout.addRow("Test split:", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Validation split:", self._with_button(self.val_split_input, self.val_split_btn))
        form_layout.addRow("Graph dir:", self._with_button(self.graph_dir_input, self.graph_dir_btn))
        form_layout.addRow("Output dir:", self._with_button(self.out_dir_input, self.out_dir_btn))

        form_layout.addRow(QLabel("<b>Parameters</b>"))
        form_layout.addRow("CPU per trial:", self.cpu_per_trials_input)
        form_layout.addRow("GPU per trial:", self.gpu_per_trials_input)
        form_layout.addRow("Number of trials:", self.num_trials_input)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def _with_button(self, line_edit, button):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def _browse_txt_file(self, title):
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Text files (*.txt);;All files (*)"
        )
        return path

    def _browse_dir(self, title):
        return QFileDialog.getExistingDirectory(self, title)

    def browse_train_split(self):
        path = self._browse_txt_file("Select train split file")
        if path:
            self.train_split_input.setText(path)

    def browse_test_split(self):
        path = self._browse_txt_file("Select test split file")
        if path:
            self.test_split_input.setText(path)

    def browse_val_split(self):
        path = self._browse_txt_file("Select validation split file")
        if path:
            self.val_split_input.setText(path)

    def browse_graph_dir(self):
        path = self._browse_dir("Select graph directory")
        if path:
            self.graph_dir_input.setText(path)

    def browse_out_dir(self):
        path = self._browse_dir("Select output directory")
        if path:
            self.out_dir_input.setText(path)

    def accept(self):
        if not self.train_split_input.text().strip():
            logger.warning("Missing input: please select the train split file.")
            return

        if not self.test_split_input.text().strip():
            logger.warning("Missing input: please select the test split file.")
            return

        if not self.val_split_input.text().strip():
            logger.warning("Missing input: please select the validation split file.")
            return

        if not self.graph_dir_input.text().strip():
            logger.warning("Missing input: please select the graph directory.")
            return

        if not self.out_dir_input.text().strip():
            logger.warning("Missing input: please select the output directory.")
            return

        self.settings.setValue("tuning/last_train_split_file", self.train_split_input.text().strip())
        self.settings.setValue("tuning/last_test_split_file", self.test_split_input.text().strip())
        self.settings.setValue("tuning/last_val_split_file", self.val_split_input.text().strip())
        self.settings.setValue("tuning/last_graph_dir", self.graph_dir_input.text().strip())
        self.settings.setValue("tuning/last_out_dir", self.out_dir_input.text().strip())

        self.settings.setValue("tuning/cpu_per_trials", self.cpu_per_trials_input.value())
        self.settings.setValue("tuning/gpu_per_trials", self.gpu_per_trials_input.value())
        self.settings.setValue("tuning/num_trials", self.num_trials_input.value())

        super().accept()

    def get_inputs(self):
        return {
            "cpu_per_trials": self.cpu_per_trials_input.value(),
            "gpu_per_trials": self.gpu_per_trials_input.value(),
            "num_trials": self.num_trials_input.value(),
            "out_dir": self.out_dir_input.text().strip(),
            "train_split_file": self.train_split_input.text().strip(),
            "val_split_file": self.val_split_input.text().strip(),
            "test_split_file": self.test_split_input.text().strip(),
            "graph_dir": self.graph_dir_input.text().strip(),
        }
