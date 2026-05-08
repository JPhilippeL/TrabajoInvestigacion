import logging

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QDoubleSpinBox
)

logger = logging.getLogger(__name__)


class PredictDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Predict CheapNet")
        self.resize(700, 560)

        self.settings = QSettings("Investigacion", "Predict_GIGN")

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("Test split file (.txt)")
        self.test_split_input.setText(
            self.settings.value("train_cheapnet/last_test_file", "")
        )
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.model_dir_input = QLineEdit()
        self.model_dir_input.setPlaceholderText(
            "Directory containing split_xx/best_model.pt"
        )
        self.model_dir_input.setText(
            self.settings.value("predict/last_model_dir", "")
        )
        self.model_dir_btn = QPushButton("Select...")
        self.model_dir_btn.clicked.connect(self.browse_model_dir)

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("File pic50.txt")
        self.pic50_file_input.setText(
            self.settings.value("dbGen/last_pic50_file", "")
        )
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_pic50_file)
        self.graph_dir_input = QLineEdit()
        self.graph_dir_input.setPlaceholderText(
            "Directory containing graph .pt files"
        )
        self.graph_dir_input.setText(
            self.settings.value("train_cheapnet/last_graph_dir", "")
        )
        self.graph_dir_btn = QPushButton("Select...")
        self.graph_dir_btn.clicked.connect(self.browse_graph_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText(
            "Directory to save prediction outputs"
        )
        self.output_dir_input.setText(
            self.settings.value("predict/last_output_dir", "")
        )
        self.output_dir_btn = QPushButton("Select...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.parameter_file_input = QLineEdit()
        self.parameter_file_input.setPlaceholderText(
            "File with hyparameters used for training"
        )
        self.drop_rate_input = QDoubleSpinBox()
        self.drop_rate_input.setValue(self.settings.value("train_cheapnet/drop_rate", 0))
        self.drop_rate_input.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.drop_rate_input.setReadOnly(True)

        self.node_dim = QSpinBox()
        self.drop_rate_input.setValue(14)
        self.drop_rate_input.setReadOnly(True)
        self.drop_rate_input.setButtonSymbols(QSpinBox.NoButtons)

        self.hidden_dim = QSpinBox()
        self.hidden_dim.setValue(self.settings.value("train_cheapnet/hidden_dim", 0))
        self.hidden_dim.setReadOnly(True)
        self.hidden_dim.setButtonSymbols(QSpinBox.NoButtons)

        self.batch_size = QSpinBox()
        self.batch_size.setValue(self.settings.value("train_cheapnet/batch_size", 0))
        self.batch_size.setReadOnly(True)
        self.batch_size.setButtonSymbols(QSpinBox.NoButtons)
        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>Required files/directories</b>"))

        form_layout.addRow(
            "Test split:",
            self._with_button(self.test_split_input, self.test_split_btn),
        )

        form_layout.addRow(
            "Model dir:",
            self._with_button(self.model_dir_input, self.model_dir_btn),
        )
        form_layout.addRow(
            "Graph dir:",
            self._with_button(self.graph_dir_input, self.graph_dir_btn),
        )
        form_layout.addRow(
            "Output dir:",
            self._with_button(self.output_dir_input, self.output_dir_btn),
        )
        form_layout.addRow(
            "Target file:",
            self._with_button(self.pic50_file_input, self.pic50_btn),
        )
        form_layout.addRow(QLabel("<b>Training hyperparameters (from last train)</b>"))
        form_layout.addRow("Dropout rate:", self.drop_rate_input)
        form_layout.addRow("Node dimension:", self.node_dim)
        form_layout.addRow("Hidden dimension:", self.hidden_dim)
        form_layout.addRow("Batch size:", self.batch_size)
        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
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
            self, title, "", "Text files (*.txt);;All files (*)"
        )
        return path

    def _browse_dir(self, title):
        return QFileDialog.getExistingDirectory(self, title)

    def browse_test_split(self):
        path = self._browse_txt_file("Select test split file")
        if path:
            self.test_split_input.setText(path)

    def browse_pic50_file(self):
        path = self._browse_txt_file("Select pic50 file")
        if path:
            self.pic50_file_input.setText(path)

    def browse_model_dir(self):
        path = self._browse_dir("Select model directory")
        if path:
            self.model_dir_input.setText(path)

    def browse_graph_dir(self):
        path = self._browse_dir("Select graph directory")
        if path:
            self.graph_dir_input.setText(path)

    def browse_output_dir(self):
        path = self._browse_dir("Select output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):

        if not self.test_split_input.text().strip():
            logger.warning("Missing input: please select the test split file.")
            return

        if not self.model_dir_input.text().strip():
            logger.warning("Missing input: please select the model directory.")
            return

        if not self.graph_dir_input.text().strip():
            logger.warning("Missing input: please select the graph directory.")
            return

        if not self.output_dir_input.text().strip():
            logger.warning("Missing input: please select the output directory.")
            return

        self.settings.setValue(
            "train_cheapnet/last_test_file", self.test_split_input.text().strip()
        )

        self.settings.setValue(
            "predict/last_model_dir", self.model_dir_input.text().strip()
        )
        self.settings.setValue(
            "train_cheapnet/last_graph_dir", self.graph_dir_input.text().strip()
        )
        self.settings.setValue(
            "predict/last_output_dir", self.output_dir_input.text().strip()
        )
        self.settings.setValue(
            "dbGen/last_pic50_file", self.pic50_file_input.text().strip()
        )

        super().accept()

    def get_inputs(self):
        return {
            "model_dir": self.model_dir_input.text().strip(),
            "graph_dir": self.graph_dir_input.text().strip(),
            "test_split_file": self.test_split_input.text().strip(),
            "output_dir": self.output_dir_input.text().strip(),
            "pic50_txt": self.pic50_file_input.text().strip(),
            "node_dim": self.node_dim.value(),
            "batch_size": self.batch_size.value(),
            "hidden_dim": self.hidden_dim.value(),
            "drop_rate": self.drop_rate_input.value(),
        }
