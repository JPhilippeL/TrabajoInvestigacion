import logging
from pathlib import Path

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
)

from job_config.cheapnet.CheapnetPredictionConfig import CheapnetPredictionConfig

logger = logging.getLogger(__name__)


class PredictDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Predict CheapNet")
        self.resize(700, 560)

        self.settings = QSettings("Investigacion", "Predict_CheapNet")

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("Test split file (.txt)")
        self.test_split_input.setText(
            self.settings.value("predict_cheapnet/last_test_split_file", "")
        )
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.model_dir_input = QLineEdit()
        self.model_dir_input.setPlaceholderText(
            "Directory containing split_xx/best_model.pt"
        )
        self.model_dir_input.setText(
            self.settings.value("predict_cheapnet/last_model_dir", "")
        )
        self.model_dir_btn = QPushButton("Select...")
        self.model_dir_btn.clicked.connect(self.browse_model_dir)

        self.graph_dir_input = QLineEdit()
        self.graph_dir_input.setPlaceholderText(
            "Directory containing graph .pt files"
        )
        self.graph_dir_input.setText(
            self.settings.value("predict_cheapnet/last_graph_dir", "")
        )
        self.graph_dir_btn = QPushButton("Select...")
        self.graph_dir_btn.clicked.connect(self.browse_graph_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText(
            "Directory to save prediction outputs"
        )
        self.output_dir_input.setText(
            self.settings.value("predict_cheapnet/last_output_dir", "")
        )
        self.output_dir_btn = QPushButton("Select...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("File pIC50.txt")
        self.pic50_file_input.setText(
            self.settings.value("predict_cheapnet/last_pic50_file", "")
        )
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_pic50_file)

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
            "pIC50 file:",
            self._with_button(self.pic50_file_input, self.pic50_btn),
        )

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
            self,
            title,
            "",
            "Text files (*.txt);;All files (*)",
        )
        return path

    def _browse_dir(self, title):
        return QFileDialog.getExistingDirectory(self, title)

    def browse_test_split(self):
        path = self._browse_txt_file("Select test split file")

        if path:
            self.test_split_input.setText(path)

    def browse_pic50_file(self):
        path = self._browse_txt_file("Select pIC50 file")

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

        if not self.pic50_file_input.text().strip():
            logger.warning("Missing input: please select the pIC50 file.")
            return

        self.settings.setValue(
            "predict_cheapnet/last_test_split_file",
            self.test_split_input.text().strip(),
        )

        self.settings.setValue(
            "predict_cheapnet/last_model_dir",
            self.model_dir_input.text().strip(),
        )

        self.settings.setValue(
            "predict_cheapnet/last_graph_dir",
            self.graph_dir_input.text().strip(),
        )

        self.settings.setValue(
            "predict_cheapnet/last_output_dir",
            self.output_dir_input.text().strip(),
        )

        self.settings.setValue(
            "predict_cheapnet/last_pic50_file",
            self.pic50_file_input.text().strip(),
        )

        super().accept()

    def get_inputs(self) -> CheapnetPredictionConfig:
        return CheapnetPredictionConfig(
            pic50_path=Path(self.pic50_file_input.text().strip()),
            test_split_file=Path(self.test_split_input.text().strip()),
            graphs_path=Path(self.graph_dir_input.text().strip()),
            model_path=Path(self.model_dir_input.text().strip()),
            output_path=Path(self.output_dir_input.text().strip()),
        )