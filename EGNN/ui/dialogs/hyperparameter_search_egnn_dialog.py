"""
@file hyperparameter_search_egnn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search configuration dialog for the EGNN module.
@details
This dialog allows the user to configure the parameters required to launch
EGNN hyperparameter search from the graphical interface.

The dialog provides fields for:
- dataset directory,
- graphs directory,
- models directory,
- results directory,
- temporary runs directory,
- learning rate values,
- hidden dimension values,
- batch size values.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QDialogButtonBox,
    QWidget,
    QHBoxLayout,
    QLabel,
)
from PySide6.QtCore import QSettings


class HyperparameterSearchEGNNDialog(QDialog):
    """
    @brief Dialog used to configure EGNN hyperparameter search.
    """

    def __init__(self, parent=None):
        """
        @brief Initialize the EGNN hyperparameter search dialog.
        @param parent Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Configuración de Hyperparameter Search - EGNN")
        self.resize(700, 420)

        self.settings = QSettings("Investigacion", "EGNN_HyperparameterSearch")

        # ============================================================
        # PATHS
        # ============================================================
        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Directorio de grafos Graphs_EGNN...")
        self.graphs_dir_input.setText(self.settings.value("search/graphs_dir", ""))
        self.graphs_dir_btn = QPushButton("Seleccionar...")
        self.graphs_dir_btn.clicked.connect(self.browse_graphs_dir)

        self.dataset_dir_input = QLineEdit()
        self.dataset_dir_input.setPlaceholderText("Directorio del dataset MPro-URV_Version2...")
        self.dataset_dir_input.setText(self.settings.value("search/dataset_dir", ""))
        self.dataset_dir_btn = QPushButton("Seleccionar...")
        self.dataset_dir_btn.clicked.connect(self.browse_dataset_dir)

        self.models_dir_input = QLineEdit()
        self.models_dir_input.setPlaceholderText("Directorio de salida de modelos...")
        self.models_dir_input.setText(self.settings.value("search/models_dir", ""))
        self.models_dir_btn = QPushButton("Seleccionar...")
        self.models_dir_btn.clicked.connect(self.browse_models_dir)

        self.results_dir_input = QLineEdit()
        self.results_dir_input.setPlaceholderText("Directorio de resultados...")
        self.results_dir_input.setText(self.settings.value("search/results_dir", ""))
        self.results_dir_btn = QPushButton("Seleccionar...")
        self.results_dir_btn.clicked.connect(self.browse_results_dir)

        self.temp_runs_dir_input = QLineEdit()
        self.temp_runs_dir_input.setPlaceholderText("Directorio temporal de trials...")
        self.temp_runs_dir_input.setText(self.settings.value("search/temp_runs_dir", ""))
        self.temp_runs_dir_btn = QPushButton("Seleccionar...")
        self.temp_runs_dir_btn.clicked.connect(self.browse_temp_runs_dir)

        # ============================================================
        # SEARCH SPACE
        # ============================================================
        self.lr_values_input = QLineEdit()
        self.lr_values_input.setPlaceholderText("Ejemplo: 5e-5,1e-4,5e-4,1e-3")
        self.lr_values_input.setText(self.settings.value("search/lr_values", "5e-5,1e-4,5e-4,1e-3"))

        self.hidden_dim_values_input = QLineEdit()
        self.hidden_dim_values_input.setPlaceholderText("Ejemplo: 32,64,128")
        self.hidden_dim_values_input.setText(self.settings.value("search/hidden_dim_values", "32,64,128"))

        self.batch_size_values_input = QLineEdit()
        self.batch_size_values_input.setPlaceholderText("Ejemplo: 2,4,8")
        self.batch_size_values_input.setText(self.settings.value("search/batch_size_values", "2,4,8"))

        # ============================================================
        # LAYOUT
        # ============================================================
        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>1. Directorios</b>"))
        form_layout.addRow("Graphs Directory:", self._with_button(self.graphs_dir_input, self.graphs_dir_btn))
        form_layout.addRow("Dataset Directory:", self._with_button(self.dataset_dir_input, self.dataset_dir_btn))
        form_layout.addRow("Models Directory:", self._with_button(self.models_dir_input, self.models_dir_btn))
        form_layout.addRow("Results Directory:", self._with_button(self.results_dir_input, self.results_dir_btn))
        form_layout.addRow("Temp Runs Directory:", self._with_button(self.temp_runs_dir_input, self.temp_runs_dir_btn))

        form_layout.addRow(QLabel("<br><b>2. Search Space</b>"))
        form_layout.addRow("Learning Rate Values:", self.lr_values_input)
        form_layout.addRow("Hidden Dimension Values:", self.hidden_dim_values_input)
        form_layout.addRow("Batch Size Values:", self.batch_size_values_input)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def _with_button(self, line_edit, button):
        """
        @brief Create a horizontal widget combining a line edit and a button.
        @param line_edit Line edit widget.
        @param button Button widget.
        @return QWidget containing both widgets.
        """
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def browse_graphs_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de grafos")
        if path:
            self.graphs_dir_input.setText(path)

    def browse_dataset_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta del dataset")
        if path:
            self.dataset_dir_input.setText(path)

    def browse_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de modelos")
        if path:
            self.models_dir_input.setText(path)

    def browse_results_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de resultados")
        if path:
            self.results_dir_input.setText(path)

    def browse_temp_runs_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta temporal de trials")
        if path:
            self.temp_runs_dir_input.setText(path)

    def accept(self):
        """
        @brief Save the current dialog values before closing with success.
        @return None
        """
        self.settings.setValue("search/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("search/dataset_dir", self.dataset_dir_input.text())
        self.settings.setValue("search/models_dir", self.models_dir_input.text())
        self.settings.setValue("search/results_dir", self.results_dir_input.text())
        self.settings.setValue("search/temp_runs_dir", self.temp_runs_dir_input.text())
        self.settings.setValue("search/lr_values", self.lr_values_input.text())
        self.settings.setValue("search/hidden_dim_values", self.hidden_dim_values_input.text())
        self.settings.setValue("search/batch_size_values", self.batch_size_values_input.text())

        super().accept()

    @staticmethod
    def _parse_float_list(raw_text: str):
        """
        @brief Parse a comma-separated list of float values.
        @param raw_text Input string.
        @return List of float values.
        """
        return [float(x.strip()) for x in raw_text.split(",") if x.strip()]

    @staticmethod
    def _parse_int_list(raw_text: str):
        """
        @brief Parse a comma-separated list of integer values.
        @param raw_text Input string.
        @return List of integer values.
        """
        return [int(x.strip()) for x in raw_text.split(",") if x.strip()]

    def get_inputs(self):
        """
        @brief Return the hyperparameter search configuration entered by the user.
        @return Dictionary of EGNN hyperparameter search parameters.
        """
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "dataset_dir": self.dataset_dir_input.text(),
            "models_dir": self.models_dir_input.text(),
            "results_dir": self.results_dir_input.text(),
            "temp_runs_dir": self.temp_runs_dir_input.text(),
            "lr_values": self._parse_float_list(self.lr_values_input.text()),
            "hidden_dim_values": self._parse_int_list(self.hidden_dim_values_input.text()),
            "batch_size_values": self._parse_int_list(self.batch_size_values_input.text()),
        }
