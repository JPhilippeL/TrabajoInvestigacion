from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLabel
)
from PySide6.QtCore import QSettings

class DBGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generación de Base de Datos")
        self.resize(600, 400)

        # ---------- QSettings ----------
        # Usamos una categoría distinta para no mezclar con el anterior
        self.settings = QSettings("Investigacion", "DB_Generation")

        # 1. ---------- Required: DSSP Directory ----------
        self.dssp_dir_input = QLineEdit()
        self.dssp_dir_input.setPlaceholderText("Carpeta con archivos .dssp")
        self.dssp_dir_input.setText(self.settings.value("dbGen/last_dssp_dir", ""))
        self.dssp_btn = QPushButton("Seleccionar...")
        self.dssp_btn.clicked.connect(self.browse_dssp)

        # 2. ---------- Required: Ligand Directory ----------
        self.ligand_dir_input = QLineEdit()
        self.ligand_dir_input.setPlaceholderText("Carpeta con archivos .smi")
        self.ligand_dir_input.setText(self.settings.value("dbGen/last_ligand_dir", ""))
        self.ligand_btn = QPushButton("Seleccionar...")
        self.ligand_btn.clicked.connect(self.browse_ligand)

        # 3. ---------- Required: Pocket File ----------
        self.pocket_input = QLineEdit()
        self.pocket_input.setPlaceholderText("Archivo CSV de pockets")
        self.pocket_input.setText(self.settings.value("dbGen/last_pocket_file", ""))
        self.pocket_btn = QPushButton("Seleccionar...")
        self.pocket_btn.clicked.connect(self.browse_pocket)

        # 4. ---------- Optional: Output Directory ----------
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Opcional (Default: PROCESSED_DATA_DIR)")
        self.output_dir_input.setText(self.settings.value("dbGen/last_output_dir", ""))
        self.output_btn = QPushButton("Seleccionar...")
        self.output_btn.clicked.connect(self.browse_output)

        # 5. ---------- Ratios (SpinBoxes) ----------
        # Train
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(float(self.settings.value("dbGen/train_ratio", 0.70)))

        # Val
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 1.0)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(float(self.settings.value("dbGen/val_ratio", 0.15)))

        # Test
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(float(self.settings.value("dbGen/test_ratio", 0.15)))

        # 6. ---------- Random Seed & Cleanup ----------
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("dbGen/random_seed", 42)))

        self.cleanup_check = QCheckBox("Limpiar archivos procesados temporales")
        # Recuperar estado booleano (truco: QSettings guarda bool como str a veces, mejor convertir)
        is_checked = self.settings.value("dbGen/cleanup", "true") == "true"
        self.cleanup_check.setChecked(is_checked)

        # ---------- Layout Principal ----------
        form_layout = QFormLayout() # Nota: Corregí QFormLayout a QFormLayout
        
        form_layout.addRow(QLabel("<b>Parámetros Requeridos:</b>"))
        form_layout.addRow("Dir. DSSP:", self._with_button(self.dssp_dir_input, self.dssp_btn))
        form_layout.addRow("Dir. Ligandos:", self._with_button(self.ligand_dir_input, self.ligand_btn))
        form_layout.addRow("Archivo Pocket:", self._with_button(self.pocket_input, self.pocket_btn))
        
        form_layout.addRow(QLabel("<b>Opcionales:</b>"))
        form_layout.addRow("Output Dir:", self._with_button(self.output_dir_input, self.output_btn))
        
        # Agrupar ratios visualmente si quieres, o ponerlos en fila
        form_layout.addRow("Train Ratio:", self.train_spin)
        form_layout.addRow("Validation Ratio:", self.val_spin)
        form_layout.addRow("Test Ratio:", self.test_spin)
        
        form_layout.addRow("Random Seed:", self.seed_spin)
        form_layout.addRow("Cleanup:", self.cleanup_check)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        # ---------- Buttons ----------
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

    # ---------- Browse Methods ----------
    def browse_dssp(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio DSSP")
        if path: self.dssp_dir_input.setText(path)

    def browse_ligand(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio Ligandos")
        if path: self.ligand_dir_input.setText(path)

    def browse_pocket(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo Pocket", "", "CSV (*.csv);;All Files (*)")
        if path: self.pocket_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Salida")
        if path: self.output_dir_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar valores
        self.settings.setValue("dbGen/last_dssp_dir", self.dssp_dir_input.text())
        self.settings.setValue("dbGen/last_ligand_dir", self.ligand_dir_input.text())
        self.settings.setValue("dbGen/last_pocket_file", self.pocket_input.text())
        self.settings.setValue("dbGen/last_output_dir", self.output_dir_input.text())
        
        self.settings.setValue("dbGen/train_ratio", self.train_spin.value())
        self.settings.setValue("dbGen/val_ratio", self.val_spin.value())
        self.settings.setValue("dbGen/test_ratio", self.test_spin.value())
        self.settings.setValue("dbGen/random_seed", self.seed_spin.value())
        self.settings.setValue("dbGen/cleanup", "true" if self.cleanup_check.isChecked() else "false")
        
        super().accept()

    # ---------- Get Inputs ----------
    def get_inputs(self):
        """
        Retorna un diccionario listo para pasar a la función DB_Generation usando **kwargs
        """
        # Manejo de Output Dir opcional
        out_dir = self.output_dir_input.text().strip()
        if not out_dir:
            out_dir = None

        return {
            "dssp_dir": self.dssp_dir_input.text(),
            "ligand_dir": self.ligand_dir_input.text(),
            "pocket_file": self.pocket_input.text(),
            "output_dir": out_dir,
            "train_ratio": self.train_spin.value(),
            "val_ratio": self.val_spin.value(),
            "test_ratio": self.test_spin.value(),
            "random_seed": self.seed_spin.value(),
            "cleanup_processed": self.cleanup_check.isChecked()
        }