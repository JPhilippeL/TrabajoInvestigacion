from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QDialog
from PySide6.QtGui import QAction
import os
from rdkit import Chem
import logging

from ui.dialogs.split_sdf_dialog import SDFSplitDialog
from ui.dialogs.csv_to_sdf import CSVtoSDFDialog

from graph_managment.sdf_converter import graph_to_mol, save_graph_as_sdf, split_sdf, smiles_csv_to_sdf_dir

logger = logging.getLogger(__name__)

class MenuMolecula(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("Molécula", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        #Menu de Molécula

        nuevo_action = QAction("Nueva", self)
        nuevo_action.triggered.connect(self.nuevo_molecula)
        self.addAction(nuevo_action)

        cargar_action = QAction("Cargar", self)
        cargar_action.triggered.connect(self.cargar_molecula)
        self.addAction(cargar_action)

        guardar_action = QAction("Guardar", self)
        guardar_action.triggered.connect(self.guardar_molecula)
        self.addAction(guardar_action)

        verificar_action = QAction("Verificar Molécula", self)
        verificar_action.triggered.connect(self.verificar_molecula)
        self.addAction(verificar_action)

        dividir_action = QAction("Dividir SDF", self)
        dividir_action.triggered.connect(self.dividir_sdf)
        self.addAction(dividir_action)

        dividir_csv_action = QAction("CSV a SDF", self)
        dividir_csv_action.triggered.connect(self.csv_a_sdf)
        self.addAction(dividir_csv_action)

    def nuevo_molecula(self):
        self.main_window.create_new_graph()

    def cargar_molecula(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Seleccionar archivo SDF",
            "",
            "Archivos SDF (*.sdf);;Todos los archivos (*)"
        )
        if file_path:
            self.main_window.load_graph_from_file(file_path)


    def guardar_molecula(self):
        if not self.main_window.graph_view:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Guardar archivo SDF",
            "",
            "Archivos SDF (*.sdf)"
        )
        if not file_path:
            return

        graph = self.main_window.graph_view.scene().graph

        try:
            save_graph_as_sdf(graph, file_path)
            mensaje = f"Archivo guardado correctamente en: {file_path}"
            logger.info(mensaje)
            # self.main_window.load_graph_from_file(file_path)  # Recargar el grafo guardado
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error al guardar",
                str(e)
            )

    def verificar_molecula(self):
        if not self.main_window.graph_view:
            mensaje = "No hay una molécula cargada para verificar."
            logger.warning(mensaje)
            return

        try:
            mol = graph_to_mol(self.main_window.graph_view.scene().graph)
            Chem.SanitizeMol(mol)
            mensaje = "La molécula no contiene errores químicos detectables."
            logger.info(mensaje)
        except Exception as e:
            mensaje = f"Se detectaron errores químicos en la molécula:\n{str(e)}"
            logger.exception(mensaje)

    def dividir_sdf(self):
        dialog = SDFSplitDialog(self.main_window)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # Validaciones básicas
        if not config["sdf_file"] or not os.path.isfile(config["sdf_file"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo SDF válido.")
            return

        if not config["output_dir"]:
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio de salida.")
            return

        try:
            split_sdf(config["sdf_file"], config["output_dir"])
            mensaje = f"SDF dividido correctamente. Archivos guardados en: {config['output_dir']}"
            logger.info(mensaje)
        except Exception as e:
            logger.exception(f"Error al dividir SDF: {str(e)}", exc_info=True)

    def csv_a_sdf(self):
        dialog = CSVtoSDFDialog(self.main_window)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # Validaciones
        if not config["csv_file"] or not os.path.isfile(config["csv_file"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo CSV válido.")
            return

        if not config["output_dir"]:
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio de salida.")
            return

        try:
            smiles_csv_to_sdf_dir(config["csv_file"], config["output_dir"])
            mensaje = f"CSV dividido correctamente. SDFs guardados en: {config['output_dir']}"
            logger.info(mensaje)
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error al dividir CSV", str(e))
            logger.exception(f"Error al dividir CSV: {str(e)}", exc_info=True)