from PySide6.QtWidgets import QFileDialog, QMenu
from PySide6.QtGui import QAction
import os
import logging

from ui.dialogs.model_test_dialog import ModelTestDialog
from ui.dialogs.batch_model_test_dialog import BatchModelTestDialog
from ui.dialogs.test_all_models_dialog import BatchAllModelsTestDialog
from ui.dialogs.image_dialog import ImageDialog
from ui.dialogs.batch_model_test_pt_dialog import BatchAllModelsTestDialogPT

from GNNs.model_tester import test_model_on_directory,cargar_y_predecir, obtener_info_checkpoint, test_all_models_in_directory_pt

logger = logging.getLogger(__name__)

class MenuTestGNN(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("Test GNN", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):

        # Testeo de IA con un solo SDF
        testeo_action = QAction("Predict SDF", self)
        testeo_action.triggered.connect(self.testear_modelo)
        self.addAction(testeo_action)

        # Testeo de IA con múltiples SDF
        testeo_batch_action = QAction("Test Model", self)
        testeo_batch_action.triggered.connect(self.testear_modelo_en_batch)
        self.addAction(testeo_batch_action)

        # Testeo de TODOS los modelos en un directorio
        testeo_all_models_action = QAction("Test All Models", self)
        testeo_all_models_action.triggered.connect(self.testear_directorio_modelos)
        self.addAction(testeo_all_models_action)

        # Testeo de TODOS los modelos en un directorio
        testeo_all_models_pt_action = QAction("Testear todos los modelos PT", self)
        testeo_all_models_pt_action.triggered.connect(self.testear_directorio_modelos_pt)
        self.addAction(testeo_all_models_pt_action)

        # Consultar parámetros modelo
        consultar_params_action = QAction("Inspect Model", self)
        consultar_params_action.triggered.connect(self.consultar_parametros_modelo)
        self.addAction(consultar_params_action)

    def testear_modelo(self):
        dialog = ModelTestDialog(self.main_window)
        if dialog.exec():
            model_path, sdf_path = dialog.get_paths()
            try:
                pred, target_name = cargar_y_predecir(model_path, sdf_path)

                model_name = os.path.basename(model_path)
                sdf_name = os.path.basename(sdf_path)
                msg = f"Predicción de '{target_name}' con el modelo '{model_name}' en la molécula '{sdf_name}': {pred:.4f}"
                logger.info(msg)

            except Exception as e:
                logger.exception(f"Error en testear modelo: {str(e)}", exc_info=True)

    def testear_modelo_en_batch(self):
        
        dialog = BatchModelTestDialog(self.main_window)
        if dialog.exec():
            model_path, sdf_dir, targets_file = dialog.get_paths()

            try:
                # Ejecutar función de testeo
                plot_path = test_model_on_directory(model_path, sdf_dir, targets_file)

                # Mostrar scatter plot
                self.image_dialog = ImageDialog(plot_path, self.main_window)
                self.image_dialog.show()

            except Exception as e:
                logger.exception("Error en testeo por lotes: " + str(e), exc_info=True)

    def testear_directorio_modelos(self):
        dialog = BatchAllModelsTestDialog(self.main_window)
        if dialog.exec():
            models_dir, sdf_dir, targets_file = dialog.get_paths()

            try:
                # Ejecutamos testing con el proceso
                self.main_window.testing_controller.testear_modelos(models_dir, sdf_dir, targets_file)
            except Exception as e:
                logger.exception("Error en testeo de todos los modelos: " + str(e), exc_info=True)

    def testear_directorio_modelos_pt(self):
        dialog = BatchAllModelsTestDialogPT(self.main_window)
        if dialog.exec():
            models_dir, pt_file = dialog.get_paths()

            try:
                # Ejecutamos testing con el proceso
                test_all_models_in_directory_pt(models_dir, pt_file)
            except Exception as e:
                logger.error("Error en testeo de todos los modelos: " + str(e), exc_info=True)

    def consultar_parametros_modelo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Seleccionar archivo de modelo (.pt)",
            "",
            "Modelos (*.pt)"
        )
        if not file_path:
            return

        info = obtener_info_checkpoint(file_path)
        logger.info(info)