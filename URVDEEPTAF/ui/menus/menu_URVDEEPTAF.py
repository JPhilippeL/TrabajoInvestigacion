from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6.QtGui import QAction
import logging

# Imports existentes
from URVDEEPTAF.ui.dialogs.generate_data_dialog import DBGenerationDialog
from URVDEEPTAF.ui.dialogs.train_urvdtaf_dialog import TrainDialog
from URVDEEPTAF.ui.dialogs.test_urvdtaf_dialog import TestDialog

from URVDEEPTAF.workers import DBGenerationThread, TrainThread, TestThread

logger = logging.getLogger(__name__)

class MenuURVDEEPTAF(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("URVDEEPDTAF", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # 1. Generate Data
        gendata_action = QAction("Generar Data", self)
        gendata_action.triggered.connect(self.generar_data_urvdeepdtaf)
        self.addAction(gendata_action)

        # 2. Train Model (Nueva Acción)
        train_action = QAction("Entrenar Modelo", self)
        train_action.triggered.connect(self.entrenar_modelo_urvdeepdtaf)
        self.addAction(train_action)

        # 3. Test Model (Nueva Acción)
        test_action = QAction("Evaluar Modelo", self)
        test_action.triggered.connect(self.testear_modelo_urvdeepdtaf)
        self.addAction(test_action)

    # --- GENERACIÓN DE DATOS ---
    def generar_data_urvdeepdtaf(self):
        dialog = DBGenerationDialog(self.main_window) 
        if dialog.exec():
            params = dialog.get_inputs()
            if not params["dssp_dir"] or not params["pocket_file"]:
                logger.warning("Faltan directorios requeridos para generar datos.")
                return

            logger.info("Iniciando generación de datos en segundo plano...")
            self.main_window.setEnabled(False) 

            self.generation_thread = DBGenerationThread(params)
            self.generation_thread.finished_success.connect(self.on_generation_success)
            self.generation_thread.finished_error.connect(self.on_thread_error) # <--- Error General
            self.generation_thread.start()

    def entrenar_modelo_urvdeepdtaf(self):
        dialog = TrainDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()
            if not params["data_path"]:
                logger.warning("Falta directorio de datos para el entrenamiento.")
                return

            logger.info(f"Iniciando entrenamiento ({params['model_name']}) en segundo plano...")
            self.main_window.setEnabled(False)

            self.train_thread = TrainThread(params)
            self.train_thread.finished_success.connect(self.on_train_success)
            self.train_thread.finished_error.connect(self.on_thread_error) # <--- Error General
            self.train_thread.start()

    # --- NUEVA ACCIÓN: TEST ---
    def testear_modelo_urvdeepdtaf(self):
        dialog = TestDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()
            if not params["model_path"] or not params["data_path"]:
                logger.warning("Faltan rutas de modelo o datos para la evaluación.")
                return

            logger.info(f"Iniciando evaluación del modelo en segundo plano...")
            self.main_window.setEnabled(False)

            self.test_thread = TestThread(params)
            self.test_thread.finished_success.connect(self.on_test_success)
            self.test_thread.finished_error.connect(self.on_thread_error)
            self.test_thread.start()


    # =========================================================================
    # RESPUESTAS DE LOS HILOS (SLOTS SILENCIOSOS)
    # =========================================================================

    # 1. ERROR GENERAL (Para TODOS los hilos)
    def on_thread_error(self, error_msg):
        self.main_window.setEnabled(True)
        logger.exception(f"Proceso en segundo plano fallido: {error_msg}")

    # 2. ÉXITOS ESPECÍFICOS
    def on_generation_success(self, results):
        self.main_window.setEnabled(True)
        logger.info(f"Generación completada: {results['total_pdbids']} PDBIDs guardados en {results['output_dir']}")

    def on_train_success(self, run_dir):
        self.main_window.setEnabled(True)
        logger.info(f"Entrenamiento completado exitosamente. Resultados en: {run_dir}")

    # --- NUEVO ÉXITO DE TEST ---
    def on_test_success(self, metrics):
        self.main_window.setEnabled(True)
        
        # Formatear las métricas para que se vean bien en el log
        metrics_log = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Evaluación completada. Métricas: {metrics_log}")