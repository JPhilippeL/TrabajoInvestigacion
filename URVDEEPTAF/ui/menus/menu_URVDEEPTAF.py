from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import logging

# Imports existentes
from URVDEEPTAF.ui.dialogs.generate_data_dialog import DBGenerationDialog
from URVDEEPTAF.ui.dialogs.train_urvdtaf_dialog import TrainDialog
from URVDEEPTAF.ui.dialogs.test_urvdtaf_dialog import TestDialog
from URVDEEPTAF.ui.dialogs.batch_train_urvdtaf_dialog import BatchTrainDialog
from URVDEEPTAF.ui.dialogs.batch_test_urvdtaf_dialog import BatchTestDialog

from URVDEEPTAF.workers import DBGenerationThread, TrainThread, TrainAllModelsThread, TestThread, TestAllModelsThread

logger = logging.getLogger(__name__)

class MenuURVDEEPTAF(QMenu):
    def __init__(self, parent_window):
        super().__init__("URVDEEPDTAF", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # 1. Generate Data
        gendata_action = QAction("Generate Data", self)
        gendata_action.triggered.connect(self.generar_data_urvdeepdtaf)
        self.addAction(gendata_action)

        # 2. Train Model 
        train_action = QAction("Train Model", self)
        train_action.triggered.connect(self.entrenar_modelo_urvdeepdtaf)
        self.addAction(train_action)

        # 3. Batch Train Model (CORRECCIÓN: Añadido al menú)
        batch_train_action = QAction("Train All Models", self)
        batch_train_action.triggered.connect(self.entrenar_muchos_modelos_urvdeepdtaf)
        self.addAction(batch_train_action)

        # 4. Test Model 
        test_action = QAction("Evaluate Model", self)
        test_action.triggered.connect(self.testear_modelo_urvdeepdtaf)
        self.addAction(test_action)

        batch_test_action = QAction("Evaluate All Models (Folder)", self)
        batch_test_action.triggered.connect(self.testear_multiples_modelos_urvdeepdtaf)
        self.addAction(batch_test_action)

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
            self.generation_thread.finished_error.connect(self.on_thread_error)
            self.generation_thread.start()

    # --- ENTRENAMIENTO INDIVIDUAL ---
    def entrenar_modelo_urvdeepdtaf(self):
        dialog = TrainDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()
            if not params["data_path"]:
                logger.warning("Falta directorio de datos para el entrenamiento.")
                return

            logger.info(f"Iniciando entrenamiento ({params.get('model_name', 'Modelo')}) en segundo plano...")
            self.main_window.setEnabled(False)

            self.train_thread = TrainThread(params)
            self.train_thread.finished_success.connect(self.on_train_success)
            self.train_thread.finished_error.connect(self.on_thread_error)
            self.train_thread.start()

    # --- ENTRENAMIENTO MÚLTIPLE (CORREGIDO) ---
    def entrenar_muchos_modelos_urvdeepdtaf(self):
        dialog = BatchTrainDialog(self.main_window)
        if dialog.exec():
            # Asumimos que params trae los parámetros base y quizás la lista de modelos
            params = dialog.get_inputs() # Adapta esto según devuelva tu diálogo
            
            if not params["data_path"]:
                logger.warning("Falta directorio de datos para el entrenamiento.")
                return

            logger.info("Iniciando entrenamiento por lotes en segundo plano...")
            self.main_window.setEnabled(False) # Bloqueamos UI

            # Instanciamos el hilo (asegúrate de que los argumentos coincidan con tu __init__ del hilo)
            self.train_batch_thread = TrainAllModelsThread(params)
            
            # Conectamos las señales específicas del batch
            self.train_batch_thread.model_finished_success.connect(self.on_batch_model_success)
            self.train_batch_thread.model_finished_error.connect(self.on_batch_model_error)
            self.train_batch_thread.all_finished.connect(self.on_batch_all_finished) # <- Esta desbloquea la UI
            self.train_batch_thread.start()

    # --- TEST ---
    def testear_modelo_urvdeepdtaf(self):
        dialog = TestDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()
            if not params["model_path"] or not params["data_path"]:
                logger.warning("Faltan rutas de modelo o datos para la evaluación.")
                return

            logger.info("Iniciando evaluación del modelo en segundo plano...")
            self.main_window.setEnabled(False)

            self.test_thread = TestThread(params)
            self.test_thread.finished_success.connect(self.on_test_success)
            self.test_thread.finished_error.connect(self.on_thread_error)
            self.test_thread.start()

    def testear_multiples_modelos_urvdeepdtaf(self):
        # Usamos el BatchTestDialog que arreglamos en el paso anterior
        dialog = BatchTestDialog(self.main_window)
        if dialog.exec():
            params = dialog.get_inputs()
            
            if not params["models_dir"] or not params["data_path"]:
                logger.warning("Faltan rutas de la carpeta madre o datos para la evaluación en lote.")
                return

            logger.info(f"Iniciando evaluación por lotes en la carpeta: {params['models_dir']}")
            self.main_window.setEnabled(False) # Bloqueamos la UI

            # Instanciamos el hilo y conectamos las señales
            self.test_batch_thread = TestAllModelsThread(params)
            self.test_batch_thread.model_finished_success.connect(self.on_batch_test_model_success)
            self.test_batch_thread.model_finished_error.connect(self.on_batch_model_error) # Puedes reusar el del Train
            self.test_batch_thread.all_finished.connect(self.on_batch_test_all_finished)
            
            self.test_batch_thread.start()

    # =========================================================
    # SLOTS DE RESPUESTA
    # =========================================================
    def on_batch_test_model_success(self, model_name, metrics):
        # Formateamos las métricas para que se vean bien en consola
        metrics_log = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"✅ [Batch Test] {model_name} completado. Métricas: {metrics_log}")

    def on_batch_test_all_finished(self, csv_path):
        self.main_window.setEnabled(True) # Desbloqueamos la UI
        
        if csv_path:
            logger.info(f"🎉 Evaluación por lotes finalizada. Resumen guardado en: {csv_path}")
        else:
            logger.warning("Evaluación finalizada, pero no se generaron métricas (¿Todos los modelos fallaron?).")


    # =========================================================================
    # RESPUESTAS DE LOS HILOS (SLOTS SILENCIOSOS)
    # =========================================================================

    def on_thread_error(self, error_msg):
        self.main_window.setEnabled(True)
        logger.exception(f"Proceso en segundo plano fallido: {error_msg}")

    def on_generation_success(self, results):
        self.main_window.setEnabled(True)
        logger.info(f"Generación completada: {results['total_pdbids']} PDBIDs guardados en {results['output_dir']}")

    def on_train_success(self, run_dir):
        self.main_window.setEnabled(True)
        logger.info(f"Entrenamiento completado exitosamente. Results en: {run_dir}")

    # --- NUEVOS SLOTS PARA EL BATCH TRAIN ---
    def on_batch_model_success(self, model_name, run_dir):
        # NOTA: Aquí NO hacemos setEnabled(True) porque faltan modelos
        logger.info(f"[Batch] Modelo {model_name} entrenado. Results en: {run_dir}")

    def on_batch_model_error(self, model_name, error_msg):
        # Tampoco desbloqueamos aquí, dejamos que el hilo intente con el siguiente modelo
        logger.error(f"[Batch] Falló el entrenamiento de {model_name}: {error_msg}")

    def on_batch_all_finished(self):
        # AHORA SÍ desbloqueamos la interfaz
        self.main_window.setEnabled(True)
        logger.info("Entrenamiento por lotes completado para todos los modelos.")

    def on_test_success(self, metrics):
        self.main_window.setEnabled(True)
        metrics_log = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Evaluación completada. Métricas: {metrics_log}")