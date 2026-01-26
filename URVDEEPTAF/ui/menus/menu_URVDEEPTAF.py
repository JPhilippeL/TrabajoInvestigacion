from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6.QtGui import QAction
import logging

# Imports existentes
from URVDEEPTAF.ui.dialogs.generate_data_dialog import DBGenerationDialog
from URVDEEPTAF.Core.urvdtaf_generate_data import DB_Generation

# Nuevos imports para el entrenamiento
from URVDEEPTAF.ui.dialogs.train_urvdtaf_dialog import TrainDialog
from URVDEEPTAF.Core.urvdtaf_trainer import train
from URVDEEPTAF.ui.dialogs.test_urvdtaf_dialog import TestDialog
from URVDEEPTAF.Core.urvdtaf_tester import test_model

from URVDEEPTAF.workers import DBGenerationThread

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
                QMessageBox.warning(self.main_window, "Error", "Faltan directorios requeridos.")
                return

            logger.info("Iniciando generación de datos en segundo plano...")
            
            # BLOQUEAR la ventana principal temporalmente para que el usuario no toque nada
            # (Opcional, pero recomendado)
            self.main_window.setEnabled(False) 

            # 1. CREAR EL HILO
            self.generation_thread = DBGenerationThread(params)

            # 2. CONECTAR LAS SEÑALES (Éxito y Error) a nuestras funciones
            self.generation_thread.finished_success.connect(self.on_generation_success)
            self.generation_thread.finished_error.connect(self.on_generation_error)

            # 3. ¡INICIAR! (Esto llama al método 'run()' sin bloquear la UI)
            self.generation_thread.start()

    # --- FUNCIONES DE RESPUESTA (SLOTS) ---

    def on_generation_success(self, results):
        """Se ejecuta cuando el hilo termina correctamente"""
        self.main_window.setEnabled(True) # Desbloquear la ventana
        logger.info("Generación de datos completada exitosamente.")

    def on_generation_error(self, error_msg):
        """Se ejecuta si hubo una excepción en el hilo"""
        self.main_window.setEnabled(True) # Desbloquear la ventana
        logger.error(f"Error en generación: {error_msg}")

    # --- ENTRENAMIENTO DE MODELO ---
    # Igual que el anterior, esto congelará la UI.
    def entrenar_modelo_urvdeepdtaf(self):
        dialog = TrainDialog(self.main_window)
        
        if dialog.exec():
            # 1. Obtener parámetros del diálogo
            params = dialog.get_inputs()
            
            # 2. Validación básica
            if not params["data_path"]:
                logger.exception("No se puede iniciar el entrenamiento sin un directorio de datos.")
                QMessageBox.warning(self.main_window, "Error", "Debe seleccionar un directorio de datos.")
                return

            logger.info(f"Iniciando entrenamiento del modelo {params['model_name']}...")
            
            # 3. Llamada a la función lógica
            try:
                # La función train retorna la ruta donde se guardaron los resultados
                run_dir = train(**params)
                logger.info(f"Entrenamiento completado. Resultados en: {run_dir}")
                
                # Feedback visual para el usuario cuando termine (útil ya que la UI se descongela aquí)
            except Exception as e:
                logger.exception(f"Error crítico durante el entrenamiento: {e}")

    # --- NUEVA ACCIÓN: EVALUACIÓN DE MODELO ---
    def testear_modelo_urvdeepdtaf(self):
        dialog = TestDialog(self.main_window)
        
        if dialog.exec():
            params = dialog.get_inputs()
            
            # Validación: Necesitamos el modelo (.pt) y los datos
            if not params["model_path"] or not params["data_path"]:
                QMessageBox.warning(
                    self.main_window, 
                    "Faltan rutas", 
                    "Debe seleccionar tanto el archivo de pesos del modelo como la carpeta de datos."
                )
                return

            logger.info(f"Iniciando evaluación del modelo con pesos en: {params['model_path']}")
            
            try:
                # La función test_model retorna un diccionario de métricas
                metrics = test_model(**params)
                
                # Formateamos las métricas principales para mostrarlas en el mensaje de éxito
                metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                
                logger.info("Evaluación completada exitosamente.")
            except Exception as e:
                logger.exception(f"Error crítico durante la evaluación: {e}")