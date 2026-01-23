from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6.QtGui import QAction
import logging

# Imports existentes
from URVDEEPTAF.ui.dialogs.generate_data_dialog import DBGenerationDialog
from URVDEEPTAF.generate_data import DB_Generation

# Nuevos imports para el entrenamiento
from URVDEEPTAF.ui.dialogs.train_urvdtaf_dialog import TrainDialog
from URVDEEPTAF.trainer import train

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

    # --- GENERACIÓN DE DATOS ---
    def generar_data_urvdeepdtaf(self):
        # Nota: Usamos self.main_window en lugar de self.parent para evitar errores de PySide
        dialog = DBGenerationDialog(self.main_window) 
        
        if dialog.exec():
            # 1. Obtener parametros
            params = dialog.get_inputs()
            
            # 2. Validar
            if not params["dssp_dir"] or not params["pocket_file"]:
                logger.error("Intento de generar data sin directorios requeridos.")
                return

            logger.info(f"Iniciando DB_Generation con: {params}")
            
            # 3. Llamada a la función lógica
            try:
                DB_Generation(**params)
                logger.info("Generación de datos completada exitosamente.")
            except Exception as e:
                logger.error(f"Error durante la generación de datos: {e}")

    # --- ENTRENAMIENTO DE MODELO ---
    # Igual que el anterior, esto congelará la UI.
    def entrenar_modelo_urvdeepdtaf(self):
        dialog = TrainDialog(self.main_window)
        
        if dialog.exec():
            # 1. Obtener parámetros del diálogo
            params = dialog.get_inputs()
            
            # 2. Validación básica
            if not params["data_path"]:
                logger.error("No se puede iniciar el entrenamiento sin un directorio de datos.")
                QMessageBox.warning(self.main_window, "Error", "Debe seleccionar un directorio de datos.")
                return

            logger.info(f"Iniciando entrenamiento del modelo {params['model_name']}...")
            
            # 3. Llamada a la función lógica
            try:
                # La función train retorna la ruta donde se guardaron los resultados
                run_dir = train(**params)
                logger.info(f"Entrenamiento completado. Resultados en: {run_dir}")
                
                # Feedback visual para el usuario cuando termine (útil ya que la UI se descongela aquí)
                QMessageBox.information(
                    self.main_window, 
                    "Éxito", 
                    f"Entrenamiento completado exitosamente.\nResultados guardados en:\n{run_dir}"
                )
            except Exception as e:
                logger.error(f"Error crítico durante el entrenamiento: {e}")