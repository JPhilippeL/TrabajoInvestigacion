from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import logging

from URVDEEPTAF.ui.dialogs.generate_data_dialog import DBGenerationDialog
from URVDEEPTAF.generate_data import DB_Generation

logger = logging.getLogger(__name__)

class MenuURVDEEPTAF(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("URVDEEPDTAF", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # Generate Data
        gendata_action = QAction("Generar Data", self)
        gendata_action.triggered.connect(self.generar_data_urvdeepdtaf)
        self.addAction(gendata_action)

    # Se debería hacer otro thread o alguna otra manera para que no se congele la UI para generar los datos
    # Pero 0 ganas
    def generar_data_urvdeepdtaf(self):
        dialog = DBGenerationDialog(self.parent)
        
        if dialog.exec():
            # 1. Obtener parametros
            params = dialog.get_inputs()
            
            # 2. Validar (opcional, pero recomendado)
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