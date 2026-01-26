from PySide6.QtCore import QThread, Signal
from URVDEEPTAF.Core.urvdtaf_generate_data import URVDataGenerator

# ==============================================================================
# HILO DE SEGUNDO PLANO (QThread) PARA NO BLOQUEAR LA UI
# ==============================================================================
class DBGenerationThread(QThread):
    # Señales para comunicarse con la interfaz principal
    finished_success = Signal(dict)  # Envía el diccionario de resultados
    finished_error = Signal(str)     # Envía el mensaje de error

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        """Este método se ejecuta en un hilo separado al llamar a .start()"""
        try:
            # Instanciamos el generador aquí en el hilo secundario
            generator = URVDataGenerator()
            # Ejecutamos la tarea pesada
            results = generator.generate(**self.params)
            # Avisamos a la UI que todo salió bien
            self.finished_success.emit(results)
        except Exception as e:
            # Avisamos a la UI que hubo un error
            self.finished_error.emit(str(e))