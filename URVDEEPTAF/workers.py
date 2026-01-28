from PySide6.QtCore import QThread, Signal
from URVDEEPTAF.Core.urvdtaf_generate_data import URVDataGenerator
from URVDEEPTAF.Core.urvdtaf_trainer import train
from URVDEEPTAF.Core.urvdtaf_tester import test_model

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

# ==============================================================================
# HILO DE SEGUNDO PLANO PARA ENTRENAMIENTO (QThread)
# ==============================================================================
class TrainThread(QThread):
    # Señales para comunicarse con la interfaz principal
    # Envía la ruta (string) de donde se guardaron los resultados
    finished_success = Signal(str)  
    # Envía el mensaje de error (string)
    finished_error = Signal(str)     

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        """Este método se ejecuta en un hilo separado al llamar a .start()"""
        try:
            # Ejecutamos la función de entrenamiento pesada
            # (Tu función train ya devuelve el run_dir)
            run_dir = train(**self.params)
            
            # Avisamos a la UI que todo salió bien y le pasamos la ruta
            self.finished_success.emit(run_dir)
        except Exception as e:
            # Avisamos a la UI que hubo un error
            self.finished_error.emit(str(e))

# ==============================================================================
# HILO DE SEGUNDO PLANO PARA EVALUACIÓN (QThread)
# ==============================================================================
class TestThread(QThread):
    # Envía el diccionario con las métricas (RMSE, MAE, R2, etc.)
    finished_success = Signal(dict)  
    # Envía el mensaje de error (string)
    finished_error = Signal(str)     

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        """Ejecuta la evaluación en segundo plano"""
        try:
            # Tu función test_model ya retorna el diccionario 'metrics'
            metrics = test_model(**self.params)
            
            # Avisamos a la UI que todo salió bien
            self.finished_success.emit(metrics)
        except Exception as e:
            # Avisamos a la UI que hubo un error
            self.finished_error.emit(str(e))