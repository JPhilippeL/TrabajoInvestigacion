from PySide6.QtCore import QThread, Signal
from URVDEEPTAF.Core.urvdtaf_generate_data import URVDataGenerator
from URVDEEPTAF.Core.urvdtaf_trainer import train
from URVDEEPTAF.Core.urvdtaf_tester import test_model
from URVDEEPTAF.Core.urvdtaf_model import MODEL_DICT
import traceback

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

class TrainAllModelsThread(QThread):
    # --- Señales ---
    # Se emite cada vez que un modelo termina exitosamente: (nombre_modelo, run_dir)
    model_finished_success = Signal(str, str)  
    
    # Se emite si un modelo falla: (nombre_modelo, mensaje_error)
    model_finished_error = Signal(str, str)    
    
    # Se emite al terminar TODOS los modelos
    all_finished = Signal()                    
    
    # Se emite al inicio de cada modelo para actualizar una barra de progreso
    # (modelo_actual, total_modelos, nombre_modelo)
    progress_update = Signal(int, int, str)    

    def __init__(self, base_params, parent=None):
        super().__init__(parent)
        self.base_params = base_params
        self.model_names = MODEL_DICT # Lista de las keys de MODEL_DICT

    def run(self):
        """Itera sobre la lista de modelos y los entrena uno por uno."""
        total_models = len(self.model_names)
        
        for index, model_name in enumerate(self.model_names):
            # 1. Avisar a la interfaz qué modelo va a empezar
            self.progress_update.emit(index + 1, total_models, model_name)
            
            # 2. Copiar los parámetros base para no sobrescribir la misma referencia
            current_params = self.base_params.copy()
            
            # 3. Cambiar solo el modelo. 
            # IMPORTANTE: Cambia 'model_name' por la key exacta que espere tu función train()
            current_params['model_name'] = model_name 
            
            try:
                # 4. Ejecutar el entrenamiento
                run_dir = train(**current_params)
                
                # 5. Avisar que este modelo en concreto salió bien
                self.model_finished_success.emit(model_name, run_dir)
                
            except Exception as e:
                # AQUÍ ESTÁ LA MAGIA:
                # format_exc() atrapa el error completo con archivos y líneas
                error_completo = traceback.format_exc()
                
                # Emitimos el error completo en lugar de solo str(e)
                self.model_finished_error.emit(model_name, error_completo)
        
        # 6. Al salir del bucle, avisamos que terminó el lote completo
        self.all_finished.emit()