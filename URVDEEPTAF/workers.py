from PySide6.QtCore import QThread, Signal
from URVDEEPTAF.Core.urvdtaf_generate_data import URVDataGenerator
from URVDEEPTAF.Core.urvdtaf_trainer import train
from URVDEEPTAF.Core.urvdtaf_tester import test_model
from URVDEEPTAF.Core.urvdtaf_model import MODEL_DICT, DeepDTAF_GNN, DeepDTAF_GNN_NoPocket, DeepDTAF_GNN_NoProtein, DeepDTAF_GNN_OnlyLigand
import traceback
import os
import glob
import pandas as pd

MODEL_DICT_2 = {
    'DeepDTAF_GNN': DeepDTAF_GNN,              # Model E
    'DeepDTAF_GNN_NoPocket': DeepDTAF_GNN_NoPocket,     # Model F
    'DeepDTAF_GNN_NoProtein': DeepDTAF_GNN_NoProtein,    # Model G
    'DeepDTAF_GNN_OnlyLigand': DeepDTAF_GNN_OnlyLigand,   # Model H
}

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
        self.model_names = MODEL_DICT_2 # Lista de las keys de MODEL_DICT

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

class TestAllModelsThread(QThread):
    # --- Señales ---
    # modelo_nombre, diccionario_metricas
    model_finished_success = Signal(str, dict)  
    model_finished_error = Signal(str, str)    
    # Emite la ruta del CSV final cuando termina todo
    all_finished = Signal(str)                  
    progress_update = Signal(int, int, str)    

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        models_dir = self.params.get("models_dir")
        
        # 1. Buscar todas las subcarpetas dentro de models_dir (ej. split0/)
        # Solo obtenemos directorios, no archivos sueltos
        subdirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) 
                if os.path.isdir(os.path.join(models_dir, d))]
        
        all_metrics = []
        total_models = len(subdirs)

        for index, subdir in enumerate(subdirs):
            model_name = os.path.basename(subdir) # Ej: "DeepDTAF"
            self.progress_update.emit(index + 1, total_models, model_name)
            
            # 2. Buscar el archivo .pt o .pth dentro de esta carpeta
            pt_files = glob.glob(os.path.join(subdir, "*.pt")) + glob.glob(os.path.join(subdir, "*.pth"))
            
            if not pt_files:
                self.model_finished_error.emit(model_name, f"No se encontró ningún archivo .pt en {subdir}")
                continue # Saltamos a la siguiente carpeta
            
            # Tomamos el primer .pt que encuentre
            model_path = pt_files[0] 
            
            # 3. Preparamos los parámetros exactos para test_model
            current_params = self.params.copy()
            current_params.pop("models_dir", None) # Quitamos esto porque test_model no lo acepta
            
            current_params["model_path"] = model_path
            # Le decimos que guarde los resultados (output_base) en la MISMA carpeta del modelo
            current_params["output_base"] = subdir 
            
            try:
                # 4. Ejecutamos la evaluación
                metrics = test_model(**current_params)
                
                # Preparamos los datos para el CSV general
                metrics_for_df = {"Model": model_name}
                metrics_for_df.update(metrics) # Fusionamos el nombre con las métricas
                all_metrics.append(metrics_for_df)
                
                self.model_finished_success.emit(model_name, metrics)
                
            except Exception as e:
                error_trace = traceback.format_exc()
                self.model_finished_error.emit(model_name, error_trace)
                
        # 5. Al terminar, generamos el CSV global
        csv_path = ""
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            # Se guardará como split0/resumen_metricas_modelos.csv
            csv_path = os.path.join(models_dir, "resumen_metricas_modelos.csv")
            df.to_csv(csv_path, index=False)
        
        # Avisamos a la UI que todo el lote terminó
        self.all_finished.emit(csv_path)