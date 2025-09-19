# training_controller.py

from PySide6.QtCore import QObject, Signal, QThread
import logging
from ML.model_trainer import train_and_save_model
import time
import torch
import gc
logger = logging.getLogger(__name__)

class TrainingController:
    def __init__(self, parent):
        self.parent = parent
        self.thread = None
        self.worker = None

    def entrenar(
    self,
    sdf_dir,
    target_file,
    modelo,
    epochs,
    batch_size,
    lr,
    valid_split,
    save_path,
    hidden_dim=64,   
    num_layers=3,
    patience=0       
    ):
        self.thread = QThread()
        self.worker = TrainerWorker(
            sdf_dir,
            target_file,
            modelo,
            epochs,
            save_path,
            batch_size=batch_size,
            lr=lr,
            valid_split=valid_split,
            hidden_dim=hidden_dim,     
            num_layers=num_layers,
            patience=patience      
        )

        self.worker.moveToThread(self.thread)

        # Conectar señales
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log.connect(self.parent.log)

        # Finalizar hilo correctamente
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_finished(self, ruta):
        logger.info(f"Modelo guardado en: {ruta}")

    def on_error(self, msg):
        logger.error(f"Error en entrenamiento: {msg}")
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()


class TrainerWorker(QObject):
    log = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        sdf_dir,
        target_file,
        modelo_nombre,
        epochs,
        save_path,
        batch_size=32,
        lr=0.001,
        valid_split=0.2,
        hidden_dim=64,
        num_layers=3,
        patience=0
    ):
        super().__init__()
        self.sdf_dir = sdf_dir
        self.target_file = target_file
        self.modelo_nombre = modelo_nombre
        self.epochs = epochs
        self.save_path = save_path
        self.batch_size = batch_size
        self.lr = lr
        self.valid_split = valid_split
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patience = patience


    def run(self):
        try:
            start_time = time.time()

            path = train_and_save_model(
                sdf_dir=self.sdf_dir,
                target_file=self.target_file,
                modelo_nombre=self.modelo_nombre,
                epochs=self.epochs,
                save_path=self.save_path,
                batch_size=self.batch_size,
                lr=self.lr,
                valid_split=self.valid_split,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                patience=self.patience
            )
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"Entrenamiento completado en {elapsed:.2f} segundos.")

            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            torch.cuda.empty_cache()
            gc.collect()

