# training_controller_process.py

from PySide6.QtCore import QProcess
import logging
import sys

logger = logging.getLogger(__name__)


class TrainingControllerProcess:
    def __init__(self, parent):
        self.parent = parent
        self.process = None

    def entrenar(
        self,
        sdf_dir,
        target_file,
        model_type,
        epochs,
        batch_size,
        lr,
        valid_split,
        model_name,
        hidden_dim=64,
        num_layers=3,
        patience=0
    ):
        
        # Mostrar mensaje en GUI
        logger.info("Inicializando entrenamiento...")
        
        self.process = QProcess()
        self.process.setProgram(sys.executable)  # ejecuta el mismo Python
        self.process.setArguments([
            "-m", "ML.trainer_worker",
            "--sdf_dir", sdf_dir,
            "--target_file", target_file,
            "--model_type", model_type,
            "--epochs", str(epochs),
            "--model_name", model_name,
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--valid_split", str(valid_split),
            "--hidden_dim", str(hidden_dim),
            "--num_layers", str(num_layers),
            "--patience", str(patience)
        ])

        # Conectar señales
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        #self.process.finished.connect(self.on_finished)

        self.process.start()

    def transfer_learning(
        self,
        sdf_dir,
        target_file,
        pretrained_model_path,
        transfer_mode,
        epochs,
        batch_size,
        lr,
        valid_split,
        model_name,
        patience=0
    ):
        logger.info("Inicializando Transfer Learning...")

        self.process = QProcess()
        self.process.setProgram(sys.executable)
        self.process.setArguments([
            "-m", "ML.transfer_trainer_worker",
            "--sdf_dir", sdf_dir,
            "--target_file", target_file,
            "--pretrained_model_path", pretrained_model_path,
            "--transfer_mode", transfer_mode,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--valid_split", str(valid_split),
            "--model_name", model_name,
            "--patience", str(patience)
        ])

        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.start()


    def on_stdout(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput().data()).decode().strip()
        for line in data.splitlines():
            if line.startswith("FINISHED|"):
                _, path, elapsed = line.split("|")
                logger.info(f"Entrenamiento completado en {elapsed} segundos.")
                logger.info(f"Modelo guardado en: {path}")
            elif line.startswith("ERROR|"):
                _, msg = line.split("|", 1)
                logger.error(f"Error en entrenamiento: {msg}")
                #self.parent.log(f"Error: {msg}")
            else:
                # logs normales
                self.parent.log(line)

    def on_stderr(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError().data()).decode().strip()
        if data:
            logger.error(data)
            self.parent.log(f"[stderr] {data}")

    #def on_finished(self, exitCode, exitStatus):
    #    logger.info(f"Proceso de entrenamiento terminado con código {exitCode}")
