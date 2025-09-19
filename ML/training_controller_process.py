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
        
        # Mostrar mensaje en GUI
        self.parent.log("Inicializando entrenamiento...")
        
        self.process = QProcess()
        self.process.setProgram(sys.executable)  # ejecuta el mismo Python
        self.process.setArguments([
            "-m", "ML.trainer_worker",
            "--sdf_dir", sdf_dir,
            "--target_file", target_file,
            "--modelo_nombre", modelo,
            "--epochs", str(epochs),
            "--save_path", save_path,
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

    def on_stdout(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput().data()).decode().strip()
        for line in data.splitlines():
            if line.startswith("FINISHED|"):
                _, path, elapsed = line.split("|")
                logger.info(f"Modelo guardado en: {path} (tiempo {elapsed}s)")
                #self.parent.log(f"Entrenamiento completado en {elapsed}s")
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
