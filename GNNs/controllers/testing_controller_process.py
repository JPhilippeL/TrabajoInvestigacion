# testing_controller_process.py
from PySide6.QtCore import QProcess
import logging
import sys

logger = logging.getLogger(__name__)

class TestingControllerProcess:
    def __init__(self, parent):
        self.parent = parent
        self.process = None

    def testear_modelos(self, models_dir, sdf_dir, targets_file):
        logger.info("Inicializando test de modelos...")

        self.process = QProcess()
        self.process.setProgram(sys.executable)
        self.process.setArguments([
            "-m", "GNNs.workers.tester_worker",
            "--models_dir", models_dir,
            "--sdf_dir", sdf_dir,
            "--targets_file", targets_file,
        ])

        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.start()

    def on_stdout(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput().data()).decode().strip()
        for line in data.splitlines():
            if line.startswith("RESULT|"):
                _, model, rmse = line.split("|")
                logger.info(f"{model}: RMSE = {rmse}")
            elif line.startswith("FINISHED|"):
                _, resumen_path, elapsed = line.split("|")
                logger.info(f"Testeo finalizado en {elapsed} segundos.")
                logger.info(f"Resumen guardado en: {resumen_path}")
            elif line.startswith("ERROR|"):
                _, msg = line.split("|", 1)
                logger.exception(f"Error en test: {msg}")
            else:
                #logger.info(line)
                self.parent.log(line)

    def on_stderr(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError().data()).decode().strip()
        if data:
            logger.exception(data)
            self.parent.log(f"[stderr] {data}")
