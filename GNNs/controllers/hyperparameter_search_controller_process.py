from PySide6.QtCore import QProcess
import logging
import sys

logger = logging.getLogger(__name__)


class HyperparameterSearchControllerProcess:
    def __init__(self, parent):
        self.parent = parent
        self.process = None

    def launch_search(self, config):
        if (
            self.process is not None
            and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self._log("A hyperparameter search is already running.")
            return

        logger.info("Inicializando hyperparameter search GNN...")
        self._log("Initializing GNN hyperparameter search...")

        self.process = QProcess()
        self.process.setProgram(sys.executable)
        self.process.setArguments([
            "-m", "GNNs.workers.hyperparameter_search_worker",

            "--train_sdf_dir", config["train_sdf_dir"],
            "--target_file", config["target_file"],
            "--output_root", config["output_root"],

            "--eval_sdf_dir", config.get("eval_sdf_dir", ""),
            "--eval_targets_file", config.get("eval_targets_file", ""),

            "--model_names", config["model_names"],

            "--lr_values", config["lr_values"],
            "--batch_size_values", config["batch_size_values"],
            "--hidden_dim_values", config["hidden_dim_values"],
            "--num_layers_values", config["num_layers_values"],
            "--atom_emb_dim_values", config["atom_emb_dim_values"],
            "--hibrid_emb_dim_values", config["hibrid_emb_dim_values"],
            "--bond_emb_dim_values", config["bond_emb_dim_values"],

            "--epochs", str(config["epochs"]),
            "--patience", str(config["patience"]),
            "--valid_split", str(config["valid_split"]),

            "--objective_metric", config["objective_metric"],
            "--objective_mode", config["objective_mode"],

            "--resume", str(config["resume"]).lower(),
            "--rerun_failed", str(config["rerun_failed"]).lower(),
            "--seed", str(config["seed"]),
        ])

        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_finished)

        self.process.start()

    def on_stdout(self):
        if self.process is None:
            return

        data = bytes(self.process.readAllStandardOutput().data()).decode().strip()

        for line in data.splitlines():
            if line.startswith("STARTED|"):
                _, msg = line.split("|", 1)
                logger.info(msg)
                self._log(msg)

            elif line.startswith("FINISHED|"):
                parts = line.split("|", 7)

                if len(parts) != 8:
                    logger.error(f"Malformed FINISHED message: {line}")
                    self._log(f"Malformed FINISHED message: {line}")
                    return

                (
                    _,
                    status,
                    message,
                    trials_csv,
                    failed_trials_csv,
                    best_config_json,
                    best_config_yaml,
                    elapsed,
                ) = parts

                logger.info(f"Hyperparameter search finished in {elapsed} seconds.")
                logger.info(f"Status: {status}")
                logger.info(f"Message: {message}")
                logger.info(f"Trials CSV: {trials_csv}")
                logger.info(f"Failed trials CSV: {failed_trials_csv}")
                logger.info(f"Best config JSON: {best_config_json}")
                logger.info(f"Best config YAML: {best_config_yaml}")

                self._log(f"Hyperparameter search finished in {elapsed} seconds.")
                self._log(f"Status: {status}")
                self._log(f"Message: {message}")
                self._log(f"Trials CSV: {trials_csv}")
                self._log(f"Failed trials CSV: {failed_trials_csv}")
                self._log(f"Best config JSON: {best_config_json}")
                self._log(f"Best config YAML: {best_config_yaml}")

            elif line.startswith("ERROR|"):
                _, msg = line.split("|", 1)
                logger.error(f"Error in hyperparameter search: {msg}")
                self._log(f"[ERROR] {msg}")

            else:
                self._log(line)

    def on_stderr(self):
        if self.process is None:
            return

        data = bytes(self.process.readAllStandardError().data()).decode().strip()

        if data:
            logger.error(data)
            self._log(f"[stderr] {data}")

    def on_finished(self):
        exit_code = self.process.exitCode() if self.process is not None else None
        logger.info(f"Hyperparameter search process finished with exit code: {exit_code}")
        self.process = None

    def _log(self, message):
        if hasattr(self.parent, "log"):
            self.parent.log(message)
        else:
            logger.info(message)
