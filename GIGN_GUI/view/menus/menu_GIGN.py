from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction


class MenuGIGN(QMenu):
    def __init__(self, parent_window):
        super().__init__("GIGN", parent_window)
        self.main_window = parent_window

        self.init_actions()

    def init_actions(self):
        # 1. Generate Data
        gendata_action = QAction("DB Generation", self)
        gendata_action.triggered.connect(self.generate_db)
        self.addAction(gendata_action)

        # 2. Train Model
        train_action = QAction("Entrenar Modelo", self)
        train_action.triggered.connect(self.train_gign)
        self.addAction(train_action)

        # 4. Test Model
        test_action = QAction("Predict Modelo", self)
        test_action.triggered.connect(self.predict_gign)
        self.addAction(test_action)

        hyperparameter_tuning_action = QAction("Hyperparameter Tuning", self)
        hyperparameter_tuning_action.triggered.connect(self.hyperparameter_tuning)
        self.addAction(hyperparameter_tuning_action)

    def generate_db(self):
        pass

    def train_gign(self):
        pass

    def predict_gign(self):
        pass

    def hyperparameter_tuning(self):
        pass

    def on_batch_test_model_success(self, model_name, metrics):
        pass

    def on_batch_test_all_finished(self, csv_path):
        pass

    def on_generation_success(self, results):
        pass

    def on_train_success(self, run_dir):
        pass

    def on_hyperparameter_tuning_success(self, hyperparameter_tuning):
        pass

    def on_batch_model_success(self, model_name, run_dir):
        pass

    def on_batch_model_error(self, model_name, error_msg):
        pass

    def on_test_success(self, metrics):
        pass
