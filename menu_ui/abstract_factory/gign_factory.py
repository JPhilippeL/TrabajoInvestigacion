from menu_ui.dialogs.gign.data_generation_dialog import DBGenerationDialog
from menu_ui.dialogs.gign.hyperparameter_dialog import HyperparameterSearchDialog
from menu_ui.dialogs.gign.predict_dialog import PredictDialog
from menu_ui.dialogs.gign.train_dialog import TrainDialog
from menu_ui.menu import Menu


def gign_menu(parent_window):
    return Menu(
        parent_window=parent_window,
        model_name="gign",
        db_dialog=DBGenerationDialog,
        train_dialog=TrainDialog,
        predict_dialog=PredictDialog,
        hp_dialog=HyperparameterSearchDialog,
        script_path="menu_ui/hyperparameter_adapter/gign_adapter.py",
    )
