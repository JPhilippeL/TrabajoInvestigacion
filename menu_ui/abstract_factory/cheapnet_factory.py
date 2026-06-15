from menu_ui.dialogs.cheapnet.data_generation_dialog import DBGenerationDialog
from menu_ui.dialogs.cheapnet.hyperparameter_dialog import HyperparameterSearchDialog
from menu_ui.dialogs.cheapnet.predict_dialog import PredictDialog
from menu_ui.dialogs.cheapnet.train_dialog import TrainDialog
from menu_ui.menu import Menu


def cheapnet_menu(parent_window):
    return Menu(
        parent_window=parent_window,
        model_name="CheapNet",
        db_dialog=DBGenerationDialog,
        train_dialog=TrainDialog,
        predict_dialog=PredictDialog,
        hp_dialog=HyperparameterSearchDialog,
        script_path="menu_ui/hyperparameter_adapter/cheapnet_adapter.py",
    )
