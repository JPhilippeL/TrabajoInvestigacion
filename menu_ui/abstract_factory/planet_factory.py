from menu_ui.dialogs.planet.data_generation_dialog import DBGenerationDialog
from menu_ui.dialogs.planet.hyperparameter_dialog import HyperparameterSearchDialog
from menu_ui.dialogs.planet.predict_dialog import PredictDialog
from menu_ui.dialogs.planet.train_dialog import TrainDialog
from menu_ui.menu import Menu


def planet_menu(parent_window):
    return Menu(
        parent_window=parent_window,
        model_name="planet",
        db_dialog=DBGenerationDialog,
        train_dialog=TrainDialog,
        predict_dialog=PredictDialog,
        hp_dialog=HyperparameterSearchDialog,
        script_path="menu_ui/hyperparameter_adapter/planet_adapter.py",
    )
