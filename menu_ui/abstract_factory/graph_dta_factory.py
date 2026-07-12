from menu_ui.dialogs.graph_dta.data_generation_dialog import DBGenerationDialog
from menu_ui.dialogs.graph_dta.hyperparameter_dialog import HyperparameterSearchDialog
from menu_ui.dialogs.graph_dta.predict_dialog import PredictDialog
from menu_ui.dialogs.graph_dta.train_dialog import TrainDialog
from menu_ui.menu import Menu


def graph_dta_menu(parent_window):
    return Menu(
        parent_window=parent_window,
        model_name="graph_dta",
        db_dialog=DBGenerationDialog,
        train_dialog=TrainDialog,
        predict_dialog=PredictDialog,
        hp_dialog=HyperparameterSearchDialog,
        script_path="menu_ui/hyperparameter_adapter/graph_dta_adapter.py",
    )
