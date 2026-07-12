import logging

from PySide6.QtWidgets import QMenuBar

from menu_ui.abstract_factory.cheapnet_factory import cheapnet_menu
from menu_ui.abstract_factory.gign_factory import gign_menu
from menu_ui.abstract_factory.graph_dta_factory import graph_dta_menu
from menu_ui.abstract_factory.planet_factory import planet_menu
from ui.menus import (
    MenuMolecula,
    MenuExplainerGNN,
    MenuTestGNN,
    MenuTrainGNN,
    MenuTransferGNN,
    MenuHyperparameterSearchGNN,
)
from URVDEEPTAF.ui.menus import MenuURVDEEPTAF
from EGNN.ui.menus.menu_EGNN import MenuEGNN
from EDNN.ui.menus.menu_EDNN import MenuEDNN
from DeepDTA.ui.menus.menu_DeepDTA import MenuDeepDTA
from WideDTA.ui.menus.menu_WideDTA import MenuWideDTA
from DCML.ui.menus.menu_DCML import MenuDCML
from CAPLA.ui.menus.menu_CAPLA import MenuCAPLA
from DEAttentionDTA.ui.menus.menu_DEAttentionDTA import MenuDEAttentionDTA


logger = logging.getLogger(__name__)


class MenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent  # Reference to MainWindow

        # Professor-facing model menus use the final Generate Data -> Train -> Search -> Evaluate workflow.
        self.menu_urvdeepdtaf = MenuURVDEEPTAF(self.parent)
        self.addMenu(self.menu_urvdeepdtaf)

        self.menu_cheapnet = cheapnet_menu(self.parent)
        self.addMenu(self.menu_cheapnet)

        self.menu_gign = gign_menu(self.parent)
        self.addMenu(self.menu_gign)

        self.menu_dta = graph_dta_menu(self.parent)
        self.addMenu(self.menu_dta)

        self.planet_menu = planet_menu(self.parent)
        self.addMenu(self.planet_menu)

        self.menu_EGNN = MenuEGNN(self.parent)
        self.addMenu(self.menu_EGNN)

        self.menu_EDNN = MenuEDNN(self.parent)
        self.addMenu(self.menu_EDNN)

        self.menu_DeepDTA = MenuDeepDTA(self.parent)
        self.addMenu(self.menu_DeepDTA)

        self.menu_WideDTA = MenuWideDTA(self.parent)
        self.addMenu(self.menu_WideDTA)

        self.menu_DCML = MenuDCML(self.parent)
        self.addMenu(self.menu_DCML)

        self.menu_CAPLA = MenuCAPLA(self.parent)
        self.addMenu(self.menu_CAPLA)

        self.menu_DEAttentionDTA = MenuDEAttentionDTA(self.parent)
        self.addMenu(self.menu_DEAttentionDTA)

        # Legacy GNN utilities remain available, but are visually secondary.
        self.menu_molecula = MenuMolecula(self.parent)
        self.addMenu(self.menu_molecula)

        self.menu_train = MenuTrainGNN(self.parent)
        self.addMenu(self.menu_train)

        self.menu_transfer = MenuTransferGNN(self.parent)
        self.addMenu(self.menu_transfer)

        self.menu_test = MenuTestGNN(self.parent)
        self.addMenu(self.menu_test)

        self.menu_hyperparameter_search = MenuHyperparameterSearchGNN(self.parent)
        self.addMenu(self.menu_hyperparameter_search)

        self.menu_explicacion = MenuExplainerGNN(self.parent)
        self.addMenu(self.menu_explicacion)
