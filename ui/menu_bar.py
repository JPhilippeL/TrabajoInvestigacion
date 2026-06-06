from PySide6.QtWidgets import QMenuBar
import logging

from WideDTA.ui.menus.menu_WideDTA import MenuWideDTA
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


logger = logging.getLogger(__name__)


class MenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent  # Reference to MainWindow

        # 1. Molecule menu
        self.menu_molecula = MenuMolecula(self.parent)
        self.addMenu(self.menu_molecula)

        # 2. Train menu
        self.menu_train = MenuTrainGNN(self.parent)
        self.addMenu(self.menu_train)

        # 3. Transfer menu
        self.menu_transfer = MenuTransferGNN(self.parent)
        self.addMenu(self.menu_transfer)

        # 4. Test menu
        self.menu_test = MenuTestGNN(self.parent)
        self.addMenu(self.menu_test)

        # 5. Hyperparameter Search menu
        self.menu_hyperparameter_search = MenuHyperparameterSearchGNN(self.parent)
        self.addMenu(self.menu_hyperparameter_search)

        # 6. Explainer menu
        self.menu_explicacion = MenuExplainerGNN(self.parent)
        self.addMenu(self.menu_explicacion)

        # URV DeepTAF menu
        self.menu_urvdeepdtaf = MenuURVDEEPTAF(self.parent)
        self.addMenu(self.menu_urvdeepdtaf)

        # EGNN menu
        self.menu_EGNN = MenuEGNN(self.parent)
        self.addMenu(self.menu_EGNN)

        # EDNN menu
        self.menu_EDNN = MenuEDNN(self.parent)
        self.addMenu(self.menu_EDNN)

        # DeepDTA menu
        self.menu_DeepDTA = MenuDeepDTA(self.parent)
        self.addMenu(self.menu_DeepDTA)

        # WideDTA menu
        self.menu_WideDTA = MenuWideDTA(self.parent)
        self.addMenu(self.menu_WideDTA)

        # DCML menu
        self.menu_DCML = MenuDCML(self.parent)
        self.addMenu(self.menu_DCML)

        # CAPLA menu
        self.menu_CAPLA = MenuCAPLA(self.parent)
        self.addMenu(self.menu_CAPLA)
