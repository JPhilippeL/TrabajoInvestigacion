import logging

from PySide6.QtWidgets import QMenuBar

from menu_ui.abstract_factory.cheapnet_factory import cheapnet_menu
from menu_ui.abstract_factory.gign_factory import gign_menu
from ui.menus import MenuExplainerGNN, MenuMolecula, MenuTestGNN, MenuTrainGNN, MenuTransferGNN
from URVDEEPTAF.ui.menus import MenuURVDEEPTAF

logger = logging.getLogger(__name__)


class MenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent  # Referencia a MainWindow

        # 1. Menú Molécula
        self.menu_molecula = MenuMolecula(self.parent)
        self.addMenu(self.menu_molecula)

        # 2. Menú Train
        self.menu_train = MenuTrainGNN(self.parent)
        self.addMenu(self.menu_train)

        # 3. Menú Transfer
        self.menu_transfer = MenuTransferGNN(self.parent)
        self.addMenu(self.menu_transfer)

        # 4. Menú Test
        self.menu_test = MenuTestGNN(self.parent)
        self.addMenu(self.menu_test)

        # 5. Menú Explicadores
        self.menu_explicacion = MenuExplainerGNN(self.parent)
        self.addMenu(self.menu_explicacion)

        # Menu deeptaf
        self.menu_urvdeepdtaf = MenuURVDEEPTAF(self.parent)
        self.addMenu(self.menu_urvdeepdtaf)

        # Menu Cheapnet

        self.menu_cheapnet = cheapnet_menu(self.parent)
        self.addMenu(self.menu_cheapnet)

        # Menu gign
        self.menu_gign = gign_menu(self.parent)
        self.addMenu(self.menu_gign)
