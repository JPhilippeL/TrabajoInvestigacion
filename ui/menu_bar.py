from PySide6.QtWidgets import QMenuBar
import logging
from ui.menus import MenuMolecula, MenuExplainerGNN, MenuTestGNN, MenuTrainGNN, MenuTransferGNN

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