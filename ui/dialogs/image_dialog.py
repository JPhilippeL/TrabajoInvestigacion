# ui/dialogs/image_dialog.py
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QScrollArea
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class ImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scatter Plot")
        self.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.label.setText("Could not load the image.")
        else:
            self.label.setPixmap(pixmap)

        scroll_area.setWidget(self.label)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)

        self.resize(600, 600)

        self.setMinimumSize(300, 300)
