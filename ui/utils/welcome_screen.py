from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

class WelcomeScreen(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        image_label = QLabel()
        # Cargar la imagen como QImage
        image = QImage("assets/icono3.png")

        # Invertir los colores
        image.invertPixels()

        # Convertir a QPixmap y escalar
        pixmap = QPixmap.fromImage(image).scaledToWidth(300, Qt.SmoothTransformation)

        # Mostrar en el QLabel
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        text = QLabel("Sistema de Análisis Molecular")
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("font-size: 20px; font-weight: bold;")

        layout.addWidget(image_label)
        layout.addWidget(text)
        self.setLayout(layout)
