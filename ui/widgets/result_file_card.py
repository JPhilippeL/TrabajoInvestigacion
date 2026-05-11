"""
@file result_file_card.py
@author Mohamed EL BOUKHIARI
@brief Reusable card widget for discovered result files.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ResultFileCard(QFrame):
    """
    @brief Card used to display a result file and open its containing folder.
    """

    def __init__(self, title: str, path: Path, description: str = "") -> None:
        """
        @brief Initialize the result file card.

        @param title Display title.
        @param path Result file path.
        @param description Short file description.
        """
        super().__init__()

        self.path = path
        self.setObjectName("ResultFileCard")
        self.setMinimumHeight(96)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(12)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("ResultFileTitle")

        self.description_label = QLabel(description)
        self.description_label.setObjectName("ResultFileDescription")
        self.description_label.setWordWrap(True)

        self.path_label = QLabel(str(path))
        self.path_label.setObjectName("ResultFilePath")
        self.path_label.setWordWrap(True)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.description_label)
        text_layout.addWidget(self.path_label)

        self.open_button = QPushButton("Open folder")
        self.open_button.clicked.connect(self.open_folder)

        root_layout.addLayout(text_layout, 1)
        root_layout.addWidget(self.open_button)

    def open_folder(self) -> None:
        """
        @brief Open the folder containing the result file.
        """
        folder = self.path.parent if self.path.is_file() else self.path

        if folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
