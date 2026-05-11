"""
@file module_card.py
@author Mohamed EL BOUKHIARI
@brief Reusable dashboard card for modules and actions.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout


class ModuleCard(QFrame):
    """
    Dashboard card containing a title, a description and action buttons.
    """

    def __init__(
        self,
        title: str,
        description: str,
        actions: list[tuple[str, Callable[[], None] | None]],
    ) -> None:
        super().__init__()

        self.setObjectName("Card")
        self.setMinimumHeight(158)
        self.setMaximumHeight(190)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")

        self.description_label = QLabel(description)
        self.description_label.setObjectName("CardDescription")
        self.description_label.setWordWrap(True)
        self.description_label.setMinimumHeight(34)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 8, 0, 0)
        buttons_layout.setSpacing(8)

        for index, (button_text, callback) in enumerate(actions):
            button = QPushButton(button_text)
            button.setMinimumWidth(74)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

            if index == 0:
                button.setObjectName("PrimaryButton")

            if callback is None:
                button.setEnabled(False)
            else:
                button.clicked.connect(callback)

            buttons_layout.addWidget(button)

        buttons_layout.addStretch(1)

        layout.addWidget(self.title_label)
        layout.addWidget(self.description_label)
        layout.addStretch(1)
        layout.addLayout(buttons_layout)
