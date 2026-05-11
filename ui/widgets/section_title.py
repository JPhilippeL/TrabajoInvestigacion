"""
@file section_title.py
@author Mohamed EL BOUKHIARI
@brief Reusable section title widget.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel


class SectionTitle(QLabel):
    """
    Small reusable title used to separate dashboard sections.
    """

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("SectionTitle")
