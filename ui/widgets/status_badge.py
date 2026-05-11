"""
@file status_badge.py
@author Mohamed EL BOUKHIARI
@brief Small reusable status badge widget for the GUI.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class StatusBadge(QLabel):
    """
    Compact QLabel-based badge used to display application or task status.
    """

    def __init__(self, text: str = "Ready", status: str = "ready") -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("StatusBadge")
        self.set_status(status, text)

    def set_status(self, status: str, text: str | None = None) -> None:
        """
        Update badge status and displayed text.
        """
        if text is not None:
            self.setText(text)

        colors = {
            "ready": ("#E8F5E9", "#1B5E20", "#A5D6A7"),
            "running": ("#E3F2FD", "#0B5CAD", "#90CAF9"),
            "success": ("#E8F5E9", "#1B5E20", "#A5D6A7"),
            "warning": ("#FFF8E1", "#8A5A00", "#FFE082"),
            "error": ("#FFEBEE", "#B71C1C", "#EF9A9A"),
        }

        bg, fg, border = colors.get(status, colors["ready"])

        self.setStyleSheet(
            f"""
            QLabel#StatusBadge {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 3px 10px;
                font-weight: 600;
            }}
            """
        )
