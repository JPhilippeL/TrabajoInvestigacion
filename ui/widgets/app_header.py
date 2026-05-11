"""
@file app_header.py
@author Mohamed EL BOUKHIARI
@brief Application header widget for the Molecular Analysis System GUI.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout

from ui.widgets.status_badge import StatusBadge


class AppHeader(QFrame):
    """
    Top application header containing institutional branding, application identity,
    device information and status.
    """

    home_requested = Signal()

    def __init__(
        self,
        title: str = "Molecular Analysis System",
        subtitle: str = "Protein-ligand binding affinity prediction platform",
        urv_logo_path: str | Path | None = None,
        app_logo_path: str | Path | None = None,
        device_text: str = "Device: Auto",
        status_text: str = "Ready",
    ) -> None:
        super().__init__()

        self.setObjectName("AppHeader")
        self.setMinimumHeight(78)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(22, 10, 20, 10)
        main_layout.setSpacing(14)

        self.urv_logo_label = self._create_logo_label(
            logo_path=urv_logo_path,
            fallback_text="URV",
            width=54,
            height=48,
            fallback_color="#0B5CAD",
        )

        self.separator = QFrame()
        self.separator.setObjectName("HeaderSeparator")
        self.separator.setFixedWidth(1)
        self.separator.setMinimumHeight(44)

        self.app_logo_label = self._create_logo_label(
            logo_path=app_logo_path,
            fallback_text="APP",
            width=48,
            height=48,
            fallback_color="#334155",
        )

        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("AppTitle")
        self.title_label.setToolTip("Return to dashboard")

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("AppSubtitle")
        self.subtitle_label.setToolTip("Return to dashboard")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)

        self.device_label = QLabel(device_text)
        self.device_label.setObjectName("HeaderDevice")
        self.device_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_badge = StatusBadge(status_text, "ready")

        main_layout.addWidget(self.urv_logo_label)
        main_layout.addWidget(self.separator)
        main_layout.addWidget(self.app_logo_label)
        main_layout.addLayout(title_layout)
        main_layout.addStretch(1)
        main_layout.addWidget(self.device_label)
        main_layout.addWidget(self.status_badge)

        self.urv_logo_label.installEventFilter(self)
        self.app_logo_label.installEventFilter(self)
        self.title_label.installEventFilter(self)
        self.subtitle_label.installEventFilter(self)

    def _create_logo_label(
        self,
        logo_path: str | Path | None,
        fallback_text: str,
        width: int,
        height: int,
        fallback_color: str,
    ) -> QLabel:
        """
        Create a logo QLabel with a fallback if the image is missing.
        """
        label = QLabel()
        label.setFixedSize(width, height)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setToolTip("Return to dashboard")

        if logo_path is not None and Path(logo_path).exists():
            pixmap = QPixmap(str(logo_path))
            label.setPixmap(
                pixmap.scaled(
                    width,
                    height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            label.setText(fallback_text)
            label.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {fallback_color};
                    color: white;
                    border-radius: 8px;
                    font-weight: 800;
                }}
                """
            )

        return label

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Detect clicks on the logo/title area and request dashboard navigation.
        """
        clickable_widgets = {
            self.urv_logo_label,
            self.app_logo_label,
            self.title_label,
            self.subtitle_label,
        }

        if watched in clickable_widgets and event.type() == QEvent.Type.MouseButtonPress:
            mouse_event = event

            if isinstance(mouse_event, QMouseEvent):
                if mouse_event.button() == Qt.MouseButton.LeftButton:
                    self.home_requested.emit()
                    return True

        return super().eventFilter(watched, event)

    def set_device(self, device_text: str) -> None:
        self.device_label.setText(device_text)

    def set_status(self, status: str, text: str) -> None:
        self.status_badge.set_status(status, text)
