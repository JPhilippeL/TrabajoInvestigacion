"""
@file metric_card.py
@author Mohamed EL BOUKHIARI
@brief Reusable metric card widget for result summaries.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class MetricCard(QFrame):
    """
    @brief Small dashboard card used to display a result metric.
    """

    def __init__(self, title: str, value: str = "-", hint: str = "") -> None:
        """
        @brief Initialize the metric card.

        @param title Metric title.
        @param value Displayed metric value.
        @param hint Short contextual information.
        """
        super().__init__()

        self.setObjectName("MetricCard")
        self.setMinimumHeight(105)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")

        self.hint_label = QLabel(hint)
        self.hint_label.setObjectName("MetricHint")
        self.hint_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.hint_label)

    def set_value(self, value: str, hint: str | None = None) -> None:
        """
        @brief Update the displayed metric value.

        @param value New metric value.
        @param hint Optional updated hint.
        """
        self.value_label.setText(value)

        if hint is not None:
            self.hint_label.setText(hint)
