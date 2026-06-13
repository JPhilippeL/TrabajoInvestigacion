"""Shared PySide6 helpers for DEAttentionDTA dialogs."""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


def with_button(line_edit: QLineEdit, button: QPushButton) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit)
    layout.addWidget(button)
    return container


def browse_existing_directory(parent, title: str, target: QLineEdit) -> None:
    path = QFileDialog.getExistingDirectory(parent, title, target.text())
    if path:
        target.setText(path)


def browse_existing_file(parent, title: str, file_filter: str, target: QLineEdit) -> None:
    path, _ = QFileDialog.getOpenFileName(parent, title, target.text(), file_filter)
    if path:
        target.setText(path)
