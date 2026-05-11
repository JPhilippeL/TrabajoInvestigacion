"""
@file path_selector.py
@author Mohamed EL BOUKHIARI
@brief Reusable path selection widget.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class PathSelector(QWidget):
    """
    @brief Composite widget containing a path field and a browse button.
    """

    def __init__(
        self,
        mode: str = "directory",
        dialog_title: str = "Select path",
        file_filter: str = "All files (*)",
    ) -> None:
        """
        @brief Initialize the path selector.

        @param mode Selection mode. Supported values: directory, file.
        @param dialog_title Dialog title.
        @param file_filter File filter used in file mode.
        """
        super().__init__()

        self.mode = mode
        self.dialog_title = dialog_title
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select path...")

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)

        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_button)

    def text(self) -> str:
        """
        @brief Return the current path text.

        @return Current path.
        """
        return self.path_edit.text().strip()

    def set_text(self, value: str) -> None:
        """
        @brief Set the current path text.

        @param value Path text.
        @return None.
        """
        self.path_edit.setText(value)

    def browse(self) -> None:
        """
        @brief Open a QFileDialog and update the path field.

        @return None.
        """
        current_text = self.text()
        start_dir = str(Path(current_text).parent if current_text else Path.home())

        if self.mode == "file":
            selected_path, _ = QFileDialog.getOpenFileName(
                self,
                self.dialog_title,
                start_dir,
                self.file_filter,
            )
        else:
            selected_path = QFileDialog.getExistingDirectory(
                self,
                self.dialog_title,
                start_dir,
            )

        if selected_path:
            self.set_text(selected_path)
