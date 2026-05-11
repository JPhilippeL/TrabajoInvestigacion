"""
@file theme_loader.py
@author Mohamed EL BOUKHIARI
@brief Theme loading utilities for the graphical interface.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication


UI_DIR = Path(__file__).resolve().parents[1]
THEMES_DIR = UI_DIR / "themes"


def load_stylesheet(theme_name: str = "light") -> str:
    """
    Load a QSS stylesheet from the ui/themes directory.

    Args:
        theme_name: Name of the theme without extension.

    Returns:
        Stylesheet content as a string.

    Raises:
        FileNotFoundError: If the requested theme does not exist.
    """
    theme_path = THEMES_DIR / f"{theme_name}.qss"

    if not theme_path.exists():
        raise FileNotFoundError(f"Theme file not found: {theme_path}")

    return theme_path.read_text(encoding="utf-8")


def apply_theme(app: QApplication, theme_name: str = "light") -> None:
    """
    Apply a QSS theme to the QApplication instance.

    Args:
        app: Current QApplication.
        theme_name: Name of the theme to apply.
    """
    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet(theme_name))
