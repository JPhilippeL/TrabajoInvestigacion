"""
@file resources.py
@author Mohamed EL BOUKHIARI
@brief Resource path helpers for GUI assets.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = PROJECT_ROOT / "ui"
ASSETS_DIR = UI_DIR / "assets"
LOGOS_DIR = ASSETS_DIR / "logos"
ICONS_DIR = ASSETS_DIR / "icons"


def logo_path(filename: str) -> Path:
    """
    Return the absolute path of a logo asset.

    Args:
        filename: Logo filename.

    Returns:
        Absolute logo path.
    """
    return LOGOS_DIR / filename


def icon_path(filename: str) -> Path:
    """
    Return the absolute path of an icon asset.

    Args:
        filename: Icon filename.

    Returns:
        Absolute icon path.
    """
    return ICONS_DIR / filename
