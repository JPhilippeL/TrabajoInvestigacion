"""
@file app_settings.py
@author Mohamed EL BOUKHIARI
@brief Persistent application settings manager.
@details
This module centralizes persistent GUI settings such as dataset paths,
model output paths, runtime defaults and application resource paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from ui.utils.resources import PROJECT_ROOT, logo_path, icon_path


class AppSettings:
    """
    @brief Wrapper around QSettings for the Molecular Analysis System.
    """

    ORGANIZATION_NAME = "URV"
    APPLICATION_NAME = "Molecular Analysis System"

    DEFAULTS = {
        "paths/dataset_root": "",
        "paths/ligand_sdf": "",
        "paths/protein_pdb": "",
        "paths/pic50_file": "",
        "paths/splits_folder": "",
        "paths/egnn_root": str(PROJECT_ROOT / "EGNN"),
        "paths/ednn_root": str(PROJECT_ROOT / "EDNN"),
        "paths/deepdta_root": str(PROJECT_ROOT / "DeepDTA"),
        "paths/widedta_root": str(PROJECT_ROOT / "WideDTA"),
        "paths/exports_dir": str(PROJECT_ROOT / "exports"),
        "runtime/default_device": "auto",
        "runtime/default_seed": "42",
        "appearance/urv_logo": str(logo_path("urv_logo.png")),
        "appearance/app_logo": str(logo_path("app_logo.png")),
        "appearance/app_icon": str(icon_path("app_icon.png")),
    }

    def __init__(self) -> None:
        """
        @brief Initialize the persistent settings storage.
        """
        self.settings = QSettings(self.ORGANIZATION_NAME, self.APPLICATION_NAME)

    def get_value(self, key: str) -> str:
        """
        @brief Return a setting value as a string.

        @param key Setting key.
        @return Stored value or default value.
        """
        default_value = self.DEFAULTS.get(key, "")
        value = self.settings.value(key, default_value)

        if value is None:
            return str(default_value)

        return str(value)

    def set_value(self, key: str, value: Any) -> None:
        """
        @brief Store a setting value.

        @param key Setting key.
        @param value Value to store.
        @return None.
        """
        self.settings.setValue(key, str(value))

    def reset_defaults(self) -> None:
        """
        @brief Reset all managed settings to their default values.

        @return None.
        """
        for key, value in self.DEFAULTS.items():
            self.settings.setValue(key, str(value))

    def sync(self) -> None:
        """
        @brief Synchronize settings to disk.

        @return None.
        """
        self.settings.sync()

    def get_path(self, key: str) -> Path:
        """
        @brief Return a setting value as a Path.

        @param key Setting key.
        @return Path object.
        """
        return Path(self.get_value(key)).expanduser()
