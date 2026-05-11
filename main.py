"""
@file main.py
@author Mohamed EL BOUKHIARI
@brief Entry point of the Molecular Analysis System GUI.
@details
This file initializes the execution environment, configures multiprocessing,
loads the Qt application, applies the graphical theme and starts the main
application window.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Limit BLAS/OpenMP backend threads before importing scientific libraries.
# This avoids excessive CPU usage and reduces multiprocessing conflicts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.utils.resources import icon_path
from ui.utils.theme_loader import apply_theme


def configure_multiprocessing() -> None:
    """
    @brief Configure the multiprocessing start method.
    @details
    The spawn method is safer for GUI applications and PyTorch-based workers,
    because child processes start from a clean interpreter instead of inheriting
    the full state of the parent process.

    @return None.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # The multiprocessing start method was already configured.
        pass


def create_application() -> QApplication:
    """
    @brief Create and configure the Qt application instance.
    @details
    This function initializes QApplication, configures desktop integration,
    sets the application icon and applies the global light theme.

    @return Configured QApplication instance.
    """
    app = QApplication(sys.argv)

    # Must match the .desktop filename without the ".desktop" extension.
    app.setDesktopFileName("molecular-analysis-system")

    app.setApplicationName("Molecular Analysis System")
    app.setWindowIcon(QIcon(str(icon_path("app_icon.png"))))

    apply_theme(app, "light")

    return app


def main() -> None:
    """
    @brief Launch the Molecular Analysis System GUI.
    @details
    This function configures multiprocessing, creates the Qt application,
    initializes the main window and starts the Qt event loop.

    @return None.
    """
    configure_multiprocessing()

    app = create_application()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
