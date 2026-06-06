"""
@file main_window.py
@author Mohamed EL BOUKHIARI
@brief Main application window for the Molecular Analysis System GUI.
"""

from __future__ import annotations

import logging

import networkx as nx

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from graph_managment.sdf_converter import parse_sdf
from GNNs.controllers.hyperparameter_search_controller_process import (
    HyperparameterSearchControllerProcess,
)
from GNNs.controllers.testing_controller_process import TestingControllerProcess
from GNNs.controllers.training_controller_process import TrainingControllerProcess
from ui.graph_interface.graph_view import MoleculeGraphView
from ui.menu_bar import MenuBar
from ui.pages.dashboard_page import DashboardPage
from ui.pages.results_page import ResultsPage
from ui.utils.logger import QtHandler
from ui.widgets.app_header import AppHeader
from ui.utils.resources import logo_path
from ui.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    """
    Main window of the Molecular Analysis System application.
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Molecular Analysis System")
        self.resize(1650, 820)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Hidden internal menu bar.
        # It is used only as an action registry for dashboard buttons.
        # It must not be parented to QMainWindow, otherwise Qt may still render it.
        self.menu_bar = MenuBar(self)
        self.menu_bar.setParent(None)
        self.menu_bar.hide()
        self.setMenuWidget(None)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.app_header = AppHeader(
            title="Molecular Analysis System",
            subtitle="Protein-ligand binding affinity prediction platform",
            urv_logo_path=logo_path("urv_logo.png"),
            app_logo_path=logo_path("app_logo.png"),
            device_text="Device: Auto",
            status_text="Ready",
        )
        self.app_header.home_requested.connect(self.show_dashboard)
        self.main_layout.addWidget(self.app_header)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self.central_widget)
        self.main_layout.addWidget(self.splitter)

        self.dashboard_page = DashboardPage(callbacks=self.build_dashboard_callbacks())
        self.splitter.addWidget(self.dashboard_page)
        self.results_page = ResultsPage()
        self.settings_page = SettingsPage()

        self.log_output = QTextEdit()
        self.log_output.setObjectName("LogConsole")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("System messages...")
        self.splitter.addWidget(self.log_output)

        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([560, 140])

        self.qt_handler = QtHandler(self.log)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.qt_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(self.qt_handler)
        logger.setLevel(logging.DEBUG)

        self.training_controller = TrainingControllerProcess(self)
        self.testing_controller = TestingControllerProcess(self)
        self.hyperparameter_search_controller = HyperparameterSearchControllerProcess(self)

        logging.info("Application initialized.")

    def build_dashboard_callbacks(self) -> dict[str, object]:
        """
        Build dashboard callbacks by reusing existing menu actions.

        Returns:
            Dictionary mapping dashboard action names to callables.
        """
        return {
            "molecule_new": lambda: self.trigger_menu_action("Molecule", "New"),
            "molecule_load": lambda: self.trigger_menu_action("Molecule", "Load"),
            "molecule_save": lambda: self.trigger_menu_action("Molecule", "Save"),
            "verify_molecule": lambda: self.trigger_menu_action("Molecule", "Verify Molecule"),
            "split_sdf": lambda: self.trigger_menu_action("Molecule", "Split SDF"),
            "csv_to_sdf": lambda: self.trigger_menu_action("Molecule", "CSV to SDF"),

            "train_model": lambda: self.trigger_menu_action("Train GNN", "Train Model"),
            "train_multiple": lambda: self.trigger_menu_action("Train GNN", "Train Multiple Models"),

            "transfer_learning": lambda: self.trigger_menu_action("Transfer GNN", "Transfer Learning"),
            "transfer_learning_multiple": lambda: self.trigger_menu_action("Transfer GNN","Transfer Learning - Multiple Models",),

            "predict_sdf": lambda: self.trigger_menu_action("Test GNN", "Predict SDF"),
            "test_model": lambda: self.trigger_menu_action("Test GNN", "Test Model"),
            "test_all_models": lambda: self.trigger_menu_action("Test GNN", "Test All Models"),
            "inspect_model": lambda: self.trigger_menu_action("Test GNN", "Inspect Model"),

            "open_results_page": self.show_results_page,
            "open_settings_page": self.show_settings_page,

            "hyperparameter_search_gnn": lambda: self.trigger_menu_action(
                "Hyperparameter Search GNN",
                "Run Hyperparameter Search",
            ),

            "graph_explainer": lambda: self.trigger_menu_action("Explainer GNN", "Run GraphExplainer"),
            "gnn_explainer": lambda: self.trigger_menu_action("Explainer GNN","Run GNNExplainer",),
            "compare_explainers": lambda: self.trigger_menu_action("Explainer GNN", "Compare Explainers"),
            "compare_explainers_batch": lambda: self.trigger_menu_action("Explainer GNN","Compare Explainers Batch",),

            "urv_generate": lambda: self.trigger_menu_action("URVDEEPTAF", "Generate Data"),
            "urv_train": lambda: self.trigger_menu_action("URVDEEPTAF", "Train Model"),
            "urv_train_all": lambda: self.trigger_menu_action("URVDEEPTAF", "Train All Models"),
            "urv_evaluate": lambda: self.trigger_menu_action("URVDEEPTAF", "Evaluate Model"),
            "urv_evaluate_all": lambda: self.trigger_menu_action("URVDEEPTAF","Evaluate All Models (Folder)",),

            "egnn_generate": lambda: self.trigger_menu_action("EGNN", "Generate Data"),
            "egnn_train": lambda: self.trigger_menu_action("EGNN", "Train Model"),
            "egnn_search": lambda: self.trigger_menu_action("EGNN", "Hyperparameter Search"),
            "egnn_evaluate": lambda: self.trigger_menu_action("EGNN", "Evaluate Model"),
            "egnn_evaluate_all": lambda: self.trigger_menu_action("EGNN", "Evaluate All Models"),

            "ednn_generate": lambda: self.trigger_menu_action("EDNN", "Generate Data"),
            "ednn_train": lambda: self.trigger_menu_action("EDNN", "Train Model"),
            "ednn_search": lambda: self.trigger_menu_action("EDNN", "Hyperparameter Search"),
            "ednn_evaluate": lambda: self.trigger_menu_action("EDNN", "Evaluate Model"),
            "ednn_evaluate_all": lambda: self.trigger_menu_action("EDNN", "Evaluate All Models"),

            "deepdta_search": lambda: self.trigger_menu_action("DeepDTA", "Hyperparameter Search"),
            "widedta_search": lambda: self.trigger_menu_action("WideDTA", "Hyperparameter Search"),

            "dcml_train": lambda: self.trigger_menu_action("DCML", "Train Model"),
            "dcml_search": lambda: self.trigger_menu_action("DCML", "Hyperparameter Search"),
            "dcml_evaluate": lambda: self.trigger_menu_action("DCML", "Evaluate Model"),

            "capla_validate": lambda: self.trigger_menu_action("CAPLA", "Validate Prepared Dataset"),
            "capla_train": lambda: self.trigger_menu_action("CAPLA", "Train Official Splits From Scratch"),
            "capla_search": lambda: self.trigger_menu_action("CAPLA", "Hyperparameter Search"),
            "capla_evaluate": lambda: self.trigger_menu_action("CAPLA", "Evaluate Model on Prepared Dataset"),
            "capla_compare": lambda: self.trigger_menu_action("CAPLA", "Compare Pretrained vs Fine-Tuned"),
        }

    def trigger_menu_action(self, menu_title: str, action_text: str) -> None:
        """
        Trigger an existing menu action by top-level menu title and action text.

        Args:
            menu_title: Top-level menu title.
            action_text: Action text inside the menu.
        """
        def clean(text: str) -> str:
            return text.replace("&", "").strip()

        for top_action in self.menu_bar.actions():
            if clean(top_action.text()) != menu_title:
                continue

            menu = top_action.menu()

            if menu is None:
                break

            for action in menu.actions():
                if action.isSeparator():
                    continue

                if clean(action.text()) == action_text:
                    logging.info("Dashboard action triggered: %s > %s", menu_title, action_text)
                    action.trigger()
                    return

        QMessageBox.warning(
            self,
            "Action not found",
            f"The action could not be found:\n{menu_title} > {action_text}",
        )
        logging.warning("Menu action not found: %s > %s", menu_title, action_text)

    def show_dashboard(self) -> None:
        """
        Return to the main dashboard page.
        """
        current_widget = self.splitter.widget(0)

        if current_widget is self.dashboard_page:
            return

        self.clear_content_area()
        self.splitter.insertWidget(0, self.dashboard_page)

        self.app_header.set_status("ready", "Ready")
        logging.info("Returned to dashboard.")

    def show_results_page(self) -> None:
        """
        @brief Display the central results page.
        @details
        The page is refreshed before being displayed to detect newly generated
        experiment outputs.

        @return None.
        """
        current_widget = self.splitter.widget(0)

        if current_widget is self.results_page:
            self.results_page.refresh_results()
            return

        self.results_page.refresh_results()
        self.clear_content_area()
        self.splitter.insertWidget(0, self.results_page)

        self.app_header.set_status("ready", "Results")
        logging.info("Opened results page.")

    def show_settings_page(self) -> None:
        """
        @brief Display the application settings page.
        @details
        The settings page centralizes persistent paths, runtime defaults and
        application resource locations.

        @return None.
        """
        current_widget = self.splitter.widget(0)

        if current_widget is self.settings_page:
            return

        self.clear_content_area()
        self.splitter.insertWidget(0, self.settings_page)

        self.app_header.set_status("ready", "Settings")
        logging.info("Opened settings page.")

    def create_new_graph(self) -> None:
        """
        Create a new empty molecule graph and display it in the graph viewer.
        """
        graph = nx.Graph()
        new_graph_view = MoleculeGraphView(graph)

        self.graph_view = new_graph_view
        self.clear_content_area()

        self.splitter.insertWidget(0, self.graph_view)
        self.app_header.set_status("ready", "Graph ready")

        QMessageBox.information(
            self,
            "Tip",
            "Right-click on the area to add a node.",
        )

    def load_graph_from_file(self, file_path: str) -> None:
        """
        Load a molecule graph from an SDF file and display it.

        Args:
            file_path: Path to the SDF file.
        """
        if not file_path:
            return

        try:
            graph = parse_sdf(file_path)
        except Exception as error:
            self.app_header.set_status("error", "Load failed")
            QMessageBox.critical(
                self,
                "Loading error",
                f"Could not load the file:\n{str(error)}",
            )
            logging.exception("Could not load graph from file.")
            return

        new_graph_view = MoleculeGraphView(graph)

        self.clear_content_area()
        self.graph_view = new_graph_view
        self.splitter.insertWidget(0, self.graph_view)

        self.app_header.set_status("success", "Graph loaded")
        logging.info("Graph loaded from file: %s", file_path)

    def clear_content_area(self) -> None:
        """
        Remove the current main content widget from the splitter.
        """
        widget = self.splitter.widget(0)

        if widget is not None:
            widget.setParent(None)

    def log(self, message: str) -> None:
        """
        Append a log message to the GUI log area.

        Args:
            message: Message to display.
        """
        self.log_output.append(message)

        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()
