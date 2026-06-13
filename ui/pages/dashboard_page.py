"""
@file dashboard_page.py
@author Mohamed EL BOUKHIARI
@brief Main dashboard page for the Molecular Analysis System GUI.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.module_card import ModuleCard
from ui.widgets.section_title import SectionTitle


class DashboardPage(QWidget):
    """
    Home dashboard containing the main application workflows.
    """

    def __init__(self, callbacks: dict[str, Callable[[], None]]) -> None:
        super().__init__()

        self.callbacks = callbacks
        self.setObjectName("DashboardPage")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(26, 24, 26, 20)
        root_layout.setSpacing(12)

        title = QLabel("Dashboard")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Central access point for molecule tools, GNN training, specialized models, "
            "hyperparameter search, explainability and results."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)
        root_layout.addSpacing(8)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("DashboardScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("DashboardContent")
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 10, 0)
        content_layout.setSpacing(22)

        content_layout.addWidget(SectionTitle("Quick workflows"))

        quick_grid = QGridLayout()
        quick_grid.setContentsMargins(0, 0, 0, 0)
        quick_grid.setHorizontalSpacing(18)
        quick_grid.setVerticalSpacing(18)

        self._add_card(
            quick_grid,
            0,
            0,
            "Molecule tools",
            "Create, load and verify molecular structures before model execution.",
            [
                ("New", self.callbacks.get("molecule_new")),
                ("Load", self.callbacks.get("molecule_load")),
                ("Save", self.callbacks.get("molecule_save")),
                ("Verify", self.callbacks.get("verify_molecule")),
            ],
        )

        self._add_card(
            quick_grid,
            0,
            1,
            "SDF utilities",
            "Prepare input molecular files for training, testing and prediction workflows.",
            [
                ("Split SDF", self.callbacks.get("split_sdf")),
                ("CSV to SDF", self.callbacks.get("csv_to_sdf")),
            ],
        )

        self._add_card(
            quick_grid,
            0,
            2,
            "GNN training",
            "Train single or multiple GNN models using the configured datasets.",
            [
                ("Train", self.callbacks.get("train_model")),
                ("Batch", self.callbacks.get("train_multiple")),
            ],
        )

        self._add_card(
            quick_grid,
            1,
            0,
            "Transfer learning",
            "Run transfer learning workflows for one model or multiple models.",
            [
                ("Transfer", self.callbacks.get("transfer_learning")),
                ("Batch transfer", self.callbacks.get("transfer_learning_multiple")),
            ],
        )

        self._add_card(
            quick_grid,
            1,
            1,
            "GNN testing",
            "Run prediction, testing and model inspection workflows.",
            [
                ("Predict SDF", self.callbacks.get("predict_sdf")),
                ("Test model", self.callbacks.get("test_model")),
                ("Test all", self.callbacks.get("test_all_models")),
                ("Inspect", self.callbacks.get("inspect_model")),
            ],
        )

        self._add_card(
            quick_grid,
            1,
            2,
            "GNN hyperparameter search",
            "Launch model testing and hyperparameter search experiments.",
            [
                ("Search", self.callbacks.get("hyperparameter_search_gnn")),
            ],
        )

        self._add_card(
            quick_grid,
            2,
            0,
            "Explainability",
            "Run graph explainers and compare explanation outputs.",
            [
                ("GraphExplainer", self.callbacks.get("graph_explainer")),
                ("GNNExplainer", self.callbacks.get("gnn_explainer")),
                ("Compare", self.callbacks.get("compare_explainers")),
                ("Batch compare", self.callbacks.get("compare_explainers_batch")),
            ],
        )

        self._add_card(
            quick_grid,
            2,
            1,
            "Results",
            "Evaluate models and inspect generated outputs.",
            [
                ("Open results", self.callbacks.get("open_results_page")),
            ],
        )

        self._add_card(
            quick_grid,
            2,
            2,
            "Settings",
            "Configure dataset paths, model folders, runtime defaults and application resources.",
            [
                ("Open settings", self.callbacks.get("open_settings_page")),
            ],
        )

        quick_grid.setColumnStretch(0, 1)
        quick_grid.setColumnStretch(1, 1)
        quick_grid.setColumnStretch(2, 1)

        content_layout.addLayout(quick_grid)
        content_layout.addSpacing(6)
        content_layout.addWidget(SectionTitle("Specialized models"))

        model_grid = QGridLayout()
        model_grid.setContentsMargins(0, 0, 0, 0)
        model_grid.setHorizontalSpacing(18)
        model_grid.setVerticalSpacing(18)

        self._add_card(
            model_grid,
            0,
            0,
            "URVDEEPTAF",
            "URV-specific workflow for data generation, training and evaluation.",
            [
                ("Generate", self.callbacks.get("urv_generate")),
                ("Train", self.callbacks.get("urv_train")),
                ("Train all", self.callbacks.get("urv_train_all")),
                ("Evaluate", self.callbacks.get("urv_evaluate")),
                ("Evaluate all", self.callbacks.get("urv_evaluate_all")),
            ],
        )

        self._add_card(
            model_grid,
            0,
            1,
            "EGNN",
            "Generate data, train, search hyperparameters and evaluate EGNN models.",
            [
                ("Generate", self.callbacks.get("egnn_generate")),
                ("Train", self.callbacks.get("egnn_train")),
                ("Search", self.callbacks.get("egnn_search")),
                ("Evaluate", self.callbacks.get("egnn_evaluate")),
                ("Evaluate all", self.callbacks.get("egnn_evaluate_all")),
            ],
        )

        self._add_card(
            model_grid,
            0,
            2,
            "EDNN",
            "Generate data, train, search hyperparameters and evaluate EDNN models.",
            [
                ("Generate", self.callbacks.get("ednn_generate")),
                ("Train", self.callbacks.get("ednn_train")),
                ("Search", self.callbacks.get("ednn_search")),
                ("Evaluate", self.callbacks.get("ednn_evaluate")),
                ("Evaluate all", self.callbacks.get("ednn_evaluate_all")),
            ],
        )

        self._add_card(
            model_grid,
            1,
            0,
            "DeepDTA",
            "Launch DeepDTA-specific hyperparameter search workflow.",
            [
                ("Search", self.callbacks.get("deepdta_search")),
            ],
        )

        self._add_card(
            model_grid,
            1,
            1,
            "WideDTA",
            "Launch WideDTA-specific hyperparameter search workflow.",
            [
                ("Search", self.callbacks.get("widedta_search")),
            ],
        )

        self._add_card(
            model_grid,
            1,
            2,
            "DCML",
            "Train, search hyperparameters and evaluate DCML model.",
            [
                ("Train", self.callbacks.get("dcml_train")),
                ("Search", self.callbacks.get("dcml_search")),
                ("Evaluate", self.callbacks.get("dcml_evaluate")),
            ],
        )

        self._add_card(
            model_grid,
            2,
            0,
            "CAPLA",
            "Validate dataset, train, search hyperparameters, evaluate and compare CAPLA model.",
            [
                ("Validate", self.callbacks.get("capla_validate")),
                ("Train", self.callbacks.get("capla_train")),
                ("Search", self.callbacks.get("capla_search")),
                ("Evaluate", self.callbacks.get("capla_evaluate")),
                ("Compare", self.callbacks.get("capla_compare")),
            ],
        )

        self._add_card(
            model_grid,
            2,
            1,
            "DEAttentionDTA",
            "Validate dataset, train, search hyperparameters, evaluate and compare DEAttentionDTA model.",
            [
                ("Validation", self.callbacks.get("deattentiondta_validate")),
                ("Train", self.callbacks.get("deattentiondta_train")),
                ("Search", self.callbacks.get("deattentiondta_search")),
                ("Evaluate", self.callbacks.get("deattentiondta_evaluate")),
                ("Compare", self.callbacks.get("deattentiondta_compare")),
            ],
        )

        model_grid.setColumnStretch(0, 1)
        model_grid.setColumnStretch(1, 1)

        content_layout.addLayout(model_grid)
        content_layout.addStretch(1)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)

    def _add_card(
        self,
        grid: QGridLayout,
        row: int,
        column: int,
        title: str,
        description: str,
        actions: list[tuple[str, Callable[[], None] | None]],
    ) -> None:
        """
        Add a dashboard card to a grid with consistent sizing.

        Args:
            grid: Target grid layout.
            row: Grid row.
            column: Grid column.
            title: Card title.
            description: Card description.
            actions: Card action buttons.
        """
        card = ModuleCard(title, description, actions)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid.addWidget(card, row, column)
