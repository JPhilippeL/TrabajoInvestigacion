"""
@file results_page.py
@author Mohamed EL BOUKHIARI
@brief Results page for the Molecular Analysis System GUI.
@details
This page displays detected experiment outputs, parsed metrics and direct links
to result folders.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.results_controller import ExperimentRecord, ResultsController, ResultsSummary
from ui.utils.resources import PROJECT_ROOT
from ui.widgets.metric_card import MetricCard
from ui.widgets.result_file_card import ResultFileCard
from ui.widgets.section_title import SectionTitle


class ResultsPage(QWidget):
    """
    @brief Page used to inspect generated result files and experiment metrics.
    """

    def __init__(self) -> None:
        """
        @brief Initialize the results page.
        """
        super().__init__()

        self.setObjectName("ResultsPage")
        self.results_controller = ResultsController(PROJECT_ROOT)
        self.current_records: list[ExperimentRecord] = []
        self.record_paths = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(26, 24, 26, 20)
        root_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)

        title = QLabel("Results")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Inspect generated experiment files, parsed metrics summaries and best configurations."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        self.export_button = QPushButton("Export summary")
        self.export_button.clicked.connect(self.export_summary)

        self.open_selected_button = QPushButton("Open selected folder")
        self.open_selected_button.clicked.connect(self.open_selected_folder)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(self.refresh_results)

        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        header_layout.addWidget(self.open_selected_button)
        header_layout.addWidget(self.export_button)
        header_layout.addWidget(self.refresh_button)

        root_layout.addLayout(header_layout)

        self.metric_grid = QGridLayout()
        self.metric_grid.setHorizontalSpacing(14)
        self.metric_grid.setVerticalSpacing(14)

        self.best_rmse_card = MetricCard("Best RMSE", "-", "Lower is better.")
        self.best_pearson_card = MetricCard("Best Pearson", "-", "Higher is better.")
        self.best_spearman_card = MetricCard("Best Spearman", "-", "Higher is better.")
        self.total_trials_card = MetricCard("Trials", "-", "Parsed experiment rows.")
        self.total_files_card = MetricCard("Result files", "-", "Detected CSV/YAML outputs.")

        self.metric_grid.addWidget(self.best_rmse_card, 0, 0)
        self.metric_grid.addWidget(self.best_pearson_card, 0, 1)
        self.metric_grid.addWidget(self.best_spearman_card, 0, 2)
        self.metric_grid.addWidget(self.total_trials_card, 0, 3)
        self.metric_grid.addWidget(self.total_files_card, 0, 4)

        root_layout.addLayout(self.metric_grid)

        root_layout.addWidget(SectionTitle("Experiment summary"))

        self.results_table = QTableWidget()
        self.results_table.setObjectName("ResultsTable")
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels(
            ["Model", "Source", "RMSE", "Pearson", "Spearman", "Status", "Trials"]
        )
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)

        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        root_layout.addWidget(self.results_table)

        root_layout.addWidget(SectionTitle("Discovered result files"))

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("ResultsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.results_container = QWidget()
        self.results_container.setObjectName("ResultsContent")

        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 10, 0)
        self.results_layout.setSpacing(12)

        self.scroll_area.setWidget(self.results_container)
        root_layout.addWidget(self.scroll_area, 1)

        self.refresh_results()

    def refresh_results(self) -> None:
        """
        @brief Refresh discovered result files and parsed metrics.

        @return None.
        """
        summary = self.results_controller.build_summary()
        self.current_records = summary.records

        self._update_metric_cards(summary)
        self._populate_results_table(summary.records)
        self._populate_result_files(summary)

    def export_summary(self) -> None:
        """
        @brief Export the current experiment summary table to a CSV file.

        @return None.
        """
        if not self.current_records:
            QMessageBox.information(
                self,
                "No results to export",
                "No parsed experiment records are available for export.",
            )
            return

        try:
            export_path = self.results_controller.export_summary_csv(self.current_records)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not export the results summary:\n{str(error)}",
            )
            return

        QMessageBox.information(
            self,
            "Export completed",
            f"Results summary exported successfully:\n{export_path}",
        )

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(export_path.parent)))

    def open_selected_folder(self) -> None:
        """
        @brief Open the folder containing the selected result table entry.

        @return None.
        """
        selected_row = self.results_table.currentRow()

        if selected_row < 0 or selected_row >= len(self.record_paths):
            QMessageBox.information(
                self,
                "No row selected",
                "Select one experiment row before opening its folder.",
            )
            return

        path = self.record_paths[selected_row]
        folder = path.parent if path.is_file() else path

        if not folder.exists():
            QMessageBox.warning(
                self,
                "Folder not found",
                f"The folder does not exist:\n{folder}",
            )
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _update_metric_cards(self, summary: ResultsSummary) -> None:
        """
        @brief Update metric cards from a results summary.

        @param summary Results summary.
        @return None.
        """
        self.best_rmse_card.set_value(self._format_metric(summary.best_rmse))
        self.best_pearson_card.set_value(self._format_metric(summary.best_pearson))
        self.best_spearman_card.set_value(self._format_metric(summary.best_spearman))
        self.total_trials_card.set_value(str(summary.total_trials))
        self.total_files_card.set_value(
            str(summary.total_files),
            f"{summary.csv_files} CSV / {summary.yaml_files} YAML",
        )

    def _populate_results_table(self, records: list[ExperimentRecord]) -> None:
        """
        @brief Populate the experiment summary table.

        @param records Parsed experiment records.
        @return None.
        """
        self.results_table.setRowCount(len(records))
        self.record_paths = [record.path for record in records]

        for row_index, record in enumerate(records):
            values = [
                record.model,
                record.source_file,
                self._format_metric(record.rmse),
                self._format_metric(record.pearson),
                self._format_metric(record.spearman),
                record.status,
                str(record.trials),
            ]

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(record.path))

                if column_index in {2, 3, 4, 6}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                self.results_table.setItem(row_index, column_index, item)

    def _populate_result_files(self, summary: ResultsSummary) -> None:
        """
        @brief Populate the discovered result files section.

        @param summary Results summary.
        @return None.
        """
        self._clear_results_layout()

        if not summary.result_files:
            empty_label = QLabel(
                "No result files were detected yet. Run a training, testing or hyperparameter "
                "search workflow, then refresh this page."
            )
            empty_label.setObjectName("EmptyResultsLabel")
            empty_label.setWordWrap(True)
            self.results_layout.addWidget(empty_label)
            self.results_layout.addStretch(1)
            return

        for file_info in summary.result_files:
            card = ResultFileCard(
                title=f"{file_info.model} — {file_info.path.name}",
                path=file_info.path,
                description=file_info.description,
            )
            self.results_layout.addWidget(card)

        self.results_layout.addStretch(1)

    def _clear_results_layout(self) -> None:
        """
        @brief Remove all widgets from the result files layout.

        @return None.
        """
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def _format_metric(self, value: float | None) -> str:
        """
        @brief Format a metric value for display.

        @param value Metric value.
        @return Formatted metric text.
        """
        if value is None:
            return "-"

        return f"{value:.4f}"
