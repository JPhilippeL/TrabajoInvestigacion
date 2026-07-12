"""Application-wide light Qt stylesheet for the desktop GUI."""

from __future__ import annotations


def application_stylesheet() -> str:
    return """
    QWidget {
        background-color: #f5f7fb;
        color: #172033;
        font-family: "Segoe UI", "Inter", "Arial", sans-serif;
        font-size: 10.5pt;
        selection-background-color: #2563eb;
        selection-color: #ffffff;
    }

    QMainWindow,
    QDialog {
        background-color: #f5f7fb;
    }

    QMenuBar {
        background-color: #ffffff;
        color: #172033;
        border-bottom: 1px solid #d9e1ec;
        spacing: 8px;
        padding: 5px 8px;
    }

    QMenuBar::item {
        background: transparent;
        padding: 8px 11px;
        border-radius: 5px;
    }

    QMenuBar::item:selected,
    QMenuBar::item:pressed {
        background-color: #eaf1ff;
        color: #0f3f99;
    }

    QMenu {
        background-color: #ffffff;
        color: #172033;
        border: 1px solid #d7dfeb;
        padding: 6px;
    }

    QMenu::item {
        padding: 8px 30px 8px 14px;
        border-radius: 4px;
    }

    QMenu::item:selected {
        background-color: #eaf1ff;
        color: #0f3f99;
    }

    QLabel {
        background: transparent;
        color: #172033;
    }

    QGroupBox {
        background-color: #ffffff;
        border: 1px solid #d7dfeb;
        border-radius: 7px;
        margin-top: 18px;
        padding: 16px 12px 12px 12px;
        font-weight: 600;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
        color: #26344d;
    }

    QPushButton {
        background-color: #ffffff;
        color: #172033;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 7px 13px;
        min-height: 24px;
        font-weight: 600;
    }

    QPushButton:hover {
        background-color: #f1f6ff;
        border-color: #93b4e8;
        color: #0f3f99;
    }

    QPushButton:pressed {
        background-color: #e4eefc;
    }

    QPushButton:default,
    QPushButton#primaryButton,
    QPushButton[buttonRole="primary"] {
        background-color: #2563eb;
        border-color: #2563eb;
        color: #ffffff;
    }

    QPushButton:default:hover,
    QPushButton#primaryButton:hover,
    QPushButton[buttonRole="primary"]:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
        color: #ffffff;
    }

    QPushButton:disabled {
        background-color: #f3f6fa;
        border-color: #d8e0eb;
        color: #8a97aa;
    }

    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QComboBox {
        background-color: #ffffff;
        color: #172033;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 7px 9px;
        min-height: 24px;
    }

    QLineEdit:focus,
    QTextEdit:focus,
    QPlainTextEdit:focus,
    QComboBox:focus {
        border-color: #2563eb;
    }

    QTextEdit,
    QPlainTextEdit {
        font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
        line-height: 1.25;
    }

    QTextEdit#mainLogOutput {
        background-color: #ffffff;
        border-top: 1px solid #d9e1ec;
        border-left: 0;
        border-right: 0;
        border-bottom: 0;
        border-radius: 0;
        color: #243047;
    }

    QComboBox::drop-down {
        border: 0;
        width: 26px;
    }

    QComboBox QAbstractItemView {
        background-color: #ffffff;
        color: #172033;
        border: 1px solid #cbd5e1;
        selection-background-color: #eaf1ff;
        selection-color: #0f3f99;
    }

    QCheckBox {
        spacing: 8px;
        background: transparent;
    }

    QCheckBox::indicator:unchecked {
        width: 16px;
        height: 16px;
        border: 1px solid #9ca3af;
        border-radius: 3px;
        background-color: #ffffff;
    }

    QCheckBox::indicator:unchecked:hover {
        border-color: #2563eb;
    }

    QCheckBox::indicator:unchecked:disabled {
        background-color: #f3f4f6;
        border-color: #d1d5db;
    }

    QScrollArea {
        background-color: #f5f7fb;
        border: 0;
    }

    QScrollArea > QWidget > QWidget {
        background-color: #f5f7fb;
    }

    QScrollBar:vertical {
        background: #eef2f7;
        width: 12px;
        margin: 0;
    }

    QScrollBar::handle:vertical {
        background: #c5cfdd;
        min-height: 28px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical:hover {
        background: #a9b6c8;
    }

    QSplitter::handle {
        background-color: #d9e1ec;
    }

    QToolTip {
        background-color: #172033;
        color: #ffffff;
        border: 1px solid #172033;
        padding: 6px;
    }

    QMessageBox QLabel {
        color: #172033;
    }

    QWidget#dashboard {
        background-color: #f5f7fb;
    }

    QFrame#dashboardHeader,
    QFrame#dashboardCard {
        background-color: #ffffff;
        border: 1px solid #dce3ee;
        border-radius: 10px;
    }

    QLabel#dashboardTitle {
        color: #101828;
        font-size: 24pt;
        font-weight: 750;
    }

    QLabel#dashboardSubtitle,
    QLabel#dashboardDescription,
    QLabel#cardDescription {
        color: #667085;
        font-size: 10.5pt;
        line-height: 1.35;
    }

    QLabel#pageTitle {
        color: #101828;
        font-size: 20pt;
        font-weight: 750;
    }

    QLabel#sectionTitle {
        color: #1d2939;
        font-size: 13pt;
        font-weight: 750;
        padding-top: 6px;
    }

    QLabel#cardTitle {
        color: #172033;
        font-size: 12.5pt;
        font-weight: 700;
    }

    QFrame#logoPlaceholder {
        background-color: transparent;
        border: none;
    }

    QLabel#logoPlaceholderText {
        color: #667085;
        font-size: 8.5pt;
        font-weight: 700;
        letter-spacing: 0;
    }

    QLabel#statusBadge {
        background-color: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 9pt;
        font-weight: 700;
    }

    QLabel#deviceBadge {
        background-color: #eef4ff;
        color: #175cd3;
        border: 1px solid #b2ccff;
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 9pt;
        font-weight: 700;
    }

    QPushButton#cardButton {
        padding: 6px 8px;
        min-height: 22px;
        font-size: 9.5pt;
    }

    QPushButton#infoButton {
        background-color: #f8fafc;
        color: #2563eb;
        border: 1px solid #bfd0ea;
        border-radius: 13px;
        padding: 0;
        min-height: 24px;
        min-width: 24px;
        font-size: 10pt;
        font-weight: 750;
    }

    QPushButton#infoButton:hover {
        background-color: #eef4ff;
        border-color: #93b4e8;
        color: #1d4ed8;
    }

    QLabel#infoDialogText {
        background-color: #ffffff;
        color: #172033;
        padding: 12px;
        font-size: 10.5pt;
        line-height: 1.35;
    }
    """
