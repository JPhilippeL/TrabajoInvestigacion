from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ACTIONS = ("Generate Data", "Train", "Search", "Evaluate")
DISPLAY_LABELS = {
    "Entrenar Modelo con SDFs": "Train with SDFs",
    "Entrenar Modelo con .pt": "Train with .pt",
    "Entrenar Múltiples Modelos": "Train Multiple Models",
    "Entrenar Múltiples Modelos .pt": "Train Multiple Models .pt",
    "Testear todos los modelos PT": "Test All Models .pt",
    "Obtener Explicación": "Get Explanation",
    "Obtener Explicacion de Directorio": "Get Directory Explanation",
    "Comparar Explicadores": "Compare Explainers",
    "Comparar Explicadores Batch": "Compare Explainers Batch",
    "Transfer Learning - Multiple Models": "Batch Transfer",
}
SPECIAL_BUTTON_ROWS = {
    "Train GNN": (2, 2),
    "Explainer GNN": (2, 2),
}

MODEL_INFO = {
    "URVDEEPDTAF": """<h3>URVDEEPDTAF</h3>
<p><b>Overview.</b> URVDEEPDTAF is the project's URV-specific DeepDTA-family workflow. It remains separated from the newer DeepDTA module because it uses its own local data preparation and training layout.</p>
<p><b>Representation / inputs.</b> The module follows the DeepDTA idea of learning affinity from ligand and target encodings, but it should be used with the prepared files and checkpoints expected by this project-specific implementation.</p>
<p><b>Project-specific notes.</b> Treat it as a local comparison workflow for URV experiments rather than as a generic replacement for the newer adapted DeepDTA module.</p>
<p><b>Limitations / dependencies.</b> No dedicated README was found for this module, so this description is intentionally conservative. Results depend on matching the existing URVDEEPDTAF data layout and checkpoint format.</p>""",
    "EGNN": """<h3>EGNN</h3>
<p><b>Overview.</b> EGNN is the graph-neural workflow used here when spatial or geometric graph information is relevant to binding-affinity prediction.</p>
<p><b>Representation / inputs.</b> The README describes generation of graph data from raw protein and ligand files. In the MPro workflow this means pIC50 labels, ligand SDF files and protein PDB inputs are converted into graph tensors for split-based training.</p>
<p><b>Project-specific notes.</b> The integration keeps the original graph-generation, training and prediction scripts available through GUI dialogs, with split-level outputs and reports based on RMSE, Pearson and Spearman metrics.</p>
<p><b>Strengths.</b> EGNN gives a graph/geometric comparison point against sequence-only models such as DeepDTA and WideDTA.</p>
<p><b>Limitations / dependencies.</b> It is sensitive to graph generation quality and to the selected hyperparameters such as hidden dimension, learning rate and batch size. Checkpoints are tied to the graph data and split protocol used to train them.</p>""",
    "EDNN": """<h3>EDNN</h3>
<p><b>Overview.</b> EDNN is the edge-aware graph-regression workflow for the MPro-URV complexes, predicting pIC50 from generated graph data.</p>
<p><b>Representation / inputs.</b> The generated PyTorch Geometric objects include node features, positions, edge indices, optional edge attributes and labels. The documented default signature uses node_dim=12 and edge_dim=1.</p>
<p><b>Project-specific notes.</b> The graph builder exposes distance cutoffs, including an edge cutoff around 5.0 Angstrom and a protein cutoff around 6.0 Angstrom, so the prepared graphs encode local structural neighborhoods around ligand-protein complexes.</p>
<p><b>Strengths.</b> EDNN is close to EGNN in workflow but explicitly emphasizes graph relations, making it useful when comparing how edge information affects affinity prediction.</p>
<p><b>Limitations / dependencies.</b> Training and evaluation assume consistent graph construction across splits. Changing cutoff choices or feature generation changes the meaning of saved checkpoints.</p>""",
    "DeepDTA": """<h3>DeepDTA</h3>
<p><b>Overview.</b> DeepDTA is a sequence-based drug-target affinity model. It represents compounds with SMILES strings and targets with amino-acid sequences, then learns affinity from those textual encodings.</p>
<p><b>Representation / inputs.</b> The adapted workflow converts MPro-URV data into the DeepDTA-compatible files ligands_can.txt, proteins.txt and Y, while preserving official train, validation and test folds from PDB identifiers.</p>
<p><b>Project-specific notes.</b> The README notes that the original architecture is preserved and that the MPro-URV converter canonicalizes SMILES with RDKit before encoding.</p>
<p><b>Strengths.</b> DeepDTA is lighter than structure-heavy workflows because it does not require pocket feature extraction or 3D matrix generation.</p>
<p><b>Limitations / dependencies.</b> The original fixed SMILES alphabet and max length can force filtering or canonicalization. In this project, compatible results may therefore be computed on the filtered subset that fits the DeepDTA encoding constraints.</p>""",
    "WideDTA": """<h3>WideDTA</h3>
<p><b>Overview.</b> WideDTA is related to DeepDTA but expands the textual representation with word/subsequence inputs and an additional motif branch.</p>
<p><b>Representation / inputs.</b> The documented MPro-URV format contains ligands_can.txt, proteins.txt, motif2.txt and Y. The implementation uses word sizes such as ligand length 8, protein length 3 and motif length 3.</p>
<p><b>Project-specific notes.</b> The README states that motif2 is currently a deterministic technical baseline aligned with protein data, not a complete biological motif-extraction pipeline.</p>
<p><b>Strengths.</b> WideDTA can capture local textual patterns beyond character-level sequence encoding, making it a useful contrast to standard DeepDTA.</p>
<p><b>Limitations / dependencies.</b> Generated data must match the refactored WideDTA dimensions. Older wide.pt checkpoints from hardcoded dimensions are not necessarily reusable after the dynamic-dimension migration.</p>""",
    "DCML": """<h3>DCML</h3>
<p><b>Overview.</b> DCML is not a PyTorch graph model. It is a matrix-based workflow that trains a scikit-learn GradientBoostingRegressor on precomputed feature matrices.</p>
<p><b>Representation / inputs.</b> The prepared data uses a feature.zip containing one .npy matrix per sample and a label.npy target array. The distance_only variant builds ligand-protein Euclidean distance matrices from SDF/PDB coordinates.</p>
<p><b>Project-specific notes.</b> The real_charge and full variants add charge-based descriptors. The GUI accepts a device field for consistency, but the documented backend is CPU-oriented scikit-learn.</p>
<p><b>Strengths.</b> DCML provides a feature-engineering baseline against the neural graph and sequence models, especially for judging how far fixed structural descriptors can go.</p>
<p><b>Limitations / dependencies.</b> Charge variants require external charge-generation tooling such as PDB2PQR for proteins and RDKit/OpenBabel-style ligand charges. Matrix quality depends directly on valid ligand SDF and protein PDB coordinates.</p>""",
    "CAPLA": """<h3>CAPLA</h3>
<p><b>Overview.</b> CAPLA is a structure-aware protein-ligand affinity model adapted here for the URV SARS-CoV-2 Mpro dataset.</p>
<p><b>Representation / inputs.</b> The README describes three complementary inputs for each complex: a global protein feature matrix, a binding-pocket feature matrix and a ligand SMILES representation.</p>
<p><b>Project-specific notes.</b> The adapted module uses the official URV v3b splits and keeps original CAPLA code isolated under CAPLA/original. The prepared dataset contains affinity_data.csv, urv_v3b_smi.csv, global features, pocket features and split folders.</p>
<p><b>Strengths.</b> CAPLA is more structurally informed than DeepDTA/WideDTA because it combines whole-protein context with pocket-level information around the ligand.</p>
<p><b>Limitations / dependencies.</b> The current preparation reuses existing global and pocket matrices. Historical notes indicate those features were generated from DSSP files and interaction JSON files, so regenerating true features depends on structural-processing assets that may not be fully included.</p>""",
    "DEAttentionDTA": """<h3>DEAttentionDTA</h3>
<p><b>Overview.</b> DEAttentionDTA is an attention-based protein-ligand affinity model integrated for the official URV v3b SARS-CoV-2 Mpro splits.</p>
<p><b>Representation / inputs.</b> The maintained workflow expects protein sequence, ligand SMILES and pocket-related inputs. The prepared split folders contain sequence and affinity CSV files for train, validation and test subsets.</p>
<p><b>Project-specific notes.</b> The upstream repository is preserved under DEAttentionDTA/original, and the GUI integration loads the original architecture dynamically while keeping URV runners and HPO logic under DEAttentionDTA/core.</p>
<p><b>Strengths.</b> Attention helps combine ligand, protein and pocket signals, which makes this model a useful bridge between pure sequence models and structure-aware workflows.</p>
<p><b>Limitations / dependencies.</b> Pocket values are reconstructed from MPro-URV Version2 structural files when preparation is rerun. Checkpoints must match the maintained architecture and compatible generated encodings; HPO is designed around validation metrics to avoid test leakage.</p>""",
    "CheapNet": """<h3>CheapNet</h3>
<p><b>Overview.</b> CheapNet is one of the project's older GNN-family affinity workflows. The implementation defines graph message-passing blocks, dense graph pooling and attention-style combination layers.</p>
<p><b>Representation / inputs.</b> It operates on PyTorch Geometric-style graph data with node features, positions, edge indices and batch assignments produced by the existing graph data pipeline.</p>
<p><b>Project-specific notes.</b> CheapNet is exposed through the shared GNN menu and uses the existing data generation, training, search and prediction dialogs rather than a separate README-backed integration.</p>
<p><b>Strengths.</b> It provides a graph-family comparison model that is useful alongside GIGN, GraphDTA and PLANET.</p>
<p><b>Limitations / dependencies.</b> No dedicated README was found, so claims are limited to the observed implementation. Use checkpoints only with compatible generated graph inputs and split files.</p>""",
    "GIGN": """<h3>GIGN</h3>
<p><b>Overview.</b> GIGN is the project's graph-interaction neural workflow. Its implementation embeds node features, applies multiple GIGN blocks and pools graph features before the final affinity predictor.</p>
<p><b>Representation / inputs.</b> The model expects graph data with node features and interaction-style graph connectivity. Code comments indicate the blocks require intra/inter edge information in the prepared data object.</p>
<p><b>Project-specific notes.</b> GIGN is implemented in the legacy GNN model family and is launched through the existing shared dialogs for data, training, search and prediction.</p>
<p><b>Strengths.</b> It is useful for comparing interaction-focused graph modeling against CheapNet and the more specialized graph modules.</p>
<p><b>Limitations / dependencies.</b> The workflow depends on compatible generated graph files and checkpoint formats. No standalone README was found, so the description avoids unsupported architectural claims.</p>""",
    "GraphDTA": """<h3>GraphDTA</h3>
<p><b>Overview.</b> GraphDTA is a graph-based drug-target affinity workflow with selectable neural variants such as GCN, GAT, GAT-GCN and GIN-style models.</p>
<p><b>Representation / inputs.</b> The implementation represents the compound as a molecular graph and combines graph features with an encoded protein target branch, including 1D convolution over target encodings in the GCN variant.</p>
<p><b>Project-specific notes.</b> The dialogs expose model selection and the existing generated graph directories, so the same card can launch the real GraphDTA data, training, search and prediction workflows.</p>
<p><b>Strengths.</b> GraphDTA is a recognizable baseline for comparing ligand-graph modeling against sequence-only and structure-aware modules.</p>
<p><b>Limitations / dependencies.</b> Results depend on the selected variant, encoded target format and consistency of generated graph data across splits.</p>""",
    "PLANET": """<h3>PLANET</h3>
<p><b>Overview.</b> PLANET is a graph-family protein-ligand affinity workflow in this project, using dedicated protein, ligand and protein-ligand interaction components.</p>
<p><b>Representation / inputs.</b> The architecture combines ProteinEGNN, LigandGAT and ProLig modules. Its dialogs expect PLANET-compatible metadata and pickle files such as train.pkl, valid.pkl and core.pkl.</p>
<p><b>Project-specific notes.</b> PLANET keeps its own data directory format and model directory flow, so it should be interpreted through the reports and checkpoints produced by this implementation.</p>
<p><b>Strengths.</b> It offers a structurally oriented graph comparison point with separate protein, ligand and interaction modeling stages.</p>
<p><b>Limitations / dependencies.</b> No module README was found. The description is based on the local architecture and dialogs, and the workflow depends on correctly prepared PLANET data files.</p>""",
}


class LogoPlaceholder(QFrame):
    def __init__(self, label: str, image_path: Path, width: int = 116, height: int = 62):
        super().__init__()
        self.setObjectName("logoPlaceholder")
        self.setFixedSize(width, height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        content = QLabel(label)
        content.setObjectName("logoPlaceholderText")
        content.setAlignment(Qt.AlignCenter)
        content.setWordWrap(True)

        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            content.setPixmap(
                pixmap.scaled(
                    width,
                    height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        layout.addWidget(content)


class DashboardCard(QFrame):
    def __init__(
        self,
        model_name: str,
        description: str,
        actions: list[tuple[str, Callable[[], None]]],
        info_text: str | None = None,
    ):
        super().__init__()
        self.setObjectName("dashboardCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(190)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        title = QLabel(model_name)
        title.setObjectName("cardTitle")

        title_row.addWidget(title, 1)
        if info_text:
            info_button = QPushButton("?")
            info_button.setObjectName("infoButton")
            info_button.setToolTip("Model information")
            info_button.setFixedSize(26, 26)
            info_button.clicked.connect(lambda checked=False, name=model_name, text=info_text: self._show_info(name, text))
            title_row.addWidget(info_button, 0, Qt.AlignRight | Qt.AlignTop)

        body = QLabel(description)
        body.setObjectName("cardDescription")
        body.setWordWrap(True)

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(9)
        buttons.setVerticalSpacing(9)

        labels = tuple(label for label, _ in actions)
        compact_workflow = labels == WORKFLOW_ACTIONS
        workflow_widths = {
            "Generate Data": 124,
            "Train": 76,
            "Search": 82,
            "Evaluate": 94,
        }
        button_widths = [
            workflow_widths.get(label, self._compact_button_width(label))
            for label in labels
        ]
        special_rows = SPECIAL_BUTTON_ROWS.get(model_name)
        columns = 4 if compact_workflow else self._button_columns(button_widths)

        buttons.setAlignment(Qt.AlignLeft)

        for index, (label, callback) in enumerate(actions):
            button = QPushButton(label)
            button.setObjectName("cardButton")
            button.setFixedWidth(button_widths[index])
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            if index == 0:
                button.setProperty("buttonRole", "primary")
            button.clicked.connect(callback)
            if special_rows:
                row, column, column_span = self._special_button_position(index, special_rows)
                buttons.addWidget(button, row, column, 1, column_span)
            else:
                buttons.addWidget(button, index // columns, index % columns)

        for column in range(columns):
            buttons.setColumnStretch(column, 0)

        layout.addLayout(title_row)
        layout.addWidget(body)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def _compact_button_width(self, label: str) -> int:
        text_width = self.fontMetrics().horizontalAdvance(label)
        return max(64, min(text_width + 52, 410))

    def _button_columns(self, button_widths: list[int]) -> int:
        if not button_widths:
            return 1

        max_row_width = 410
        if max(button_widths) > 150 and len(button_widths) <= 4:
            return 2 if self._row_width(button_widths[:2]) <= max_row_width else 1

        for columns in range(min(4, len(button_widths)), 0, -1):
            if self._row_width(button_widths[:columns]) <= max_row_width:
                return columns
        return 1

    def _row_width(self, button_widths: list[int]) -> int:
        if not button_widths:
            return 0
        return sum(button_widths) + 9 * (len(button_widths) - 1)

    def _special_button_position(self, index: int, rows: tuple[int, ...]) -> tuple[int, int, int]:
        offset = 0
        for row, count in enumerate(rows):
            if index < offset + count:
                column = index - offset
                return row, column, 2 if count == 1 else 1
            offset += count
        overflow = index - offset
        return len(rows) + overflow, 0, 2

    def _show_info(self, model_name: str, info_text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{model_name} Information")
        dialog.resize(620, 460)

        content = QLabel(info_text)
        content.setObjectName("infoDialogText")
        content.setTextFormat(Qt.RichText)
        content.setWordWrap(True)
        content.setTextInteractionFlags(Qt.TextSelectableByMouse)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)

        layout = QVBoxLayout(dialog)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        dialog.exec()


class WelcomeScreen(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setObjectName("dashboard")
        self._card_grids: list[tuple[QGridLayout, list[DashboardCard]]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("dashboard")
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 28)
        layout.setSpacing(22)

        layout.addWidget(self._build_header())
        layout.addLayout(self._build_intro())
        self._add_section(
            layout,
            "Specialized binding-affinity models",
            [
                self._card_from_menu(
                    "URVDEEPDTAF",
                    "URV DeepTAF workflows for data generation, training, search and evaluation.",
                    "menu_urvdeepdtaf",
                    MODEL_INFO["URVDEEPDTAF"],
                ),
                self._specialized_card(
                    "EGNN",
                    "Graph neural network using molecular/protein graph features.",
                    "menu_EGNN",
                    ("generate_data_egnn", "train_model_egnn", "train_all_models_egnn", "test_model_egnn"),
                    MODEL_INFO["EGNN"],
                ),
                self._specialized_card(
                    "EDNN",
                    "Edge-aware graph model for binding affinity prediction.",
                    "menu_EDNN",
                    ("generate_data_ednn", "train_model_ednn", "train_all_models_ednn", "test_model_ednn"),
                    MODEL_INFO["EDNN"],
                ),
                self._specialized_card(
                    "DeepDTA",
                    "Sequence-based drug-target affinity model.",
                    "menu_DeepDTA",
                    (
                        "open_generate_data_dialog",
                        "open_train_dialog",
                        "open_hyperparameter_search_dialog",
                        "open_evaluate_dialog",
                    ),
                    MODEL_INFO["DeepDTA"],
                ),
                self._specialized_card(
                    "WideDTA",
                    "Word/subsequence-based drug-target affinity model.",
                    "menu_WideDTA",
                    (
                        "open_generate_data_dialog",
                        "open_train_dialog",
                        "open_hyperparameter_search_dialog",
                        "open_evaluate_dialog",
                    ),
                    MODEL_INFO["WideDTA"],
                ),
                self._specialized_card(
                    "DCML",
                    "Distance and charge matrix learning model.",
                    "menu_DCML",
                    ("open_generate_data_dialog", "open_train_dialog", "open_search_dialog", "open_test_dialog"),
                    MODEL_INFO["DCML"],
                ),
                self._specialized_card(
                    "CAPLA",
                    "Structure-aware binding affinity model using global and pocket features.",
                    "menu_CAPLA",
                    ("open_generate_data_dialog", "open_train_dialog", "open_hpo_dialog", "open_test_dialog"),
                    MODEL_INFO["CAPLA"],
                ),
                self._specialized_card(
                    "DEAttentionDTA",
                    "Attention-based sequence model for drug-target affinity prediction.",
                    "menu_DEAttentionDTA",
                    ("open_prepare_dialog", "open_train_dialog", "open_hpo_dialog", "open_test_dialog"),
                    MODEL_INFO["DEAttentionDTA"],
                ),
            ],
        )
        self._add_section(
            layout,
            "GNN model families",
            [
                self._card_from_menu(
                    "CheapNet",
                    "CheapNet graph-based affinity workflows.",
                    "menu_cheapnet",
                    MODEL_INFO["CheapNet"],
                ),
                self._card_from_menu(
                    "GIGN",
                    "Interaction graph neural network workflows.",
                    "menu_gign",
                    MODEL_INFO["GIGN"],
                ),
                self._card_from_menu(
                    "GraphDTA",
                    "GraphDTA model workflows for molecular graph affinity prediction.",
                    "menu_dta",
                    MODEL_INFO["GraphDTA"],
                ),
                self._card_from_menu(
                    "PLANET",
                    "PLANET graph model workflows.",
                    "planet_menu",
                    MODEL_INFO["PLANET"],
                ),
            ],
        )
        self._add_section(
            layout,
            "Molecular tools and GNN utilities",
            [
                self._card_from_menu(
                    "Molecule",
                    "Create, load, save and prepare molecular structure files.",
                    "menu_molecula",
                ),
                self._card_from_menu(
                    "Train GNN",
                    "Training workflows for legacy GNN models.",
                    "menu_train",
                ),
                self._card_from_menu(
                    "Transfer GNN",
                    "Transfer learning workflows for one or multiple GNN models.",
                    "menu_transfer",
                ),
                self._card_from_menu(
                    "Test GNN",
                    "Prediction, testing and inspection workflows for GNN checkpoints.",
                    "menu_test",
                ),
                self._card_from_menu(
                    "Hyperparameter Search GNN",
                    "Launch the existing GNN hyperparameter search workflow.",
                    "menu_hyperparameter_search",
                ),
                self._card_from_menu(
                    "Explainer GNN",
                    "Existing explanation and explainer comparison workflows.",
                    "menu_explicacion",
                ),
            ],
        )
        layout.addStretch(1)

        root.addWidget(scroll)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("dashboardHeader")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        urv_logo = LogoPlaceholder("URV LOGO", PROJECT_ROOT / "assets" / "urv_logo.png")
        app_logo = LogoPlaceholder("APP LOGO", PROJECT_ROOT / "assets" / "app_logo.png")

        title_block = QVBoxLayout()
        title_block.setSpacing(6)

        title = QLabel("Molecular Analysis System")
        title.setObjectName("dashboardTitle")

        subtitle = QLabel("Protein-ligand binding affinity prediction and model evaluation platform")
        subtitle.setObjectName("dashboardSubtitle")
        subtitle.setWordWrap(True)

        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        layout.addWidget(urv_logo)
        layout.addWidget(app_logo)
        layout.addLayout(title_block, 1)

        return header

    def _build_intro(self) -> QVBoxLayout:
        intro = QVBoxLayout()
        intro.setSpacing(7)

        title = QLabel("Model Dashboard")
        title.setObjectName("pageTitle")

        description = QLabel(
            "Generate molecular datasets, train models, run hyperparameter search, "
            "and evaluate predictions through the model cards below."
        )
        description.setObjectName("dashboardDescription")
        description.setWordWrap(True)

        intro.addWidget(title)
        intro.addWidget(description)
        return intro

    def _add_section(self, layout: QVBoxLayout, title: str, cards: list[DashboardCard | None]) -> None:
        visible_cards = [card for card in cards if card is not None]
        if not visible_cards:
            return

        section_title = QLabel(title)
        section_title.setObjectName("sectionTitle")
        layout.addWidget(section_title)
        layout.addLayout(self._build_card_grid(visible_cards))

    def _build_card_grid(self, cards: list[DashboardCard]) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        self._card_grids.append((grid, cards))
        self._relayout_card_grid(grid, cards)
        return grid

    def _dashboard_columns(self) -> int:
        width = self.width()
        if width >= 1420:
            return 3
        if width >= 860:
            return 2
        return 1

    def _relayout_all_card_grids(self) -> None:
        for grid, cards in self._card_grids:
            self._relayout_card_grid(grid, cards)

    def _relayout_card_grid(self, grid: QGridLayout, cards: list[DashboardCard]) -> None:
        while grid.count():
            grid.takeAt(0)

        columns = self._dashboard_columns()
        for index, card in enumerate(cards):
            grid.addWidget(card, index // columns, index % columns)

        for column in range(3):
            grid.setColumnStretch(column, 1 if column < columns else 0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout_all_card_grids()

    def _specialized_card(
        self,
        model_name: str,
        description: str,
        menu_name: str,
        method_names: tuple[str, str, str, str],
        info_text: str | None = None,
    ) -> DashboardCard | None:
        actions = self._model_actions(menu_name, method_names)
        return DashboardCard(model_name, description, actions, info_text) if actions else None

    def _model_actions(
        self,
        menu_name: str,
        method_names: tuple[str, str, str, str],
    ) -> list[tuple[str, Callable[[], None]]]:
        menu = getattr(getattr(self.main_window, "menu_bar", None), menu_name, None)
        actions: list[tuple[str, Callable[[], None]]] = []

        for label, method_name in zip(WORKFLOW_ACTIONS, method_names):
            method = getattr(menu, method_name, None)
            if callable(method):
                actions.append((label, method))

        return actions

    def _card_from_menu(
        self,
        title: str,
        description: str,
        menu_name: str,
        info_text: str | None = None,
    ) -> DashboardCard | None:
        menu = getattr(getattr(self.main_window, "menu_bar", None), menu_name, None)
        if not isinstance(menu, QMenu):
            return None

        actions: list[tuple[str, Callable[[], None]]] = []
        for action in menu.actions():
            if action.isSeparator() or not action.text().strip():
                continue

            def trigger_action(checked: bool = False, action=action) -> None:
                action.trigger()

            display_label = DISPLAY_LABELS.get(action.text(), action.text())
            actions.append((display_label, trigger_action))

        return DashboardCard(title, description, actions, info_text) if actions else None
