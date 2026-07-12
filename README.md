# Molecular Analysis System

Desktop GUI for protein-ligand binding affinity prediction, model training, hyperparameter search and evaluation.

## Project Overview

Molecular Analysis System is a PySide6 desktop application for running molecular binding-affinity experiments from a single graphical interface. The project is centered mainly on SARS-CoV-2 Mpro / MPro-URV datasets and integrates multiple model families used for protein-ligand affinity prediction.

The GUI centralizes:

- dataset generation and preparation;
- model training;
- hyperparameter search;
- checkpoint or model-bundle evaluation;
- prediction and metric outputs;
- graph-based, sequence-based, structure-aware and matrix-based model workflows.

The root README gives a high-level project map. Model-specific technical details should remain in the README file inside each model folder.

## Main GUI Workflow

The finalized model cards use the following common workflow:

```text
Generate Data -> Train -> Search -> Evaluate
```

| Action | Purpose |
|---|---|
| Generate Data | Converts raw or prepared molecular datasets into the format required by the selected model. |
| Train | Trains a model using the selected dataset, split and configuration. |
| Search | Runs hyperparameter search using validation metrics. Test metrics must not be used for model selection. |
| Evaluate | Loads trained checkpoint(s) or model bundles and computes predictions and metrics. |

Each model card opens its own dialogs and writes outputs to model-specific folders.

## Installation

The main environment definition is `environment.yml`. The project expects Python 3.10 and the dependencies listed there.

The requested environment name for the final GUI is `gui_app`:

```bash
conda env create -n gui_app -f environment.yml
conda activate gui_app
```

If the environment already exists:

```bash
conda env update -n gui_app -f environment.yml
conda activate gui_app
```

Note: the current `environment.yml` file declares `name: Philippe_v2`. Running `conda env create -f environment.yml` without `-n gui_app` will create/use that declared name instead. The environment file was not changed.

Some PyTorch Geometric and DIG-related packages may require version-compatible wheels depending on the machine's CUDA and PyTorch setup. If imports fail after environment creation, check the installed PyTorch, CUDA and PyG versions before changing project code.

## Basic Environment Checks

After activating the environment, run:

```bash
python -c "import torch; print(torch.__version__)"
python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "import torch_geometric; print('PyG OK')"
```

Optional checks for structure or charge workflows:

```bash
which mkdssp || which dssp
which pdb2pqr || which pdb2pqr30
which obabel
```

These tools are not required by every model. They are relevant for workflows that generate secondary-structure, pocket, or charge-based features.

## Running the Application

```bash
conda activate gui_app
python main.py
```

The application opens a card-based dashboard. Each card represents a model or utility module and opens the corresponding existing dialogs.

## Project Structure

```text
TrabajoInvestigacion/
├── main.py
├── environment.yml
├── README.md
├── ui/
├── EGNN/
├── EDNN/
├── DeepDTA/
├── WideDTA/
├── DCML/
├── CAPLA/
├── DEAttentionDTA/
├── URVDEEPTAF/
├── models/
├── menu_ui/
├── GNNs/
└── ...
```

- `ui/` contains the main PySide6 frontend, dashboard, dialogs, menus and shared theme.
- Model folders such as `EGNN/`, `EDNN/`, `DeepDTA/`, `WideDTA/`, `DCML/`, `CAPLA/` and `DEAttentionDTA/` contain model-specific integrations.
- Older or colleague-owned GNN modules are stored under folders such as `models/`, `menu_ui/`, `job_config/`, `core/` and `GNNs/`.
- Generated data and results are usually kept in model-specific `data/`, `models/`, `outputs/`, `results/`, `Graphs_*`, `Models_*` or `Results_*` folders.

## Dataset Note

Raw datasets such as MPro-URV may not be committed to this repository because of size, confidentiality, or path-specific constraints. GUI dialogs ask for dataset roots, prepared dataset roots, feature roots, checkpoint paths and output folders as needed.

Avoid hardcoding local machine paths in source code. Prefer selecting paths in the GUI dialogs or storing them through the application's settings.

## Model Modules

### EGNN

EGNN is a graph-based model workflow using generated molecular/protein graph data. The module uses graph structure and geometric information rather than only sequence strings.

Graph generation depends on MPro-URV-style inputs such as `pIC50`, ligand SDF files and protein PDB files. Relevant graph-construction settings include edge and protein cutoffs when building local structural neighborhoods.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [EGNN/README.md](EGNN/README.md)

### EDNN

EDNN is an edge-aware graph neural network workflow. It follows a similar GUI structure to EGNN but uses graph node information together with edge information.

The documented graph data contract includes PyTorch Geometric fields such as node features, positions, edge indices, optional edge attributes, labels and batch vectors. The module documentation describes node and edge feature assumptions and graph cutoffs.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [EDNN/README.md](EDNN/README.md)

### DeepDTA

DeepDTA is a sequence-based drug-target affinity model. It represents ligands with SMILES strings and proteins with amino-acid sequences.

The MPro-URV integration converts external data into the original DeepDTA-compatible layout: `ligands_can.txt`, `proteins.txt` and `Y`. The original fixed SMILES alphabet and length can require RDKit canonicalization and filtering, so compatible results may be computed on a filtered subset.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [DeepDTA/README.md](DeepDTA/README.md)

### WideDTA

WideDTA is a sequence/subsequence extension of the DeepDTA family. It uses ligand, protein and motif-style branches rather than only character-level sequence encodings.

The adapted MPro-URV format includes `ligands_can.txt`, `proteins.txt`, `motif2.txt` and `Y`. The current documentation states that `motif2` is a technical baseline aligned with protein data, not a complete biological motif extraction pipeline. Older `wide.pt` checkpoints may not be compatible with the refactored dynamic architecture dimensions.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [WideDTA/README.md](WideDTA/README.md)

### DCML

DCML is a matrix-based workflow, not a PyTorch graph neural network. The current implementation trains a scikit-learn `GradientBoostingRegressor` on precomputed feature matrices.

The expected prepared format includes a feature ZIP containing `.npy` matrices and a label `.npy` file. The `distance_only` variant uses ligand-protein distance matrices from structural coordinates. Charge-based variants such as `real_charge` or `full` require charge generation tools such as PDB2PQR and RDKit/OpenBabel-style ligand charge generation when enabled.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [DCML/README.md](DCML/README.md)

### CAPLA

CAPLA is a structure-aware binding-affinity model adapted to the URV SARS-CoV-2 Mpro dataset. It consumes three complementary inputs: global protein features, binding-pocket protein features and ligand SMILES.

The adapted module uses official URV v3b splits. Hyperparameter search should use training and validation subsets only; test metrics are reserved for final evaluation. True structural feature generation may depend on DSSP/mkdssp or related structural-processing assets when regenerating global or pocket features.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [CAPLA/Readme.md](CAPLA/Readme.md)

### DEAttentionDTA

DEAttentionDTA is an attention-based binding-affinity model integrated for the official URV v3b SARS-CoV-2 Mpro splits. The maintained project workflow uses encoded protein sequence, ligand SMILES and pocket-related representations.

The module preserves upstream code under `DEAttentionDTA/original/` and keeps adapted URV workflows under `DEAttentionDTA/core/`. Checkpoint compatibility matters because the GUI integration loads the maintained architecture and expects compatible generated encodings. HPO must avoid test-set leakage.

Available GUI workflow: `Generate Data`, `Train`, `Search`, `Evaluate`.

Detailed documentation: [DEAttentionDTA/README.md](DEAttentionDTA/README.md)

## Modules Pending Detailed Documentation

The following modules are available from the GUI dashboard, but detailed technical documentation has not yet been added to this repository. The responsible contributor should complete these sections with architecture summaries, dataset requirements, training/search/evaluation outputs and known limitations.

### CheapNet

This module is available from the GUI dashboard as part of the GNN model family.

TODO: add model architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation outputs.
TODO: add limitations and dependencies.

### GIGN

This module is available from the GUI dashboard as part of the GNN model family.

TODO: add model architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation outputs.
TODO: add limitations and dependencies.

### GraphDTA

This module is available from the GUI dashboard as a graph-based drug-target affinity workflow.

TODO: add model architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation outputs.
TODO: add limitations and dependencies.

### PLANET

This module is available from the GUI dashboard as a protein-ligand graph workflow.

TODO: add model architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation outputs.
TODO: add limitations and dependencies.

### URVDEEPDTAF

This module is available from the GUI dashboard as a URV-specific DeepDTA-family workflow, but no dedicated root-level technical README was found during this documentation pass.

TODO: add model architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation outputs.
TODO: add limitations and dependencies.

### Molecule Tools and GNN Utilities

The dashboard also exposes molecular file utilities, GNN training utilities, transfer learning, testing, hyperparameter search and explainer workflows.

TODO: add utility workflow reference.
TODO: add expected input/output formats.
TODO: add notes about optional explainer dependencies.

## GUI Dashboard Reference

The final GUI is card-based. Each model card opens the corresponding workflow dialogs. The `?` button on model cards shows a short model information summary. Dialogs may remember previously selected paths through `QSettings`; users should verify paths before launching long-running jobs.

The top menu is not the primary professor-facing interface. Its functionality is represented through dashboard cards.

## Results and Outputs

Each model writes outputs into its own model-specific folders. Depending on the workflow, common artifacts include:

- trained checkpoints or model bundles;
- prediction CSV files;
- metrics CSV, JSON, or YAML files;
- scatter plots and diagnostic figures;
- training histories;
- HPO trial CSV files;
- best configuration YAML/JSON files;
- generated dataset reports.

Do not assume the exact same filenames across all modules. Use the module-specific README and GUI dialog output fields for exact locations.

## Experimental Methodology Notes

- Hyperparameter search must use validation metrics only.
- Test metrics are for final evaluation, not model selection.
- Official dataset folds should be used where available.
- Smoke tests are only for checking that the pipeline runs; they must not be reported as final scientific results.
- Keep generated datasets, checkpoints and outputs tied to the exact code and configuration used to produce them.

## Troubleshooting

| Problem | Suggested check |
|---|---|
| GUI does not start because a module is missing | Activate the correct conda environment and run `python main.py` from the project root. |
| `PySide6` import error | Confirm the environment was created from `environment.yml` and that `pyside6` is installed. |
| RDKit import error | Check `python -c "from rdkit import Chem"` inside the active environment. |
| PyTorch Geometric / `torch_sparse` errors | Verify PyTorch, CUDA and PyG wheels are version-compatible on the machine. |
| Missing DSSP/mkdssp | Required only for workflows that regenerate secondary-structure or pocket features. Install system tools if those workflows are needed. |
| Missing PDB2PQR | Required only for charge-based DCML variants that need protein charge generation. |
| Missing OpenBabel/`obabel` | Required only for workflows that call ligand charge or conversion tools. |
| Stale paths after moving the project | Dialog paths may be stored in `QSettings`; reselect paths in the GUI. |
| Missing raw dataset files | Confirm the selected dataset root contains the expected MPro-URV files for the selected module. |
| CUDA requested but unavailable | Use `auto` or `cpu` in the dialog, or install a CUDA-compatible PyTorch build. |

## Contribution and Documentation Notes

- Model-specific README files should remain the source of detailed technical information.
- The root README should stay high-level and should not duplicate full model documentation.
- New model modules should include their own README with dataset format, workflow, outputs and limitations.
- Generated data, model checkpoints and experiment outputs should normally not be committed unless explicitly required.
- Do not hardcode machine-specific paths in source code.

## Final Status

| Module | Documentation status | GUI workflow | Notes |
|---|---|---|---|
| EGNN | Available | Generate/Train/Search/Evaluate | Graph model using generated molecular/protein graphs. |
| EDNN | Available | Generate/Train/Search/Evaluate | Edge-aware graph model. |
| DeepDTA | Available | Generate/Train/Search/Evaluate | Sequence model using SMILES and proteins. |
| WideDTA | Available | Generate/Train/Search/Evaluate | Word/motif-style sequence model. |
| DCML | Available | Generate/Train/Search/Evaluate | Matrix-based GradientBoostingRegressor workflow. |
| CAPLA | Available | Generate/Train/Search/Evaluate | Structure-aware global/pocket model. |
| DEAttentionDTA | Available | Generate/Train/Search/Evaluate | Attention-based model using sequence, ligand and pocket inputs. |
| CheapNet | TODO | Dashboard card present | Colleague documentation pending. |
| GIGN | TODO | Dashboard card present | Colleague documentation pending. |
| GraphDTA | TODO | Dashboard card present | Colleague documentation pending. |
| PLANET | TODO | Dashboard card present | Colleague documentation pending. |
| URVDEEPDTAF | TODO | Dashboard card present | URV-specific DeepDTA-family documentation pending. |
| Molecule/GNN utilities | TODO | Dashboard cards present | Utility workflow documentation pending. |
