# Molecular Analysis System

Molecular Analysis System is a desktop GUI application for protein-ligand binding affinity prediction workflows. The interface centralizes molecule utilities, graph-based models, sequence-based models, hyperparameter search workflows, result inspection, and persistent project settings.

The application is designed to let non-specialist users run model workflows without editing Python scripts directly.

## Current scope

The current integration focuses on:

- EGNN workflows:
    - data generation
    - single-model training
    - hyperparameter search
    - single-model evaluation
    - batch evaluation

- EDNN workflows:
    - data generation
    - single-model training
    - hyperparameter search
    - single-model evaluation
    - batch evaluation

- DeepDTA workflows:
    - hyperparameter search

- WideDTA workflows:
    - hyperparameter search

- GUI infrastructure:
    - dashboard navigation
    - persistent settings page
    - centralized results page
    - result file detection
    - metric parsing
    - CSV summary export
    - light theme
    - application logos and desktop launcher support

URVDEEPTAF is kept as an existing external module and is not the main development target of this stage.

## Project structure

```text
TrabajoInvestigacion/
├── assets/
├── DeepDTA/
│   ├── Core/
│   ├── data/
│   ├── results/
│   ├── ui/
│   ├── utils/
│   ├── train.py
│   └── workers.py
├── EDNN/
│   ├── ui/
│   ├── utils/
│   └── workers.py
├── EGNN/
│   ├── ui/
│   ├── utils/
│   └── workers.py
├── GNNs/
├── graph_managment/
├── hyperparameter_Search/
├── ui/
│   ├── assets/
│   │   ├── icons/
│   │   └── logos/
│   ├── controllers/
│   ├── dialogs/
│   ├── graph_interface/
│   ├── menus/
│   ├── pages/
│   ├── themes/
│   ├── utils/
│   ├── widgets/
│   ├── main_window.py
│   └── menu_bar.py
├── URVDEEPTAF/
├── WideDTA/
├── environment.yml
├── requirements.txt
└── main.py
```

## Requirements

Recommended system:

- Linux Ubuntu or equivalent
- Conda or Miniconda
- Python 3.10
- CUDA-capable GPU recommended for serious training
- CPU mode supported for testing and debugging

Main Python dependencies include:

- PySide6
- PyTorch
- torch-geometric
- RDKit
- NumPy
- pandas
- scikit-learn
- matplotlib
- PyYAML

The full environment should be installed from `environment.yml` when possible.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd TrabajoInvestigacion
```

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate gui_app
```

If the environment already exists and dependencies have changed, update it:

```bash
conda env update -f environment.yml --prune
```

If some packages are only listed in `requirements.txt`, install them after activating the environment:

```bash
pip install -r requirements.txt
```

Verify that Python points to the Conda environment:

```bash
which python
python --version
```

Expected behavior:

```text
/home/<user>/miniconda3/envs/gui_app/bin/python
Python 3.10.x
```

## Running the application

From the project root:

```bash
conda activate gui_app
python main.py
```

The GUI should open with the main dashboard.

The application entry point is:

```text
main.py
```

This file:

- limits BLAS/OpenMP thread usage
- sets the project root as the working directory
- configures multiprocessing with `spawn`
- initializes the Qt application
- applies the light theme
- launches the main window

## Desktop launcher

A Linux desktop shortcut can be created with a `.desktop` file.

Create the file:

```bash
nano ~/.local/share/applications/molecular-analysis-system.desktop
```

Example content:

```ini
[Desktop Entry]
Type=Application
Name=Molecular Analysis System
Comment=Protein-ligand binding affinity prediction platform
Exec=/home/<user>/miniconda3/envs/gui_app/bin/python /home/<user>/Studies/stage/TrabajoInvestigacion/main.py
Path=/home/<user>/Studies/stage/TrabajoInvestigacion
Icon=/home/<user>/Studies/stage/TrabajoInvestigacion/ui/assets/icons/app_icon.png
Terminal=false
Categories=Science;Education;
StartupWMClass=molecular-analysis-system
```

Update the application database:

```bash
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

Important details:

- `Exec` must point to the Python interpreter inside the `gui_app` environment.
- `Path` must point to the project root.
- `Icon` must point to a valid image file.
- The filename must be:

```text
molecular-analysis-system.desktop
```

The application also uses:

```python
app.setDesktopFileName("molecular-analysis-system")
```

This helps Ubuntu associate the running window with the correct desktop launcher icon.

## Assets

Recommended asset locations:

```text
ui/assets/logos/urv_logo.png
ui/assets/logos/app_logo.png
ui/assets/icons/app_icon.png
```

The GUI header displays:

- URV logo
- application logo
- application title
- device status
- execution status

If a logo file is missing, the application may fall back to a text placeholder.

## Main GUI pages

### Dashboard

The dashboard is the central navigation page.

It provides access to:

- molecule tools
- SDF utilities
- GNN training
- experiments
- explainability
- results
- settings
- specialized model workflows

The old menu bar is kept internally as an action registry, but it is hidden from the interface. Dashboard buttons trigger the existing menu actions programmatically.

### Settings

The Settings page centralizes persistent project configuration.

It stores values using `QSettings`.

Main settings include:

```text
Dataset paths:
- dataset root
- Ligand_SDF directory
- Protein_PDB directory
- pIC50 file
- splits folder

Model paths:
- EGNN root
- EDNN root
- DeepDTA root
- WideDTA root
- exports directory

Runtime:
- default device
- default seed

Application resources:
- URV logo
- application logo
- application icon
```

After saving settings, the values remain available after closing and reopening the application.

### Results

The Results page detects and displays generated experiment outputs.

It scans the project tree for files such as:

```text
*_hyperparameter_trials.csv
metrics_summary.csv
metrics_per_split.csv
best_config_*.yaml
```

It displays:

- best RMSE
- best Pearson
- best Spearman when available
- total parsed trials
- detected result files
- experiment summary table

It also supports:

- refreshing detected results
- opening the folder of a selected result file
- exporting a consolidated summary CSV

Exported summaries are written to:

```text
exports/
```

Example:

```text
exports/results_summary_20260511_093000.csv
```

## Settings-driven dialogs

Model dialogs are connected to the global Settings page.

The general flow is:

```text
SettingsPage
→ AppSettings
→ model dialog
→ get_inputs()
→ worker thread
→ training / evaluation / HPO pipeline
```

This avoids hardcoding the same paths repeatedly in multiple dialogs.

### EGNN

The EGNN dialogs use:

```text
paths/pic50_file        → Generate Data
paths/ligand_sdf        → Generate Data
paths/protein_pdb       → Generate Data
paths/egnn_root         → Graphs_EGNN / Models_EGNN / Results_EGNN
paths/splits_folder     → train / validation / test split files
runtime/default_device  → training / search / evaluation
runtime/default_seed    → training / search
```

Expected default paths:

```text
EGNN/Graphs_EGNN
EGNN/Models_EGNN
EGNN/Results_EGNN
```

### EDNN

The EDNN dialogs use:

```text
paths/pic50_file        → Generate Data
paths/ligand_sdf        → Generate Data
paths/protein_pdb       → Generate Data
paths/ednn_root         → Graphs_EDNN / Models_EDNN / Results_EDNN
paths/splits_folder     → train / validation / test split files
runtime/default_device  → training / search / evaluation
runtime/default_seed    → training / search
```

Expected default paths:

```text
EDNN/Graphs_EDNN
EDNN/Models_EDNN
EDNN/Results_EDNN
```

### DeepDTA

The DeepDTA hyperparameter search dialog uses:

```text
paths/deepdta_root       → DeepDTA output root
runtime/default_device   → device
runtime/default_seed     → seed
```

Expected output root:

```text
DeepDTA/results/deepdta_hpo/runs
```

### WideDTA

The WideDTA hyperparameter search dialog uses:

```text
paths/widedta_root       → WideDTA output root
runtime/default_device   → device
runtime/default_seed     → seed
```

Expected output root:

```text
WideDTA/results/widedta_hpo/runs
```

## Dataset layout

The application expects the MPro-URV dataset to provide molecular and affinity files.

Typical dataset elements:

```text
MPro-URV_Version2/
├── Ligand_SDF/
├── Protein_PDB/
├── Splits/
│   ├── train_index_folder.txt
│   ├── valid_index_folder.txt
│   └── test_index_folder.txt
└── pIC50.txt
```

The exact paths should be configured in:

```text
Dashboard → Settings → Open settings
```

## Recommended validation workflow

After installation and configuration, run a minimal test before launching long experiments.

### 1. Compile Python files

```bash
find ui EGNN EDNN DeepDTA WideDTA -type d -name "__pycache__" -prune -exec rm -rf {} +
python -m compileall ui EGNN EDNN DeepDTA WideDTA
```

### 2. Launch the GUI

```bash
python main.py
```

### 3. Configure Settings

Open:

```text
Dashboard → Settings → Open settings
```

Set and save:

```text
Dataset root
Ligand_SDF
Protein_PDB
pIC50 file
Splits folder
EGNN root
EDNN root
DeepDTA root
WideDTA root
Default device
Default seed
```

Close and reopen the application to confirm that settings persist.

### 4. Check model dialogs

Open each dialog and verify that paths are pre-filled correctly:

```text
EGNN → Generate / Train / Search / Evaluate
EDNN → Generate / Train / Search / Evaluate
DeepDTA → Search
WideDTA → Search
```

### 5. Run a minimal hyperparameter search

For debugging only:

```text
epochs = 1
learning rates = one value
batch sizes = one value
max_train_batches = 1
```

For WideDTA:

```text
dropout values = one value
```

The objective is not model quality. The objective is to verify that:

- parameters are accepted
- workers start correctly
- results are written
- the Results page detects outputs

## Running experiments

### EGNN hyperparameter search

From the dashboard:

```text
EGNN → Search
```

Recommended serious search space depends on available hardware. A minimal debugging configuration is:

```text
epochs = 1
patience = 1
lr_values = 1e-4
hidden_dim_values = 32
batch_size_values = 4
```

A more serious search can use multiple values:

```text
lr_values = 5e-5,1e-4,5e-4,1e-3
hidden_dim_values = 32,64,128
batch_size_values = 2,4,8
```

### EDNN hyperparameter search

From the dashboard:

```text
EDNN → Search
```

Use the same validation strategy as EGNN before launching long runs.

### DeepDTA hyperparameter search

From the dashboard:

```text
DeepDTA → Search
```

Main parameters:

```text
dataset_name
output_root
device
seed
epochs
lr_values
batch_size_values
val_split
test_split
use_dataset_folds
fold_index
max_train_batches
```

### WideDTA hyperparameter search

From the dashboard:

```text
WideDTA → Search
```

Main parameters:

```text
dataset_name
output_root
device
seed
epochs
lr_values
batch_size_values
dropout_values
val_split
test_split
use_dataset_folds
fold_index
max_train_batches
```

## Results and exported files

Generated results may include:

```text
best_config_*.yaml
*_hyperparameter_trials.csv
metrics_summary.csv
metrics_per_split.csv
```

The Results page can export a consolidated summary:

```text
Dashboard → Results → Open results → Export summary
```

The export contains:

```text
model
source_file
rmse
pearson
spearman
status
trials
path
```

## Git hygiene

Generated files should not be committed.

Recommended ignored folders:

```text
__pycache__/
*.pyc
.venv/
.env
*.pt
*.pth
*.ckpt
exports/
EGNN/Graphs_EGNN/
EGNN/Models_EGNN/
EGNN/Results_EGNN/
EDNN/Graphs_EDNN/
EDNN/Models_EDNN/
EDNN/Results_EDNN/
DeepDTA/results/
WideDTA/results/
```

If large files have already been tracked by Git, `.gitignore` alone will not remove them from Git history or tracking. Use:

```bash
git rm --cached <file-or-folder>
```

Then commit the removal.

## Troubleshooting

### The GUI opens but uses the wrong theme

Verify that `main.py` calls:

```python
apply_theme(app, "light")
```

Also verify that `ui/themes/light.qss` exists.

### Desktop launcher does not show the correct icon

Check:

```text
~/.local/share/applications/molecular-analysis-system.desktop
```

Verify:

```ini
Icon=/absolute/path/to/ui/assets/icons/app_icon.png
Path=/absolute/path/to/TrabajoInvestigacion
```

Also verify that `main.py` contains:

```python
app.setDesktopFileName("molecular-analysis-system")
```

### Spinbox arrows disappear when launching from desktop

This usually means that relative paths in the QSS are resolved from the wrong working directory.

Fix the `.desktop` file with:

```ini
Path=/absolute/path/to/TrabajoInvestigacion
```

The application also sets its working directory to the project root in `main.py`.

### Settings are saved but dialogs do not update

Check whether the dialog has been connected to `AppSettings`.

Expected import:

```python
from ui.utils.app_settings import AppSettings
```

Expected pattern:

```python
self.app_settings = AppSettings()
```

Then the dialog should use values such as:

```python
self.app_settings.get_value("paths/egnn_root")
self.app_settings.get_value("paths/ednn_root")
self.app_settings.get_value("runtime/default_device")
self.app_settings.get_value("runtime/default_seed")
```

### Results page does not show new files

Click:

```text
Refresh
```

If files still do not appear, verify that generated files contain recognizable names such as:

```text
trials
metrics_summary
metrics_per_split
best_config
```

### CUDA is unavailable

Use:

```text
device = cpu
```

or:

```text
device = auto
```

If `auto` causes an error in a specific backend, use `cpu` explicitly.

### A long path appears visually cut in a dialog

This is normal behavior for `QLineEdit` when the path is longer than the visible field width. The full text is still stored and returned by `get_inputs()`.

## Development conventions

General code conventions used in this project:

- GUI code is kept in `ui/`.
- Model-specific GUI code is kept inside each model module.
- Menus trigger dialogs.
- Dialogs return dictionaries through `get_inputs()`.
- Workers execute long operations in background threads.
- Training and evaluation pipelines remain separated from GUI code.
- Persistent GUI settings are centralized in `ui/utils/app_settings.py`.
- Result parsing is centralized in `ui/controllers/results_controller.py`.

Preferred architecture:

```text
Dashboard
→ Dialog
→ get_inputs()
→ Worker
→ Core pipeline
→ Results files
→ ResultsPage
```

## Minimal developer checklist

Before committing GUI changes:

```bash
find ui EGNN EDNN DeepDTA WideDTA -type d -name "__pycache__" -prune -exec rm -rf {} +
python -m compileall ui EGNN EDNN DeepDTA WideDTA
python main.py
```

Then manually verify:

```text
Dashboard opens
Settings opens
Results opens
EGNN dialogs open
EDNN dialogs open
DeepDTA dialog opens
WideDTA dialog opens
Log panel still works
Header logo returns to dashboard
```

## Known limitations

- DeepDTA and WideDTA currently expose hyperparameter search from the GUI, not the full set of possible training and evaluation actions.
- Some generated result formats may require parser updates before all metrics appear in the Results page.
- The GUI assumes the expected dataset files exist and are correctly formatted.
- URVDEEPTAF is preserved as an existing external module and is not the main refactoring target.

## License

This project is part of an academic internship and should be used according to the rules defined by the institution, supervisor, and repository owner.
