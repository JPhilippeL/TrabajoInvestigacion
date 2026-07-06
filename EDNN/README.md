# EDNN — Graph Generation, Training, Evaluation and Hyperparameter Search from the GUI

## 1. Purpose of this module

The `EDNN` module was integrated into the graphical application to provide a complete graph-regression workflow for protein–ligand complexes from the **MPro-URV** dataset. Its purpose is to predict an experimental `pIC50` value from graphs generated from the molecular and protein structures available in the database.

The module currently supports four main operations:

1. generation of EDNN graphs in PyTorch Geometric format;
2. training of an EDNN model on predefined dataset splits;
3. evaluation of trained EDNN models;
4. exhaustive grid-based hyperparameter search.

The EDNN integration intentionally follows the same application architecture as the `EGNN` module. This was a deliberate engineering decision. A user who understands EGNN can use EDNN without learning a second workflow, and a developer can maintain both modules without dealing with two unrelated code organizations.

This module does **not** claim to redefine the scientific EDNN architecture. The application layer only wraps the model so that it can be used from the GUI. The scientific code remains concentrated in `ednn_model.py`, `ednn_trainer.py`, and `ednn_tester.py`. The integration layer manages file paths, dialogs, background workers, result folders, YAML and CSV output, and error reporting.

---

## 2. Functional overview

The complete EDNN workflow is:

```text
MPro-URV dataset
    │
    ├── pIC50.txt
    ├── Ligand/Ligand_SDF/*.sdf
    └── Protein/Protein_PDB/*.pdb
          │
          ▼
Generate Data
          │
          ▼
Graphs_EDNN/*.pt
          │
          ├── Train Model
          │      └── Models_EDNN/split_XX/best_model.pt
          │
          ├── Evaluate Model
          │      └── Results_EDNN/*
          │
          └── Hyperparameter Search
                 ├── Models_EDNN/trial_XXX/*
                 ├── Results_EDNN/trial_XXX/*
                 ├── Models_EDNN/best_trial_models/*
                 ├── Results_EDNN/ednn_hyperparameter_trials.csv
                 └── Results_EDNN/best_config_ednn.yaml
```

The GUI exposes the required actions so that users do not need to launch the Python scripts manually from a terminal.

---

## 3. Dataset context

The MPro-URV dataset contains crystallized SARS-CoV-2 main protease (`Mpro`) complexes bound to non-covalent inhibitors. The target used by this module is the experimental `pIC50` value.

`pIC50` is defined as:

```text
pIC50 = -log10(IC50)
```

where `IC50` is expressed in molar concentration. A higher `pIC50` value corresponds to a more potent inhibitor.

The expected dataset structure is typically:

```text
MPro-URV_Version2/
├── pIC50.txt
├── Ligand/
│   └── Ligand_SDF/
│       ├── <pdb_id>_ligand.sdf
│       ├── ...
└── Protein/
    └── Protein_PDB/
        ├── <pdb_id>_protein.pdb
        ├── ...
```

The exact local path can differ between machines. For that reason, the GUI asks the user to select the input paths instead of hard-coding absolute paths in the source code.

---

## 4. Module directory structure

The recommended and currently used structure is:

```text
EDNN/
├── __init__.py
├── Core/
│   ├── __init__.py
│   ├── ednn_dataset.py
│   ├── ednn_generate_data.py
│   ├── ednn_hyperparameter_search.py
│   ├── ednn_metrics.py
│   ├── ednn_model.py
│   ├── ednn_tester.py
│   └── ednn_trainer.py
├── ui/
│   ├── __init__.py
│   ├── dialogs/
│   │   ├── __init__.py
│   │   ├── batch_test_ednn_dialog.py
│   │   ├── batch_train_ednn_dialog.py
│   │   ├── generate_data_dialog.py
│   │   ├── hyperparameter_search_ednn_dialog.py
│   │   ├── test_ednn_dialog.py
│   │   └── train_ednn_dialog.py
│   └── menus/
│       ├── __init__.py
│       └── menu_EDNN.py
├── utils/
│   ├── __init__.py
│   └── constants.py
└── workers.py
```

### 4.1. Why `__init__.py` files are present

The `__init__.py` files tell Python that the corresponding directories are importable packages. Python 3 can sometimes work without them because of namespace packages, but relying on that behavior is unnecessary and fragile in a project executed from different entry points such as:

```text
main.py
terminal commands
PyCharm
VS Code
PySide6 workers
unit tests
```

The correct filename is:

```text
__init__.py
```

This is incorrect:

```text
_init_.py
```

The files can remain empty.

### 4.2. Why `ednn_dataset.py` may remain empty for now

The current trainer and tester define their own lightweight PyTorch `Dataset` wrappers internally. Therefore, `ednn_dataset.py` is not required for the current execution path.

It is kept in the skeleton to preserve consistency with the overall module organization. A future refactoring should move the duplicated dataset wrapper classes into this file.

---

## 5. EDNN graph data contract

### 5.1. Generated graph files

The data-generation stage writes one PyTorch Geometric graph file per complex:

```text
Graphs_EDNN/
├── <pdb_id_1>.pt
├── <pdb_id_2>.pt
└── ...
```

EDNN graphs are stored separately from EGNN graphs:

```text
Graphs_EGNN/
Graphs_EDNN/
```

Even if both modules currently start from the same URV dataset and generate structurally similar graph objects, they must not share the same output directory. Keeping them separate avoids accidental overwrites and allows each module to evolve independently.

### 5.2. Expected fields in each PyTorch Geometric graph

The EDNN model receives a PyTorch Geometric `Data` object. The expected fields are:

```python
data.x          # node feature matrix
data.pos        # 3D node coordinates
data.edge_index # graph connectivity
data.edge_attr  # optional edge attributes, used if required by the model
data.y          # pIC50 regression target
data.batch      # batch vector automatically created by the DataLoader
```

The communicated EDNN model signature is:

```python
class EDNN(nn.Module):
    def __init__(self, node_dim=12, edge_dim=1, hidden_dim=64):
        ...

    def forward(self, data):
        ...
```

`node_dim=12` indicates that each node is represented by twelve input features. `edge_dim=1` indicates that one scalar edge feature is expected, usually a distance or an equivalent relation. The exact scientific interpretation depends on the implementation kept in `ednn_model.py`.

### 5.3. Graph-generation parameters

Two graph-construction parameters are exposed in the GUI:

```text
Edge Cutoff    = 5.0 Å
Protein Cutoff = 6.0 Å
```

Their roles are different.

`Protein Cutoff` limits the protein atoms retained around the ligand. Protein atoms located farther than the threshold from every ligand atom are excluded. This reduces graph size and focuses the representation on the local interaction pocket.

`Edge Cutoff` defines the maximum distance used to connect graph nodes. Two atoms farther apart than this threshold are not connected by an edge.

These parameters are editable from the GUI because they are part of the graph-construction hypothesis. They must not be silently hard-coded if the user is expected to control them.

---

## 6. Detailed description of the `Core` files

## 6.1. `ednn_generate_data.py`

This file converts the URV dataset into PyTorch Geometric graphs.

It exposes a callable function for GUI integration:

```python
generate_data(
    pic50_file: str,
    ligand_sdf_dir: str,
    protein_pdb_dir: str,
    graphs_dir: str,
    cutoff_edges: float = 5.0,
    cutoff_prot: float = 6.0,
) -> dict
```

The pipeline performs the following operations:

1. read `pIC50.txt`;
2. load ligand structures from SDF files;
3. load protein structures from PDB files;
4. extract ligand atom coordinates and features;
5. extract protein atom coordinates and features;
6. keep only protein atoms located near the ligand;
7. concatenate ligand and protein atoms;
8. create graph edges according to `cutoff_edges`;
9. create edge attributes when required by EDNN;
10. create the PyTorch Geometric `Data` object;
11. save each graph as `<pdb_id>.pt` inside `Graphs_EDNN`;
12. return a summary dictionary for GUI logging.

A critical implementation detail is that the GUI values for `cutoff_edges` and `cutoff_prot` must actually be propagated to the graph-building function. Showing editable GUI fields while silently ignoring them in the backend would be a defect, not a feature.

## 6.2. `ednn_model.py`

This file contains the EDNN architecture.

The rest of the pipeline only depends on the following public contract:

```python
class EDNN(nn.Module):
    def __init__(self, node_dim=12, edge_dim=1, hidden_dim=64):
        ...

    def forward(self, data):
        ...
```

The `forward` method must return one regression prediction per graph. To remain compatible with the trainer and tester, the final output should be flattened:

```python
return output.view(-1)
```

### Corrected issue: `Sequential is not defined`

An early execution error was:

```text
name 'Sequential' is not defined
```

This occurs when code uses:

```python
Sequential(...)
```

without importing `Sequential` explicitly. The recommended form is:

```python
import torch.nn as nn

self.mlp = nn.Sequential(
    nn.Linear(...),
    nn.ReLU(),
    nn.Linear(...),
)
```

Using the `nn.` prefix consistently reduces ambiguity and prevents this class of error.

## 6.3. `ednn_trainer.py`

This file trains EDNN on the predefined splits.

Expected public signature:

```python
train(
    graphs_dir: str,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    output_base: str,
    batch_size: int = 4,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-4,
    hidden_dim: int = 64,
    device: str | None = None,
    seed: int = 42,
)
```

The trainer:

1. automatically selects `cuda` or `cpu` when `device is None`;
2. loads train, validation and test split files;
3. creates PyTorch Geometric `DataLoader` objects;
4. instantiates one EDNN model per split;
5. trains the model;
6. evaluates validation performance after each epoch;
7. applies early stopping;
8. saves the best checkpoint for each split as `best_model.pt`.

Expected output organization:

```text
Models_EDNN/
├── split_00/
│   └── best_model.pt
├── split_01/
│   └── best_model.pt
├── ...
└── split_04/
    └── best_model.pt
```

### Early stopping and `patience`

`patience` is the maximum number of consecutive epochs allowed without validation improvement before training stops.

Example:

```text
epochs   = 150
patience = 20
```

The model may train for at most 150 epochs. If the best validation score is reached at epoch 40 and no improvement occurs during the following 20 epochs, training stops early instead of wasting time until epoch 150.

## 6.4. `ednn_tester.py`

This file evaluates trained EDNN checkpoints.

Expected public signature:

```python
test_model(
    graphs_dir: str,
    test_split_file: str,
    models_dir: str,
    results_dir: str,
    batch_size: int = 4,
    device: str | None = None,
    hidden_dim: int = 64,
)
```

The tester:

1. loads `best_model.pt` for each split;
2. runs inference on the corresponding test split;
3. calculates regression metrics;
4. generates result files and plots;
5. returns a metrics dictionary that can be consumed by the hyperparameter-search code.

Main metrics:

```text
RMSE
Pearson
Spearman
```

Depending on the local version, evaluation may also generate:

```text
metrics_per_split.csv
metrics_summary.csv
scatter_Split XX.png
scatter_global.png
ednn_labels.npy
ednn_preds.npy
```

## 6.5. `ednn_metrics.py`

This file centralizes metric-related helper functions.

A typical helper is:

```python
load_metrics_summary(results_dir: str) -> dict
```

Its purpose is to avoid duplicating CSV parsing logic in several files.

## 6.6. `ednn_hyperparameter_search.py`

This file runs exhaustive grid search for EDNN.

The currently searched hyperparameters are:

```text
learning rate
hidden dimension
batch size
```

The Cartesian product is generated with:

```python
itertools.product(lr_values, hidden_dim_values, batch_size_values)
```

For each combination, the pipeline:

1. creates a dedicated model directory;
2. creates a dedicated result directory;
3. trains EDNN;
4. evaluates EDNN;
5. appends one row to the trial CSV file;
6. compares the current metrics with the best known result;
7. copies the best checkpoints into `best_trial_models`;
8. updates `best_config_ednn.yaml`.

---

## 7. Current hyperparameter-search output structure

The corrected implementation no longer depends on an intermediate temporary-run directory. Trial files are stored directly under the visible model and result roots:

```text
EDNN/
├── Models_EDNN/
│   ├── trial_001/
│   │   ├── split_00/best_model.pt
│   │   ├── ...
│   │   └── split_04/best_model.pt
│   ├── trial_002/
│   ├── ...
│   └── best_trial_models/
│       ├── split_00/best_model.pt
│       ├── ...
│       └── split_04/best_model.pt
└── Results_EDNN/
    ├── ednn_hyperparameter_trials.csv
    ├── best_config_ednn.yaml
    ├── trial_001/
    ├── trial_002/
    └── ...
```

### Why `temp_runs_dir` is no longer used

An earlier version displayed and created a `temp_runs_dir` folder. However, the real trial folders were stored elsewhere, leaving the temporary directory empty or redundant.

The corrected backend stores trial folders directly in:

```python
trial_models_dir = os.path.join(models_root, trial_name)
trial_results_dir = os.path.join(results_root, trial_name)
```

To avoid breaking existing workers immediately, the `temp_runs_dir` argument may remain in the `run_hyperparameter_search(...)` signature. However, it is intentionally unused and must not trigger folder creation.

The GUI no longer displays this field. For compatibility, the dialog can still pass a hidden default value:

```python
"temp_runs_dir": DEFAULT_TEMP_RUNS_DIR,
```

This is a transitional compatibility measure, not a permanent architectural requirement.

---

## 8. Best-configuration selection rule

The primary optimization criterion is:

```text
minimum RMSE
```

If two trials have exactly the same RMSE, the secondary criterion is:

```text
maximum Pearson correlation
```

The selection logic is:

```python
def is_better_result(candidate_metrics, best_metrics):
    if best_metrics is None:
        return True

    if candidate_metrics["RMSE"] < best_metrics["RMSE"]:
        return True

    if candidate_metrics["RMSE"] > best_metrics["RMSE"]:
        return False

    return candidate_metrics["Pearson"] > best_metrics["Pearson"]
```

The rule is stored in `best_config_ednn.yaml` to make the final selection explicit and reproducible.

---

## 9. Total hyperparameter-search duration

The total runtime of the search is stored because it is part of the experimental information requested for the project.

The first implementation stored separate totals for seconds, minutes and hours. That format was redundant and difficult to read. The corrected implementation uses the standard format:

```text
HH:MM:SS
```

Example:

```yaml
hyperparameter_search_time: "01:27:43"
```

Formatting function:

```python
def format_duration_hms(seconds: float) -> str:
    total_seconds = int(round(seconds))

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
```

The timer starts once at the beginning of the complete search:

```python
search_start_time = time.time()
```

The elapsed time is calculated after the complete trial loop:

```python
search_elapsed_seconds = time.time() - search_start_time
search_elapsed_hms = format_duration_hms(search_elapsed_seconds)
```

The time must not be finalized only when a new best trial is found. Otherwise, the YAML file would contain the time elapsed until the best trial was encountered rather than the total search duration.

---

## 10. YAML serialization fix

A major bug was identified during the EDNN search:

```text
yaml.representer.RepresenterError:
('cannot represent an object', np.float64(...))
```

Training and evaluation were successful, but PyYAML could not serialize NumPy scalar types such as `np.float64`. This produced misleading output: a trial could first be appended as `success`, then appended again as `failed_exception` when YAML saving crashed.

The fix is to recursively convert NumPy and Torch values into built-in Python types before writing YAML:

```python
def to_python_builtin(value):
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {k: to_python_builtin(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_python_builtin(v) for v in value]

    if isinstance(value, tuple):
        return tuple(to_python_builtin(v) for v in value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()

    return value
```

Then:

```python
def save_yaml(data, yaml_path):
    clean_data = to_python_builtin(data)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean_data, f, sort_keys=False, allow_unicode=True)
```

---

## 11. Example `best_config_ednn.yaml`

```yaml
model_name: EDNN
status: computed
best_trial:
  trial_id: 7
  trial_name: trial_007
  lr: 0.0005
  hidden_dim: 128
  batch_size: 4
  model_dir: /path/to/EDNN/Models_EDNN/trial_007
  result_dir: /path/to/EDNN/Results_EDNN/trial_007
best_metrics:
  rmse_mean: 0.9778
  pearson_mean: 0.1828
  spearman_mean: 0.2138
selection_rule:
  primary_metric: RMSE
  primary_goal: min
  secondary_metric: Pearson
  secondary_goal: max
artifacts:
  trials_csv: /path/to/EDNN/Results_EDNN/ednn_hyperparameter_trials.csv
  best_models_dir: /path/to/EDNN/Models_EDNN/best_trial_models
hyperparameter_search_time: "00:43:15"
```

---

## 12. GUI integration

## 12.1. `menu_EDNN.py`

`menu_EDNN.py` is the entry point from the main application window.

Expected actions:

```text
Generate Data
Train Model
Train All Models
Evaluate Model
Evaluate All Models (Folder)
```

In the current interface, `Train All Models` means EDNN hyperparameter search. A future UI cleanup should rename it to:

```text
Hyperparameter Search
```

Each menu action:

1. opens the corresponding dialog;
2. collects user input;
3. checks required fields;
4. disables the main window temporarily;
5. starts a background worker;
6. re-enables the main window when the operation finishes;
7. writes success or error information to the application logs.

## 12.2. Why background workers are required

Training and evaluation can take minutes or hours. Running them inside the GUI main thread would freeze the application.

`workers.py` therefore contains `QThread` subclasses:

```text
DBGenerationThread
TrainThread
TrainAllModelsThread
TestThread
TestAllModelsThread
```

Each worker exposes signals such as:

```python
finished_success = Signal(...)
finished_error = Signal(str)
```

Example:

```python
class TrainAllModelsThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def run(self):
        try:
            results = run_hyperparameter_search(**self.params)
            self.finished_success.emit(results)
        except Exception:
            self.finished_error.emit(traceback.format_exc())
```

## 12.3. Temporary GUI disabling

While a long-running operation is active:

```python
self.main_window.setEnabled(False)
```

When it finishes:

```python
self.main_window.setEnabled(True)
```

This prevents the user from launching incompatible concurrent operations accidentally.

---

## 13. Available dialogs

## 13.1. Graph generation dialog

File:

```text
EDNN/ui/dialogs/generate_data_dialog.py
```

Visible fields:

```text
pIC50 File
Ligand SDF Directory
Protein PDB Directory
Output Graphs Directory
Edge Cutoff
Protein Cutoff
```

The first four fields are required. The cutoffs are optional and initialized with default values.

## 13.2. Single-model training dialog

File:

```text
EDNN/ui/dialogs/train_ednn_dialog.py
```

Typical fields:

```text
Graphs Path
Train Split
Validation Split
Test Split
Output Base
Device
Random Seed
Learning Rate
Batch Size
Epochs
Patience
Hidden Dim
```

## 13.3. Hyperparameter-search dialog

File:

```text
EDNN/ui/dialogs/batch_train_ednn_dialog.py
```

Visible fields:

```text
Graphs Path
Train Split
Validation Split
Test Split
Models Root
Results Root
Device
Random Seed
Epochs
Patience
Learning Rate Values
Hidden Dim Values
Batch Size Values
```

The obsolete `Temp Runs Dir` field has been removed from the UI.

### Handling `Device = auto`

The GUI proposes:

```text
auto
cuda
cpu
cuda:0
cuda:1
```

PyTorch does not recognize a literal device named `auto`. The dialog converts it to `None`:

```python
device = self.device_combo.currentText()
if device == "auto":
    device = None
```

The trainer then resolves the device:

```python
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 13.4. Single-model evaluation dialog

File:

```text
EDNN/ui/dialogs/test_ednn_dialog.py
```

Typical fields:

```text
Graphs Path
Test Split
Models Directory
Results Directory
Device
Batch Size
Hidden Dim
```

`Hidden Dim` must match the value used during training for the checkpoint being loaded.

## 13.5. Multi-model evaluation dialog

File:

```text
EDNN/ui/dialogs/batch_test_ednn_dialog.py
```

This dialog evaluates several experiment directories contained inside a common model root directory. It is useful for comparing already-trained runs.

---

## 14. Hyperparameter-search configurations

### 14.1. Initial reasonable sweep

```text
Learning Rate Values: 5e-5,1e-4,5e-4,1e-3
Hidden Dim Values:    32,64,128
Batch Size Values:    2,4,8
Epochs:               50
Patience:             10
Seed:                 42
Device:               auto
```

Number of trials:

```text
4 × 3 × 3 = 36 trials
```

### 14.2. More thorough sweep

```text
Learning Rate Values: 1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3
Hidden Dim Values:    64,128,256
Batch Size Values:    2,4,8
Epochs:               150
Patience:             20
Seed:                 42
Device:               auto
```

Number of trials:

```text
7 × 3 × 3 = 63 trials
```

### 14.3. Smoke test before a long run

Before launching a long search, validate the full pipeline with one very small configuration:

```text
Learning Rate Values: 1e-4
Hidden Dim Values:    64
Batch Size Values:    4
Epochs:               2 or 5
Patience:             1 or 2
```

The purpose is not to obtain meaningful scientific performance. The purpose is to verify that graph loading, model creation, training, evaluation, checkpoint saving, CSV writing and YAML serialization all work.

---

## 15. Complete GUI usage procedure

## Step 1 — Generate EDNN graphs

Open:

```text
EDNN > Generate Data
```

Select:

```text
pIC50 File
Ligand SDF Directory
Protein PDB Directory
Output Graphs Directory
```

Review the cutoffs, then start generation.

Expected output:

```text
EDNN/Graphs_EDNN/*.pt
```

## Step 2 — Run a single training smoke test

Open:

```text
EDNN > Train Model
```

Use a small number of epochs to verify that the model can be instantiated, graphs are loaded correctly and checkpoints are written.

## Step 3 — Evaluate the trained model

Open:

```text
EDNN > Evaluate Model
```

Select the directory containing the trained split folders.

## Step 4 — Run hyperparameter search

Open:

```text
EDNN > Train All Models
```

Enter comma-separated lists:

```text
5e-5,1e-4,5e-4,1e-3
32,64,128
2,4,8
```

The dialog converts these strings into Python lists before starting the worker.

## Step 5 — Inspect the results

Open:

```text
EDNN/Results_EDNN/ednn_hyperparameter_trials.csv
EDNN/Results_EDNN/best_config_ednn.yaml
EDNN/Models_EDNN/best_trial_models/
```

---

## 16. Main dependencies

The module mainly depends on:

```text
Python 3.10
PyTorch
PyTorch Geometric
PySide6
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
RDKit
Biopython
PyYAML
```

Minimal Conda example:

```bash
conda create -n gui_app python=3.10
conda activate gui_app

pip install pyside6 numpy pandas scipy scikit-learn matplotlib pyyaml tqdm
pip install torch torchvision torchaudio
pip install torch-geometric
pip install rdkit biopython
```

The exact PyTorch installation command depends on whether CUDA support is required.

---

## 17. Issues fixed during integration

### 17.1. Incorrect package filename

Incorrect:

```text
_init_.py
```

Correct:

```text
__init__.py
```

### 17.2. Incorrect `bash` naming

Incorrect:

```text
bash_train_ednn_dialog.py
bash_test_ednn_dialog.py
```

Correct:

```text
batch_train_ednn_dialog.py
batch_test_ednn_dialog.py
```

`bash` is a Linux shell. `batch` describes multi-run processing.

### 17.3. Empty temporary folder

`temp_runs_dir` was visible in the GUI while trials were actually stored elsewhere. It has been removed from the visible interface and is no longer used to create redundant folders.

### 17.4. `Sequential is not defined`

Use `nn.Sequential(...)` consistently or import `Sequential` explicitly.

### 17.5. YAML serialization failure for `np.float64`

Convert NumPy scalar values to built-in Python types before calling `yaml.safe_dump(...)`.

### 17.6. `Device = auto`

Convert the GUI string `auto` to `None` before calling the trainer.

### 17.7. Unreadable search-time format

Store one standard field:

```yaml
hyperparameter_search_time: "HH:MM:SS"
```

---

## 18. What is implemented

| Feature | Status | Notes |
|---|---:|---|
| EDNN graph generation | Implemented | Separate output in `Graphs_EDNN` |
| Graph-generation GUI | Implemented | Paths and cutoffs |
| Single-model training | Implemented | Checkpoint saved per split |
| Single-model evaluation | Implemented | Regression metrics |
| Hyperparameter search | Implemented | Grid over learning rate, hidden dimension and batch size |
| Trial CSV output | Implemented | Trial status and error message |
| Best-configuration YAML output | Implemented | NumPy serialization issue fixed |
| Total search duration | Implemented | `HH:MM:SS` format |
| Best-model copy | Implemented | `best_trial_models` |
| Non-blocking GUI execution | Implemented | `QThread` workers |
| Automatic CPU/GPU selection | Implemented | Through `auto -> None` |
| Hidden temporary-run folder | Implemented | Kept only as a compatibility argument |

---

## 19. What is not implemented or remains limited

| Feature | Status | Notes |
|---|---:|---|
| Bayesian optimization | Not implemented | Current search is exhaustive grid search |
| Automatic resume after interruption | Not implemented | A full checkpoint-aware resume mechanism is still missing |
| GUI cancellation button | Not implemented | Workers do not yet support clean interruption |
| Detailed progress bar | Not implemented | Logs remain the primary progress indicator |
| Dynamic total-trial count in GUI | Not implemented | Recommended improvement |
| Strict validation of every path before launch | Partial | Must be strengthened in dialogs |
| Friendly error dialog boxes | Partial | Full tracebacks are currently available in logs |
| Automated unit tests | Not implemented | Recommended before further refactoring |
| Shared `URVGraphDataset` implementation | Not implemented | Could be moved into `ednn_dataset.py` |
| Complete removal of `temp_runs_dir` from every signature | Not implemented | Kept temporarily for compatibility |
| Multi-seed evaluation | Not implemented | More robust but more expensive |
| Python package publication | Not implemented | Module is integrated into the local application |

---

## 20. Recommended improvements

### 20.1. Display the total number of trials in the GUI

The dialog should calculate and display:

```text
Total trials: 36
```

The calculation is trivial:

```python
total_trials = len(lr_values) * len(hidden_dim_values) * len(batch_size_values)
```

### 20.2. Add strict GUI-side validation

Before starting a worker, verify:

```text
- all required paths exist;
- split files are selected;
- hyperparameter lists are not empty;
- every value can be parsed;
- epochs > 0;
- patience > 0.
```

### 20.3. Add an “Open Results Folder” button

After training or search completion, provide:

```text
Open Results Folder
```

### 20.4. Add a GUI or terminal summary table

Display the best trials sorted by RMSE:

```text
Trial | LR | Hidden Dim | Batch Size | RMSE | Pearson | Spearman
```

### 20.5. Remove `temp_runs_dir` completely after compatibility cleanup

Once all imports and calls are checked:

```text
- remove the argument from run_hyperparameter_search;
- remove the constant;
- remove the hidden key from get_inputs;
- verify the worker.
```

### 20.6. Centralize dataset loading

Move the duplicated `URVGraphDataset` wrapper into:

```text
EDNN/Core/ednn_dataset.py
```

---

## 21. Quick verification commands

Run from the project root:

```bash
python -m compileall EDNN
```

Test model import and construction:

```bash
python - <<'PY'
from EDNN.Core.ednn_model import EDNN

model = EDNN()
print(model)
print("EDNN model import: OK")
PY
```

Test pipeline imports:

```bash
python - <<'PY'
from EDNN.Core.ednn_generate_data import generate_data
from EDNN.Core.ednn_trainer import train
from EDNN.Core.ednn_tester import test_model
from EDNN.Core.ednn_hyperparameter_search import run_hyperparameter_search

print("EDNN pipeline imports: OK")
PY
```

Inspect possible `Sequential` issues:

```bash
grep -R "Sequential" EDNN -n
```

Inspect generated files:

```bash
find EDNN -maxdepth 4 -type f | sort
```

---

## 22. Maintenance notes

The EDNN module was intentionally kept parallel to EGNN. That consistency should be preserved whenever possible.

When a generic improvement is added to EDNN, check whether EGNN should receive the same change:

```text
- search-time format;
- YAML conversion;
- auto-device handling;
- path validation;
- folder structure;
- logging;
- trial summaries.
```

However, symmetry must not be forced when the scientific models require different data fields or preprocessing steps. The application architecture can remain consistent while the scientific model implementation stays faithful to its own definition.

---

## 23. Conclusion

The EDNN module is now integrated into the graphical application with a complete workflow: graph generation, training, evaluation and hyperparameter search. PySide6 dialogs collect user input, while long-running operations execute in `QThread` workers so that the GUI remains responsive.

The main integration issues were identified and corrected: incorrect package filenames, `Sequential` namespace errors, NumPy scalar serialization failures in YAML, redundant temporary directories, automatic device handling and unreadable search-time formatting.

The current implementation is suitable for running EDNN experiments on the MPro-URV dataset and keeping a clear record of every hyperparameter trial. The remaining work is mainly ergonomic and structural: stronger input validation, trial-count display, clean cancellation, resume support, dataset-wrapper centralization and automated tests.
