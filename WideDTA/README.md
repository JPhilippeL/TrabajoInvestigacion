# WideDTA — MPro-URV Adaptation and GUI Integration

## 1. Scope

This module contains the refactored WideDTA implementation used in the molecular binding-affinity application. The objective was to transform the original research scripts into a maintainable Python package that can be trained on the MPro-URV dataset, launched from the command line, and integrated into the PySide6 GUI.

WideDTA predicts protein–ligand binding affinity from textual chemical and biological representations. Unlike DeepDTA, which uses character-based sequence encoding, WideDTA uses a word-based representation and an additional motif branch.

This README documents:

- the original code state;
- the refactor;
- the MPro-URV adaptation;
- the GUI integration;
- the dependency fixes;
- the current limitations;
- the work that is still not implemented.

The focus is honesty: the current module is a usable technical baseline, not a perfect reproduction of the original WideDTA paper.

---

## 2. Initial code state

The original files were:

```text
WideDTA/
├── data_w.py
├── model_w.py
├── train_w.py
├── predict_w.py
└── wide.pt
```

The main problems were:

1. Paths were hardcoded for Davis or KIBA.
2. Tensor reshapes were hardcoded for Davis.
3. `WideCNN` input dimensions were hardcoded for Davis or KIBA.
4. Training parameters were embedded directly in the script.
5. Training, validation and prediction were not cleanly separated.
6. There was no structured hyperparameter search.
7. There was no GUI dialog.
8. There was no background worker.
9. The original checkpoint `wide.pt` was tied to the old hardcoded architecture.

---

## 3. Refactored module structure

```text
WideDTA/
├── __init__.py
├── data.py
├── model.py
├── train.py
├── predict.py
├── workers.py
├── environment.yml
│
├── Core/
│   ├── __init__.py
│   ├── widedta_trainer.py
│   ├── widedta_hyperparameter_search.py
│   └── widedta_mpro_urv_converter.py
│
├── utils/
│   ├── __init__.py
│   └── constants.py
│
├── ui/
│   ├── __init__.py
│   ├── dialogs/
│   │   ├── __init__.py
│   │   └── hyperparameter_search_widedta_dialog.py
│   └── menus/
│       ├── __init__.py
│       └── menu_WideDTA.py
│
├── data/
│   ├── davis/
│   ├── kiba/
│   └── mpro_urv/
│       ├── ligands_can.txt
│       ├── proteins.txt
│       ├── motif2.txt
│       ├── Y
│       ├── metadata.json
│       └── folds/
│           ├── train_fold_setting1.txt
│           ├── valid_fold_setting1.txt
│           └── test_fold_setting1.txt
│
└── results/
    ├── widedta_runs/
    └── widedta_hpo/
        └── runs/
```

Each file has one responsibility:

| File | Responsibility |
|---|---|
| `data.py` | WideDTA-specific dataset loading and preprocessing |
| `model.py` | `WideCNN` model definition |
| `train.py` | Command-line entry point for one training run |
| `predict.py` | Command-line checkpoint evaluation |
| `Core/widedta_trainer.py` | Clean training, validation, testing and checkpoint logic |
| `Core/widedta_hyperparameter_search.py` | Grid search with CSV/YAML reports |
| `Core/widedta_mpro_urv_converter.py` | MPro-URV conversion helper |
| `utils/constants.py` | Default values |
| `workers.py` | PySide6 background threads |
| `ui/dialogs/...` | Hyperparameter-search dialog |
| `ui/menus/...` | WideDTA GUI menu |

---

## 4. DeepDTA and WideDTA are not the same model

They solve the same regression task but consume different input formats.

### DeepDTA

```text
ligands_can.txt
proteins.txt
Y
```

DeepDTA uses character-based sequence encoding.

### WideDTA

```text
ligands_can.txt
proteins.txt
motif2.txt
Y
```

WideDTA uses word-based encoding and three branches:

```text
Ligand SMILES -> DeepSMILES -> ligand words -> one-hot matrix
Protein sequence -> protein words -> one-hot matrix
Motif sequence -> motif words -> one-hot matrix
```

The original token sizes were:

```text
Ligand word length: 8
Protein word length: 3
Motif word length: 3
```

The GUI architecture can be similar to DeepDTA. The dataset conversion cannot simply be copied.

---

## 5. MPro-URV input format

The expected folder is:

```text
WideDTA/data/mpro_urv/
```

Required files:

```text
ligands_can.txt
proteins.txt
motif2.txt
Y
metadata.json
folds/train_fold_setting1.txt
folds/valid_fold_setting1.txt
folds/test_fold_setting1.txt
```

### `ligands_can.txt`

JSON mapping ligand identifiers to canonical SMILES:

```json
{
  "0": "CCOC(=O)...",
  "1": "CNC(=O)..."
}
```

### `proteins.txt`

JSON mapping protein identifiers to sequences. For MPro-URV there is normally one target protein:

```json
{
  "0": "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDD..."
}
```

### `Y`

Pickled affinity matrix containing pIC50 values. For a single MPro target, the expected shape is:

```text
(number_of_ligands, 1)
```

Conceptually:

```python
Y[i][0] = pIC50 of ligand i against MPro
```

### `motif2.txt`

WideDTA requires motif input. The MPro-URV database did not provide the original motif representation used by the original WideDTA implementation.

The current baseline is:

```text
motif2.txt = proteins.txt
```

Creation command:

```bash
python - <<'PY'
import json
from pathlib import Path

dataset_dir = Path("WideDTA/data/mpro_urv")
proteins_path = dataset_dir / "proteins.txt"
motif_path = dataset_dir / "motif2.txt"

with open(proteins_path, "r", encoding="utf-8") as f:
    proteins = json.load(f)

with open(motif_path, "w", encoding="utf-8") as f:
    json.dump(proteins, f, indent=4)

print(f"Created: {motif_path}")
PY
```

This is a deterministic technical baseline. It is not a biologically complete motif extraction pipeline.

---

## 6. `data.py`

`data.py` replaces `data_w.py`.

Responsibilities:

1. Load ligand SMILES.
2. Load protein sequences.
3. Load motif sequences.
4. Load affinity values.
5. Filter sequences.
6. Convert SMILES to DeepSMILES.
7. Tokenize ligands, proteins and motifs.
8. One-hot encode each modality.
9. Build protein–ligand–motif samples.
10. Expose a PyTorch `Dataset`.

Logical sample format:

```python
((ligand_tensor, protein_tensor, motif_tensor), target)
```

Model call order:

```python
output = model(protein, ligand, motif)
```

This matches the original script:

```python
out = model(p, m, mt)
```

where `p` is protein, `m` is ligand and `mt` is motif.

---

## 7. Why the original `WideCNN` was not reusable

The original model was hardcoded for Davis:

```python
self.pconv1 = nn.Conv1d(in_channels=9552, out_channels=16, kernel_size=2, stride=1, padding=1)
self.lconv1 = nn.Conv1d(in_channels=141, out_channels=16, kernel_size=2, stride=1, padding=1)
self.mconv1 = nn.Conv1d(in_channels=2017, out_channels=16, kernel_size=2, stride=1, padding=1)
self.FC1 = nn.Linear(5344, 512)
```

The commented KIBA alternatives were different again.

This design causes errors when a new dataset produces different one-hot dimensions:

```text
expected input to have 9552 channels, but got ...
```

or:

```text
mat1 and mat2 shapes cannot be multiplied
```

The refactor uses dynamic dimensions with lazy PyTorch layers such as:

```python
nn.LazyConv1d(...)
nn.LazyLinear(...)
```

The first forward pass infers the input dimensions.

Important consequence:

```text
wide.pt
```

must not be reused for MPro. A new checkpoint must be trained.

---

## 8. `Core/widedta_trainer.py`

The trainer replaces the old `train_w.py`.

The old script hardcoded:

```python
epochs = 3
optimizer = optim.Adam(model.parameters(), lr=0.003)
```

and Davis reshapes:

```python
m = m.reshape((1, 141, 101))
p = p.reshape((1, 9552, 467))
mt = mt.reshape((1, 2017, 96))
```

The clean trainer separates responsibilities.

Main helper functions:

```python
set_seed(seed)
resolve_device(device)
get_dataset_paths(dataset_name)
get_dataset_fold_paths(dataset_name)
dataset_has_fold_files(dataset_name)
read_fold_file(path)
validate_indices(indices, dataset_size, split_name)
build_dataloaders_from_indices(...)
build_dataloaders_from_fold_files(...)
build_dataloaders(...)
prepare_batch(...)
regression_metrics(...)
evaluate(...)
load_saved_model(...)
train(...)
```

Training flow:

1. Fix seeds.
2. Resolve device.
3. Load selected dataset.
4. Build train, validation and test loaders.
5. Use official folds when available.
6. Create `WideCNN`.
7. Create RMS loss.
8. Create Adam optimizer.
9. Train for the requested epochs.
10. Evaluate validation metrics after each epoch.
11. Save the best checkpoint according to validation RMSE.
12. Reload the best checkpoint.
13. Evaluate train, validation and test metrics.
14. Return a structured result dictionary.

Selection rule:

```text
Primary metric: validation RMSE
Goal: minimize
Secondary metric: validation Pearson
Goal: maximize
```

Returned values include:

```text
train_rmse
train_pearson
val_rmse
val_pearson
test_rmse
test_pearson
checkpoint_path
dataset
batch_size
epochs
lr
device
seed
split_mode
fold_index
```

---

## 9. Dataset folds

MPro-URV already includes:

```text
folds/train_fold_setting1.txt
folds/valid_fold_setting1.txt
folds/test_fold_setting1.txt
```

The trainer should use them.

Recommended configuration:

```text
use_dataset_folds = True
fold_index = 0
```

This is required for a fair comparison with DeepDTA.

---

## 10. Command-line usage

### Check dataset loading

```bash
python - <<'PY'
from WideDTA.Core.widedta_trainer import get_dataset_paths
from WideDTA.data import WideDTADataset

ligand_path, protein_path, motif_path, affinity_path = get_dataset_paths("mpro_urv")

dataset = WideDTADataset(
    ligand_path=ligand_path,
    protein_path=protein_path,
    motif_path=motif_path,
    affinity_path=affinity_path,
)

print("Dataset size:", len(dataset))
print("Input shapes:", dataset.input_shapes())
PY
```

### Smoke test

```bash
python -m WideDTA.train   --dataset mpro_urv   --epochs 1   --batch-size 1   --max-train-batches 2   --use-dataset-folds   --fold-index 0
```

This checks the complete mechanical pipeline:

```text
dataset -> dataloader -> model -> loss -> backward -> checkpoint
```

### Longer training run

```bash
python -m WideDTA.train   --dataset mpro_urv   --epochs 100   --batch-size 1   --lr 0.001   --dropout 0.3   --use-dataset-folds   --fold-index 0
```

### Prediction

```bash
python -m WideDTA.predict   --dataset mpro_urv   --checkpoint WideDTA/results/widedta_runs/widedta_best.pt   --output-csv WideDTA/results/widedta_predictions.csv
```

---

## 11. Hyperparameter search

File:

```text
WideDTA/Core/widedta_hyperparameter_search.py
```

Supported parameters:

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
fold_index
use_dataset_folds
max_train_batches
```

Example search space:

```text
Learning rates: 0.003, 0.001, 0.0005
Batch sizes: 1, 2, 4
Dropout values: 0.3
```

Output structure:

```text
WideDTA/results/widedta_hpo/runs/run_YYYYMMDD_HHMMSS/
├── models/
│   ├── trial_001/
│   ├── trial_002/
│   └── ...
└── reports/
    ├── widedta_hyperparameter_trials.csv
    └── best_config_widedta.yaml
```

CSV columns:

```text
trial_id
dataset
split_mode
fold_index
lr
batch_size
dropout
epochs
train_rmse
train_pearson
val_rmse
val_pearson
test_rmse
test_pearson
checkpoint_path
duration_seconds
duration_hms
status
error_message
```

The YAML file stores the best trial and the selection rule.

---

## 12. GUI integration

WideDTA follows the same GUI architecture as DeepDTA.

### `utils/constants.py`

Stores default values:

```python
DEFAULT_DATASET = "mpro_urv"
DEFAULT_DEVICE = "auto"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_LR_VALUES = "0.003,0.001,0.0005"
DEFAULT_BATCH_SIZE_VALUES = "1,2,4"
DEFAULT_DROPOUT_VALUES = "0.3"
DEFAULT_MAX_TRAIN_BATCHES = 0
```

### `ui/dialogs/hyperparameter_search_widedta_dialog.py`

Exposes:

```text
Dataset
Output root
Device
Random seed
Epochs
Validation split
Test split
Use dataset folds
Fold index
Maximum train batches
Learning-rate values
Batch-size values
Dropout values
```

The dialog returns a validated parameter dictionary through:

```python
get_inputs()
```

### `workers.py`

Uses PySide6 threads:

```python
class TrainThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)

class TrainAllModelsThread(QThread):
    finished_success = Signal(dict)
    finished_error = Signal(str)
```

This prevents the GUI from freezing during training.

### `ui/menus/menu_WideDTA.py`

Adds:

```text
WideDTA -> Hyperparameter Search
```

The action opens the dialog, launches the worker and displays final paths and metrics.

To register the menu in the main application:

```python
from WideDTA.ui.menus.menu_WideDTA import MenuWideDTA
```

Then add it to the main menu bar according to the existing GUI structure.

---

## 13. Environment

Recommended `environment.yml`:

```yaml
name: gui_app

channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - tqdm
  - pyyaml
  - joblib
  - pyside6
  - rdkit
  - biopython
  - pytorch=2.3.1
  - torchvision=0.18.1
  - torchaudio=2.3.1
  - pytorch-cuda=11.8
  - ipython
  - ipykernel
  - jupyterlab
  - lifelines
  - pip:
      - deepsmiles
      - torch-geometric==2.7.0
```

Update:

```bash
conda activate gui_app
conda env update -f environment.yml
```

Do not use `--prune` until the YAML is stable. It removes undeclared packages and can break unrelated modules.

Environment check:

```bash
python - <<'PY'
import torch
import deepsmiles
import torch_geometric
from Bio.PDB import PDBParser
from PySide6.QtWidgets import QApplication
from rdkit import Chem
from lifelines.utils import concordance_index

print("Environment OK")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

---

## 14. Dependency issues already fixed

### Missing `deepsmiles`

Error:

```text
ModuleNotFoundError: No module named 'deepsmiles'
```

Fix:

```bash
python -m pip install deepsmiles
```

### Missing `Bio`

Error:

```text
ModuleNotFoundError: No module named 'Bio'
```

Correct package:

```text
biopython
```

Fix:

```bash
conda install -c conda-forge biopython
```

---

## 15. Implemented work

Implemented at design and code level:

- clean package structure;
- MPro-URV folder support;
- `motif2.txt` support;
- motif baseline creation;
- DeepSMILES preprocessing;
- word-based tokenization;
- dynamic model dimensions;
- clean trainer;
- validation/test support;
- official fold support;
- checkpoint selection by validation RMSE;
- HPO grid search;
- CSV reports;
- YAML best-configuration report;
- PySide6 dialog;
- PySide6 workers;
- WideDTA GUI menu;
- environment cleanup;
- `deepsmiles` addition;
- `biopython` addition;
- CLI smoke-test commands;
- CLI training commands;
- CLI prediction structure.

---

## 16. Not implemented yet

Do not claim these as completed:

### Biological motif extraction

Not implemented:

```text
true biological motif extraction
binding-site motif extraction
domain-aware motif extraction
external motif database integration
```

### Full scientific validation

Not demonstrated yet:

```text
complete 5-fold results
final MPro benchmark
comparison against DeepDTA
comparison against original WideDTA
multi-seed reproducibility
```

### Stable preprocessing vocabulary export

Not implemented:

```text
ligand_vocab.json
protein_vocab.json
motif_vocab.json
unknown-token support
preprocessing_config.yaml
```

### Early stopping

Not implemented:

```text
patience
min_delta
best_epoch
automatic stop
```

### Additional metrics

Still to add:

```text
MSE
MAE
Spearman
R²
Concordance Index
```

### Plot generation

Still to add:

```text
loss_curve.png
val_rmse_curve.png
val_pearson_curve.png
scatter_train.png
scatter_valid.png
scatter_test.png
```

### GUI model comparison

Still to add:

```text
automatic 5-fold batch run
DeepDTA vs WideDTA comparison
central leaderboard
GUI export button
```

---

## 17. Known limitations

1. `motif2.txt` is currently a copy of `proteins.txt`.
2. MPro-URV is a small single-target dataset.
3. Old `wide.pt` checkpoints are incompatible with the refactored architecture.
4. One-hot matrices can be memory-heavy.
5. Vocabulary export is not yet stable for deployment prediction.
6. Dynamic layers improve compatibility but require fresh training.

---

## 18. Recommended next steps

Priority order:

1. Run the dataset-loading test.
2. Run the 1-epoch smoke test.
3. Run one complete fold.
4. Add early stopping.
5. Add complete regression metrics.
6. Save preprocessing vocabularies.
7. Generate plots.
8. Run all official folds.
9. Compare against DeepDTA with identical folds.
10. Replace the motif baseline with a biologically meaningful method.

---

## 19. Final status

The refactor transforms WideDTA from Davis/KIBA-oriented experimental scripts into a structured MPro-URV-compatible module.

The current module is appropriate for:

```text
technical validation
GUI integration
baseline MPro experiments
hyperparameter search
comparison-framework development
```

It is not yet a complete scientific reproduction of the original WideDTA implementation because the motif representation remains approximate and full multi-fold evaluation is still pending.
