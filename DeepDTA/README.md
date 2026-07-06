# DeepDTA Module — MPro-URV Integration, Training Pipeline, Hyperparameter Search, and GUI Support

## 1. Purpose of this module

This directory contains the DeepDTA part of the molecular analysis application. The original DeepDTA codebase was designed around two benchmark datasets, `davis` and `kiba`. The work documented here extends that codebase so that it can also train on a SARS-CoV-2 main protease dataset named `mpro_urv`.

The goal was not to redesign DeepDTA from scratch. The goal was to preserve the existing model architecture as much as possible, add a clean training layer around it, convert the external MPro-URV dataset into the format expected by DeepDTA, and expose the hyperparameter search from the graphical interface.

The current implementation supports:

- loading `davis`, `kiba`, and `mpro_urv` from a common `DeepDTA/data/` directory;
- converting the external `MPro-URV_Version2` dataset into a DeepDTA-compatible local dataset;
- canonicalizing SMILES with RDKit so that they remain compatible with the original DeepDTA alphabet;
- filtering molecules that exceed the original fixed SMILES length limit;
- converting the official MPro-URV train, validation, and test folds from PDB IDs to integer indices;
- training DeepDTA from a clean Python function rather than executing the old training script directly;
- using predefined dataset folds when they exist, with a fallback to random train/validation/test splitting;
- running a grid search over learning rate and batch size;
- saving checkpoints, CSV reports, YAML summaries, and execution durations;
- launching the hyperparameter search from the PySide6 GUI without freezing the interface.

This README also states what has **not** been implemented. That distinction matters. A README that only lists features is documentation theater.

---

## 2. High-level directory structure

The relevant structure is:

```text
DeepDTA/
├── Core/
│   ├── deepdta_trainer.py
│   ├── deepdta_hyperparameter_search.py
│   └── deepdta_mpro_urv_converter.py
├── data/
│   ├── davis/
│   │   ├── ligands_can.txt
│   │   ├── proteins.txt
│   │   └── Y
│   ├── kiba/
│   │   ├── ligands_can.txt
│   │   ├── proteins.txt
│   │   └── Y
│   └── mpro_urv/
│       ├── ligands_can.txt
│       ├── proteins.txt
│       ├── Y
│       ├── metadata.json
│       └── folds/
│           ├── train_fold_setting1.txt
│           ├── valid_fold_setting1.txt
│           └── test_fold_setting1.txt
├── results/
│   └── deepdta_hpo/
│       └── runs/
│           └── run_YYYYMMDD_HHMMSS/
│               ├── models/
│               │   ├── trial_001/
│               │   │   └── deepdta_best.pt
│               │   ├── trial_002/
│               │   │   └── deepdta_best.pt
│               │   └── ...
│               └── reports/
│                   ├── deepdta_hyperparameter_trials.csv
│                   └── best_config_deepdta.yaml
├── ui/
│   ├── dialogs/
│   │   └── hyperparameter_search_deepdta_dialog.py
│   └── menus/
│       └── menu_DeepDTA.py
├── utils/
│   └── constants.py
├── workers.py
├── data.py
├── model.py
├── train.py
├── predict.py
├── deep.pt
└── environment.yml
```

The files `data.py`, `model.py`, `train.py`, and `predict.py` are part of the original or legacy DeepDTA structure. The new integration is concentrated in `Core/`, `ui/`, `utils/constants.py`, and `workers.py`.

---

## 3. Original DeepDTA assumptions

The original DeepDTA implementation expects three files per dataset:

```text
ligands_can.txt
proteins.txt
Y
```

The meaning of each file is:

| File | Purpose |
|---|---|
| `ligands_can.txt` | JSON dictionary mapping ligand IDs to canonical SMILES strings. |
| `proteins.txt` | JSON dictionary mapping protein IDs to amino-acid sequences. |
| `Y` | Pickled NumPy array containing affinity targets. |

The DeepDTA data loader encodes ligands and proteins with fixed one-hot encodings. In the current code, the expected tensor dimensions are:

| Input | Channels | Maximum length | Tensor shape per sample |
|---|---:|---:|---|
| Ligand SMILES | `62` | `50` | `(62, 50)` |
| Protein sequence | `25` | `600` | `(25, 600)` |

These dimensions are enforced in `DeepDTA/Core/deepdta_trainer.py` inside `prepare_batch()`:

```python
ligand = ligand.reshape(current_batch_size, ligand_channels, ligand_length)
protein = protein.reshape(current_batch_size, protein_channels, protein_length)
```

The integration deliberately preserves these dimensions. That choice avoids silently changing the scientific model while pretending it is still the same baseline.

---

## 4. Why the MPro-URV dataset required a converter

The external MPro-URV dataset does not have the DeepDTA directory layout. Its source structure is different:

```text
MPro-URV_Version2/
├── pIC50.txt
├── Ligand/
│   └── Ligand_SMI/
│       ├── 5RGV_UGG.smi
│       ├── 5RGX_UGP.smi
│       ├── ...
├── Protein/
│   └── Protein_PDB/
│       ├── 5RGV_protein.pdb
│       ├── ...
└── Splits/
    ├── train_index_folder.txt
    ├── valid_index_folder.txt
    └── test_index_folder.txt
```

The raw dataset contains:

- one `pIC50.txt` file mapping PDB IDs to affinity targets;
- one `.smi` file per ligand;
- protein structure files in PDB format;
- predefined folds represented with PDB IDs rather than integer row indices.

DeepDTA cannot consume this source structure directly. A dedicated converter was therefore added:

```text
DeepDTA/Core/deepdta_mpro_urv_converter.py
```

The converter is a preprocessing utility. It is needed when generating or regenerating `DeepDTA/data/mpro_urv/`. It is **not** needed during normal training once the converted files already exist.

The raw MPro-URV dataset can therefore remain outside the Git repository. That is preferable: source datasets are usually large, environment-specific, and not appropriate to duplicate inside the application repository.

---

## 5. MPro-URV conversion pipeline

### 5.1 Command-line usage

Run the converter from the repository root:

```bash
python DeepDTA/Core/deepdta_mpro_urv_converter.py \
  --source-root ../MPro-URV_Version2/MPro-URV_Version2 \
  --output-root DeepDTA/data/mpro_urv
```

Optional explicit protein selection:

```bash
python DeepDTA/Core/deepdta_mpro_urv_converter.py \
  --source-root ../MPro-URV_Version2/MPro-URV_Version2 \
  --output-root DeepDTA/data/mpro_urv \
  --protein-pdb-path ../MPro-URV_Version2/MPro-URV_Version2/Protein/Protein_PDB/5RGV_protein.pdb
```

If `--protein-pdb-path` is omitted, the converter selects the first `.pdb` file found in `Protein/Protein_PDB/` after sorting the directory contents.

### 5.2 Converter responsibilities

The converter performs the following operations:

1. Read `pIC50.txt` and build a dictionary `{pdb_id: target}`.
2. Read all ligand `.smi` files and extract the PDB ID from the filename prefix.
3. Canonicalize every ligand SMILES with RDKit.
4. Remove stereochemical SMILES markers that are not supported by the original DeepDTA alphabet.
5. Filter canonical SMILES longer than 50 characters.
6. Extract one Mpro protein sequence from a PDB file.
7. Verify that the protein sequence length does not exceed the DeepDTA limit of 600 residues.
8. Keep only IDs that exist both in the target file and in the ligand directory.
9. Rebuild the target matrix `Y` with the filtered ligand order.
10. Convert official split PDB IDs to integer indices after filtering.
11. Save the converted dataset files.
12. Save `metadata.json` so the conversion can be audited later.

### 5.3 SMILES canonicalization

The MPro-URV ligand files contain SMILES with stereochemical symbols such as:

```text
@
/
\
```

The original DeepDTA molecular alphabet does not support all of these symbols. Without preprocessing, the one-hot encoder fails with errors such as:

```text
KeyError: '@'
```

The converter uses RDKit:

```python
Chem.MolFromSmiles(smiles)
Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
```

Setting `isomericSmiles=False` removes stereochemical markers that are incompatible with the fixed DeepDTA alphabet.

This is a compatibility decision, not a chemically neutral one. Stereochemical information can matter for molecular activity. The current module therefore implements a DeepDTA-compatible baseline, not a stereochemistry-preserving molecular model.

### 5.4 SMILES length filtering

The original DeepDTA pipeline uses a maximum ligand sequence length of 50 characters. The MPro-URV dataset contains ligands whose canonical SMILES exceed that limit.

The converter removes those ligands instead of modifying the model architecture.

Observed conversion result:

```text
[WARNING] 154 ligands skipped because canonical SMILES length > 50.
[OK] Ligands kept: 224
[OK] Ligands skipped length > 50: 154
[OK] Protein sequence length: 306
[OK] Y shape: (224, 1)
[OK] Target range: 4.0042 -> 7.7048
```

The raw dataset contains 378 ligand-target entries. The DeepDTA-compatible converted subset contains 224 ligands.

This limitation must be stated clearly when reporting DeepDTA performance. Results obtained with this module are results on the **filtered DeepDTA-compatible MPro-URV subset**, not on all 378 raw ligands.

### 5.5 Fold conversion after filtering

The original MPro-URV split files contain PDB IDs. The trainer needs integer indices into the filtered dataset.

Filtering changes the dataset order and removes some IDs. The converter therefore rebuilds each split from the final filtered ligand dictionary.

The conversion warns when official split IDs are skipped because the corresponding ligand was removed:

```text
[WARNING] train: skipped 506 IDs because they were removed during DeepDTA filtering.
[WARNING] valid: skipped 110 IDs because they were removed during DeepDTA filtering.
[WARNING] test: skipped 154 IDs because they were removed during DeepDTA filtering.
```

The number of skipped IDs across all folds can exceed 154 because the same ligand can appear in different fold-specific split lists.

### 5.6 Generated files

After conversion:

```text
DeepDTA/data/mpro_urv/
├── ligands_can.txt
├── proteins.txt
├── Y
├── metadata.json
└── folds/
    ├── train_fold_setting1.txt
    ├── valid_fold_setting1.txt
    └── test_fold_setting1.txt
```

`metadata.json` is not required by the trainer. It exists for traceability and reproducibility. It records information such as:

- source dataset path;
- selected PDB file;
- number of ligands kept;
- number of ligands removed;
- target statistics;
- protein sequence length;
- number of train, validation, and test folds.

---

## 6. Validating the converted dataset

Use the following checks after conversion.

### 6.1 Compile the converter

```bash
python -m compileall DeepDTA/Core/deepdta_mpro_urv_converter.py
```

### 6.2 Confirm that incompatible stereochemical characters were removed

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("DeepDTA/data/mpro_urv/ligands_can.txt")

with path.open() as f:
    ligands = json.load(f)

bad_chars = sorted({c for smi in ligands.values() for c in {"@", "/", "\\"} if c in smi})
print("bad chars:", bad_chars)
print("num ligands:", len(ligands))
PY
```

Expected result:

```text
bad chars: []
num ligands: 224
```

### 6.3 Confirm the target shape and fold index validity

```bash
python - <<'PY'
import json
import pickle
from pathlib import Path

root = Path("DeepDTA/data/mpro_urv")

with open(root / "ligands_can.txt") as f:
    ligands = json.load(f)

with open(root / "Y", "rb") as f:
    y = pickle.load(f)

print("num ligands:", len(ligands))
print("Y shape:", y.shape)
print("max SMILES length:", max(len(s) for s in ligands.values()))

for name in ["train", "valid", "test"]:
    with open(root / "folds" / f"{name}_fold_setting1.txt") as f:
        folds = json.load(f)

    max_index = max(max(fold) for fold in folds)
    print(name, "folds:", len(folds), "max index:", max_index)
PY
```

Expected values from the current converted dataset:

```text
num ligands: 224
Y shape: (224, 1)
max SMILES length: 50
train folds: 5 max index: 223
valid folds: 5 max index: 221
test folds: 5 max index: 223
```

---

## 7. Clean DeepDTA training pipeline

The clean training entry point is:

```text
DeepDTA/Core/deepdta_trainer.py
```

The principal function is:

```python
train(
    dataset_name: str = "davis",
    output_base: str | None = None,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 0.003,
    device: str | None = "auto",
    seed: int = 42,
    val_split: float = 0.1,
    test_split: float = 0.2,
    max_train_batches: int | None = None,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
) -> Dict[str, Any]
```

### 7.1 Supported datasets

The trainer accepts exactly:

```text
davis
kiba
mpro_urv
```

The dataset files are resolved from:

```text
DeepDTA/data/<dataset_name>/
```

Required files:

```text
ligands_can.txt
proteins.txt
Y
```

If one of these files is missing, the trainer raises a `FileNotFoundError` rather than continuing with partial data.

### 7.2 Device resolution

The trainer supports:

```text
auto
cpu
cuda
cuda:0
cuda:1
```

When `device="auto"`, the trainer uses CUDA if available, otherwise CPU.

### 7.3 Reproducibility

The trainer sets random seeds for:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

This improves reproducibility. It does not guarantee perfect bitwise reproducibility across every hardware and CUDA configuration.

### 7.4 Loss function

The trainer uses root mean squared error as the training loss:

```python
class RMSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
```

### 7.5 Evaluation metrics

The current trainer reports:

| Metric | Meaning | Optimization direction |
|---|---|---|
| RMSE | Root mean squared error between predicted and true affinity values. | Lower is better. |
| Pearson | Linear correlation coefficient between predictions and targets. | Higher is better. |

Pearson is returned as `NaN` when it is mathematically undefined, for example when predictions are constant.

### 7.6 Model checkpointing

At the end of every epoch, validation RMSE is computed. When it improves, the full model object is saved:

```text
<output_base>/deepdta_best.pt
```

The checkpoint is later reloaded and evaluated on train, validation, and test loaders.

The current implementation saves the full model object with `torch.save(model, path)`, not only a `state_dict`.

---

## 8. Dataset folds and random splits

### 8.1 Why folds matter

The converted `mpro_urv` dataset contains official train, validation, and test fold files. Using these files is preferable to generating a new random split because it preserves the intended experimental protocol.

A fold is one specific split of the dataset into:

- training samples used to update model weights;
- validation samples used to select the best checkpoint and compare hyperparameters;
- test samples used to estimate final generalization performance.

The MPro-URV conversion currently generates five folds:

```text
fold_index = 0
fold_index = 1
fold_index = 2
fold_index = 3
fold_index = 4
```

The folds differ because the sample allocation changes from one fold to another. A ligand can be in the training set for one fold and in the test set for another.

### 8.2 Automatic fold detection

The trainer checks whether the selected dataset contains:

```text
folds/train_fold_setting1.txt
folds/valid_fold_setting1.txt
folds/test_fold_setting1.txt
```

If `use_dataset_folds=True` and these files exist, the trainer uses them.

Otherwise, it falls back to a random split controlled by:

```text
val_split
test_split
seed
```

### 8.3 Index validation

Before building data loaders, the trainer verifies that:

- train, validation, and test lists are non-empty;
- every index is within `0 <= index < len(dataset)`;
- train, validation, and test sets do not overlap.

This validation caught an earlier bug where fold files still referenced the original 378 ligands after the dataset had been filtered to 224 samples.

### 8.4 When GUI fields are used

When `Use dataset folds` is enabled and fold files exist:

- `Fold index` is used;
- `Validation split` is ignored;
- `Test split` is ignored.

When dataset folds are disabled or unavailable:

- `Fold index` is ignored;
- `Validation split` is used;
- `Test split` is used.

---

## 9. Running a manual smoke test

Use a short run to verify that the training pipeline works:

```bash
python - <<'PY'
from DeepDTA.Core.deepdta_trainer import train

result = train(
    dataset_name="mpro_urv",
    batch_size=8,
    epochs=3,
    lr=0.001,
    device="auto",
    fold_index=0,
    use_dataset_folds=True,
)

print(result)
PY
```

A successful run prints messages similar to:

```text
[DeepDTA] dataset=mpro_urv samples=224 split_mode=dataset_folds fold_index=0 batch_size=8 epochs=3 lr=0.001 device=cpu
[DeepDTA] epoch=001 train_loss=2.083023 val_rmse=0.828382 val_pearson=0.300400
[DeepDTA] epoch=002 train_loss=0.945655 val_rmse=0.904762 val_pearson=0.346269
[DeepDTA] epoch=003 train_loss=0.900354 val_rmse=0.834578 val_pearson=0.370071
```

Observed test metrics from one debug run:

```text
test_rmse = 0.913705102256307
test_pearson = 0.4392372364496478
```

These values are only evidence that the pipeline executes. They are not a final scientific result. Three epochs on one fold are insufficient for a defensible comparison.

---

## 10. Hyperparameter search

The HPO entry point is:

```text
DeepDTA/Core/deepdta_hyperparameter_search.py
```

Main function:

```python
run_hyperparameter_search(
    dataset_name: str,
    output_root: str,
    device: str | None,
    seed: int,
    epochs: int,
    lr_values: list[float],
    batch_size_values: list[int],
    val_split: float = 0.1,
    test_split: float = 0.2,
    max_train_batches: int | None = None,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
) -> Dict[str, Any]
```

### 10.1 Search space

The current grid search varies:

```text
learning rate
batch size
```

The Cartesian product is generated with:

```python
itertools.product(lr_values, batch_size_values)
```

Example:

```text
Learning rates: 0.001,0.00075,0.0005,0.0001
Batch sizes: 4,8,16
```

This produces:

```text
4 × 3 = 12 trials
```

for one selected fold.

### 10.2 Output directory structure

Each HPO execution creates a timestamped run:

```text
DeepDTA/results/deepdta_hpo/runs/run_YYYYMMDD_HHMMSS/
```

Inside it:

```text
models/
    trial_001/
        deepdta_best.pt
    trial_002/
        deepdta_best.pt
    ...
reports/
    deepdta_hyperparameter_trials.csv
    best_config_deepdta.yaml
```

### 10.3 CSV report

The CSV file contains one row per trial:

```text
deepdta_hyperparameter_trials.csv
```

Columns:

| Column | Meaning |
|---|---|
| `trial_id` | Trial identifier such as `trial_001`. |
| `dataset` | Dataset used: `davis`, `kiba`, or `mpro_urv`. |
| `split_mode` | `dataset_folds` or `random`. |
| `fold_index` | Selected fold index when dataset folds are used. |
| `lr` | Learning rate. |
| `batch_size` | Batch size. |
| `epochs` | Number of epochs requested. |
| `train_rmse` | RMSE on training split after reloading the best checkpoint. |
| `train_pearson` | Pearson correlation on training split. |
| `val_rmse` | RMSE on validation split. |
| `val_pearson` | Pearson correlation on validation split. |
| `test_rmse` | RMSE on test split. |
| `test_pearson` | Pearson correlation on test split. |
| `checkpoint_path` | Saved best model path. |
| `duration_seconds` | Trial duration in seconds. |
| `duration_hms` | Trial duration formatted as `HH:MM:SS`. |
| `status` | `success` or `failed_exception`. |
| `error_message` | Truncated exception text when a trial fails. |

### 10.4 YAML summary

The best configuration is saved to:

```text
best_config_deepdta.yaml
```

The selection rule is:

1. minimize validation RMSE;
2. if validation RMSE is exactly tied, maximize validation Pearson.

The test set is **not** used to select the best hyperparameter configuration. Using test RMSE for model selection would leak test information into the optimization procedure.

### 10.5 Error handling

A failed trial does not stop the complete search. The exception is recorded in the CSV and the grid search continues.

This is useful on long experiments: one invalid parameter combination should not discard every other result.

---

## 11. GUI integration

The graphical interface uses PySide6.

Relevant files:

```text
DeepDTA/ui/dialogs/hyperparameter_search_deepdta_dialog.py
DeepDTA/ui/menus/menu_DeepDTA.py
DeepDTA/workers.py
DeepDTA/utils/constants.py
```

### 11.1 Menu entry

The menu class adds:

```text
DeepDTA -> Hyperparameter Search
```

When the user clicks the action:

1. the configuration dialog opens;
2. the dialog returns validated parameters;
3. the menu starts a `TrainAllModelsThread`;
4. the GUI action is disabled during execution;
5. the HPO pipeline runs in the background;
6. the GUI displays a success or error message at the end;
7. the action is re-enabled.

### 11.2 Why a worker thread is used

Training is computationally expensive. Running it directly in the GUI event loop would freeze the application until the search finishes.

`DeepDTA/workers.py` provides:

```python
class TrainThread(QThread)
class TrainAllModelsThread(QThread)
```

`TrainAllModelsThread` executes:

```python
results = run_hyperparameter_search(**self.params)
```

and emits either:

```text
finished_success(dict)
finished_error(str)
```

### 11.3 GUI fields

The current dialog includes:

| Section | Field | Meaning |
|---|---|---|
| Dataset and output | `Dataset` | Choose `davis`, `kiba`, or `mpro_urv`. |
| Dataset and output | `Output root` | Root directory where timestamped HPO runs are written. |
| General configuration | `Device` | `auto`, `cuda`, `cpu`, `cuda:0`, or `cuda:1`. |
| General configuration | `Random seed` | Seed used by Python, NumPy, and PyTorch. |
| General configuration | `Epochs` | Number of training epochs per trial. |
| Split configuration | `Use dataset folds` | Use dataset-provided fold files when available. Recommended for `mpro_urv`. |
| Split configuration | `Fold index` | Select one fold from `0` to `4`. |
| Split configuration | `Validation split` | Used only for random splitting. |
| Split configuration | `Test split` | Used only for random splitting. |
| Debug configuration | `Max train batches` | Per-epoch training batch cap. `0` disables the cap. |
| Search space | `Learning rate values` | Comma-separated float list. |
| Search space | `Batch size values` | Comma-separated integer list. |

### 11.4 Saved GUI settings

The dialog stores the most recent values through `QSettings`:

```python
QSettings("ResearchApp", "DeepDTA_HyperparameterSearch")
```

This prevents the user from re-entering every field after each run.

---

## 12. Default values

Defaults are stored in:

```text
DeepDTA/utils/constants.py
```

Current values:

```python
DEFAULT_DATASET = "davis"
DEFAULT_DEVICE = "auto"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_LR_VALUES = "0.003,0.001,0.0005"
DEFAULT_BATCH_SIZE_VALUES = "4,8"
DEFAULT_MAX_TRAIN_BATCHES = 0
```

The default epoch count is intentionally small enough for quick checks. It should not be mistaken for a final scientific protocol.

---

## 13. Recommended experiment procedure

### 13.1 Fast GUI smoke test

Use this only to check that the GUI, worker thread, dataset loading, fold loading, and result writing are functioning:

```text
Dataset: mpro_urv
Device: cpu or auto
Seed: 42
Epochs: 3 to 5
Use dataset folds: checked
Fold index: 0
Validation split: ignored
Test split: ignored
Max train batches: 2
Learning rate values: 0.003
Batch size values: 4,8
```

This is a debug configuration. It is not a serious training configuration because only two training batches are processed per epoch.

### 13.2 Serious single-fold HPO run

On a laboratory computer with GPU:

```text
Dataset: mpro_urv
Device: cuda:0
Seed: 42
Epochs: 150
Use dataset folds: checked
Fold index: 0
Max train batches: 0
Learning rate values: 0.001,0.00075,0.0005,0.0001
Batch size values: 4,8,16
```

This produces 12 trials for one fold.

### 13.3 Full five-fold protocol

Repeat the serious run with:

```text
Fold index: 0
Fold index: 1
Fold index: 2
Fold index: 3
Fold index: 4
```

Total:

```text
12 trials per fold × 5 folds = 60 training runs
```

For a proper comparison, aggregate the final metrics across folds and report at least mean and standard deviation.

---

## 14. Compilation checks

Compile the complete module:

```bash
python -m compileall DeepDTA
```

This verifies Python syntax. It does not prove that the experiment design is correct, that the folds are scientifically appropriate, or that the metrics are meaningful.

---

## 15. Current implementation status

### 15.1 Implemented

The following components are implemented and tested at least at smoke-test level:

- [x] Common `DeepDTA/data/` root for `davis`, `kiba`, and `mpro_urv`.
- [x] MPro-URV converter with command-line arguments.
- [x] External raw dataset path support through `--source-root`.
- [x] Optional explicit PDB path through `--protein-pdb-path`.
- [x] pIC50 target loading.
- [x] Ligand SMILES loading from `.smi` files.
- [x] Protein sequence extraction from PDB files.
- [x] RDKit validation and canonicalization of SMILES.
- [x] Removal of incompatible stereochemical SMILES markers.
- [x] Filtering of canonical SMILES longer than 50 characters.
- [x] Target matrix regeneration after filtering.
- [x] Official fold conversion from PDB IDs to integer indices.
- [x] Fold filtering after removed ligands.
- [x] Fold index validation.
- [x] Split overlap validation.
- [x] Clean training function in `deepdta_trainer.py`.
- [x] Automatic CUDA/CPU device selection.
- [x] RMSE loss and RMSE/Pearson reporting.
- [x] Best-checkpoint saving based on validation RMSE.
- [x] Grid search over learning rate and batch size.
- [x] Trial CSV report generation.
- [x] Best YAML configuration generation.
- [x] Per-trial duration reporting.
- [x] GUI dataset selector including `mpro_urv`.
- [x] GUI fold controls.
- [x] GUI debug batch cap.
- [x] Worker thread execution to avoid GUI freezing.
- [x] CLI smoke test on `mpro_urv`.
- [x] GUI smoke test on `mpro_urv`.
- [x] Python compilation check.

### 15.2 Not implemented

The following features are **not** currently implemented in the code described here:

- [ ] Automatic `all folds` execution from a single GUI action.
- [ ] Automatic multi-fold aggregation into a summary CSV.
- [ ] Mean and standard deviation report across folds.
- [ ] Automatic multi-seed experiments.
- [ ] Early stopping with patience.
- [ ] Learning-rate scheduling.
- [ ] HPO over architecture parameters such as convolution size, hidden dimensions, or number of layers.
- [ ] Bayesian optimization, Optuna integration, or random search.
- [ ] Dedicated evaluation-only GUI dialog for selecting an existing checkpoint and computing metrics.
- [ ] Scatter-plot generation for predicted versus true affinity values.
- [ ] Spearman correlation reporting.
- [ ] Model explainability tools for DeepDTA.
- [ ] Support for ligand SMILES longer than 50 characters without filtering.
- [ ] Preservation of stereochemical information in the current DeepDTA input representation.
- [ ] Modification of the original DeepDTA architecture to use larger or variable-length ligand sequences.
- [ ] Automated dataset download.
- [ ] Automatic raw dataset discovery outside the repository.
- [ ] Formal scientific comparison against published DeepDTA reference numbers on MPro-URV.

Some of these are reasonable next steps. Others would change the baseline enough that they should be treated as a separate model variant, not smuggled into the same name.

---

## 16. Known limitations

### 16.1 Filtered MPro-URV subset

The most important limitation is the dataset reduction:

```text
378 raw ligands -> 224 DeepDTA-compatible ligands
```

The removed ligands exceed the original maximum SMILES length of 50 characters after canonicalization.

Any report must state that DeepDTA is evaluated on this filtered subset.

### 16.2 Loss of stereochemical information

Using:

```python
isomericSmiles=False
```

removes stereochemical distinctions. This avoids encoder crashes but may discard chemically meaningful information.

### 16.3 Small sample size

A 224-sample dataset is small for a convolutional neural network. Overfitting is a realistic risk. Results should be interpreted across folds rather than from one split.

### 16.4 Full model serialization

Checkpoints currently store the full PyTorch model object rather than a `state_dict`. Full model pickles are easy to reload inside the same codebase but can be more fragile across refactors and PyTorch versions.

### 16.5 Single-fold GUI execution

The GUI exposes one fold index at a time. A full five-fold study currently requires five separate runs.

### 16.6 No early stopping

The trainer saves the best validation checkpoint but does not stop training when validation performance stagnates. Large epoch values may waste compute and increase overfitting risk.

---

## 17. Troubleshooting

### 17.1 `FileNotFoundError: No PDB file found`

Example:

```text
FileNotFoundError: No PDB file found in .../Protein/Protein_PDB
```

Cause: the `--source-root` path does not point to the real MPro-URV dataset root, or the PDB directory is missing.

Fix:

```bash
python DeepDTA/Core/deepdta_mpro_urv_converter.py \
  --source-root ../MPro-URV_Version2/MPro-URV_Version2 \
  --output-root DeepDTA/data/mpro_urv
```

### 17.2 `KeyError: '@'`

Cause: non-compatible stereochemical SMILES symbols remain in `ligands_can.txt`.

Fix: regenerate the dataset with the corrected RDKit canonicalization pipeline and verify:

```text
bad chars: []
```

### 17.3 Invalid fold indices

Example:

```text
ValueError: train split contains invalid indices. Dataset size=224, invalid examples=[...]
```

Cause: fold files were generated before filtering or were not rebuilt after removing long SMILES.

Fix: regenerate the full `DeepDTA/data/mpro_urv/` directory using the corrected converter.

### 17.4 GUI run is suspiciously fast

Check:

```text
Max train batches
```

If it is set to `2`, the trainer only processes two batches per epoch. That is useful for debugging and useless for conclusions.

Use:

```text
Max train batches: 0
```

for a real experiment.

### 17.5 Validation split and test split appear unused

This is expected when:

```text
Use dataset folds: checked
```

and fold files exist. In that mode, the official fold files define train, validation, and test indices.

---

## 18. Suggested next improvements

The most useful next engineering improvements are:

1. Add an `All folds` mode in the GUI.
2. Aggregate metrics across folds automatically.
3. Add early stopping with a configurable patience value.
4. Save `state_dict` checkpoints plus explicit model metadata.
5. Generate prediction CSV files and predicted-vs-true scatter plots.
6. Add Spearman correlation for consistency with the other model modules.
7. Add an evaluation-only workflow for existing checkpoints.
8. Document the exact versions of PyTorch, RDKit, NumPy, and PySide6 used for reproducible runs.
9. Consider a separate `DeepDTA-long` experimental variant if support for longer SMILES is needed.
10. Consider a stereochemistry-preserving representation only as a separate model variant with a clearly documented architectural change.

---

## 19. Minimal end-to-end workflow

### Step 1 — Convert MPro-URV once

```bash
python DeepDTA/Core/deepdta_mpro_urv_converter.py \
  --source-root ../MPro-URV_Version2/MPro-URV_Version2 \
  --output-root DeepDTA/data/mpro_urv
```

### Step 2 — Validate the generated files

```bash
python -m compileall DeepDTA
```

Then run the dataset checks described earlier.

### Step 3 — Run a CLI smoke test

```bash
python - <<'PY'
from DeepDTA.Core.deepdta_trainer import train

print(train(
    dataset_name="mpro_urv",
    batch_size=8,
    epochs=3,
    lr=0.001,
    device="auto",
    fold_index=0,
    use_dataset_folds=True,
))
PY
```

### Step 4 — Run HPO from the GUI

Open:

```text
DeepDTA -> Hyperparameter Search
```

Choose `mpro_urv`, enable dataset folds, select a fold index, configure the search space, and launch.

### Step 5 — Repeat for all five folds

Run folds `0`, `1`, `2`, `3`, and `4` separately until an automatic all-fold mode is implemented.

---

## 20. Final note

The current module is a controlled adaptation of DeepDTA to MPro-URV. It keeps the original fixed input assumptions and adjusts the dataset around them. That makes the baseline easier to understand and less likely to break, but it also creates explicit scientific limitations: filtered molecules, removed stereochemical markers, and single-fold GUI execution.

Those limitations are not hidden. They are the actual boundary of the current code.
