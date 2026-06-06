# CAPLA Integration for the Molecular Analysis System

## 1. Overview

This module integrates **CAPLA** into the Molecular Analysis System GUI and adapts the original research implementation to the URV SARS-CoV-2 Mpro dataset.

CAPLA is a deep-learning model for protein-ligand binding-affinity prediction. The model consumes three complementary inputs for each protein-ligand complex:

- a global protein feature matrix;
- a binding-pocket feature matrix;
- the ligand SMILES representation.

The adapted module supports four experimental workflows:

1. validation of the prepared URV dataset;
2. training fresh CAPLA models on the official URV v3b splits;
3. evaluation of a CAPLA checkpoint on a prepared dataset;
4. comparison of the original pretrained checkpoint against split-specific fine-tuning;
5. hyperparameter search using official training and validation subsets only.

The integration was designed to remain compatible with the existing PySide6 application while keeping CAPLA isolated from the rest of the GUI backend.

---

## 2. What Was Added

The original CAPLA repository was preserved separately from the GUI-adapted implementation.

```text
CAPLA/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── common.py
│   ├── data_utils.py
│   ├── metrics_utils.py
│   ├── model_adapter.py
│   ├── Debug_Graphs.py
│   ├── Prepare_URV_V3B_CAPLA_Dataset.py
│   ├── Predict_CAPLA.py
│   ├── Run_URV_V3B_5Splits_CAPLA.py
│   ├── Run_CAPLA_Pretrained_vs_Finetuned_URV_V3B.py
│   ├── Train_CAPLA.py
│   └── capla_hyperparameter_search.py
│
├── original/
│   └── src/
│       ├── __init__.py
│       ├── capla.py
│       ├── dataset.py
│       ├── main.py
│       ├── metrics.py
│       ├── self_attention.py
│       └── test.py
│
├── data/
│   ├── urv_dataset/
│   ├── urv_dataset_v3b/
│   └── urv_dataset_v3b_prepared/
│
├── models/
│   ├── pretrained/
│   │   └── best_model.pt
│   ├── from_scratch/
│   ├── finetuned/
│   └── hpo/
│
├── outputs/
│   ├── debug/
│   ├── from_scratch/
│   ├── predictions/
│   ├── pretrained_vs_finetuned/
│   └── hpo/
│
├── ui/
│   ├── __init__.py
│   ├── dialogs/
│   │   ├── __init__.py
│   │   ├── _shared.py
│   │   ├── debug_capla_dialog.py
│   │   ├── train_capla_dialog.py
│   │   ├── test_capla_dialog.py
│   │   ├── finetune_capla_dialog.py
│   │   └── hyperparameter_search_capla_dialog.py
│   └── menus/
│       ├── __init__.py
│       └── menu_CAPLA.py
│
└── workers.py
```

### Separation of responsibilities

The module follows a clear separation:

| Folder | Responsibility |
|---|---|
| `CAPLA/original/src/` | Original research code kept as reference. |
| `CAPLA/core/` | Clean backend used by the GUI and CLI. |
| `CAPLA/data/` | URV source data and prepared CAPLA-compatible datasets. |
| `CAPLA/models/` | Pretrained, trained, fine-tuned and HPO checkpoints. |
| `CAPLA/outputs/` | Metrics, predictions, reports and plots. |
| `CAPLA/ui/` | PySide6 dialogs and CAPLA menu integration. |
| `CAPLA/workers.py` | Background subprocess workers used by the GUI. |

The original `main.py` from CAPLA remains isolated under `CAPLA/original/src/`. It must not replace the main entrypoint of the Molecular Analysis System.

---

## 3. Dataset

The adapted module uses the URV v3b SARS-CoV-2 Mpro dataset.

The dataset contains **378 protein-ligand complexes** with experimentally measured `pIC50` values.

The prepared CAPLA dataset is stored in:

```text
CAPLA/data/urv_dataset_v3b_prepared/
```

Its expected structure is:

```text
urv_dataset_v3b_prepared/
├── affinity_data.csv
├── urv_v3b_smi.csv
├── global/
├── pocket/
├── split_manifest.csv
├── splits/
│   ├── split_01/
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   └── test.csv
│   ├── split_02/
│   ├── split_03/
│   ├── split_04/
│   └── split_05/
└── reports/
    └── prepare_report.json
```

### Required runtime files

The following elements are mandatory for training and evaluation:

```text
affinity_data.csv
urv_v3b_smi.csv
global/
pocket/
splits/
```

`split_manifest.csv` and `reports/prepare_report.json` are not required by the training loop, but they are kept for traceability and reproducibility.

### Official split sizes

| Split | Train | Validation | Test | Total |
|---:|---:|---:|---:|---:|
| 1 | 245 | 57 | 76 | 378 |
| 2 | 245 | 57 | 76 | 378 |
| 3 | 245 | 57 | 76 | 378 |
| 4 | 246 | 57 | 75 | 378 |
| 5 | 246 | 57 | 75 | 378 |

The official splits are preserved exactly. No random replacement split is generated for the main URV v3b experiments.

---

## 4. Dataset Preparation

The script:

```text
CAPLA/core/Prepare_URV_V3B_CAPLA_Dataset.py
```

builds the final `urv_dataset_v3b_prepared/` directory from:

```text
CAPLA/data/urv_dataset/
CAPLA/data/urv_dataset_v3b/
```

It performs the following operations:

1. reads `Info.csv` from the URV v3b source;
2. normalizes PDB identifiers;
3. exports `affinity_data.csv` and `urv_v3b_smi.csv`;
4. reuses existing CAPLA `global/` and `pocket/` matrices;
5. copies or symlinks feature folders;
6. exports the five official train, validation and test splits;
7. validates the generated dataset;
8. writes `split_manifest.csv` and `reports/prepare_report.json`.

### Important limitation

This script does **not** generate the original `global/*.csv` and `pocket/*.csv` matrices from raw molecular structures. It reuses feature matrices that already exist under:

```text
CAPLA/data/urv_dataset/global/
CAPLA/data/urv_dataset/pocket/
```

The historical reports indicate that those features were previously generated using DSSP files and interaction JSON files. The corresponding raw feature-generation script was not included in the received CAPLA package.

### CLI example

Run this only when the prepared dataset must be rebuilt:

```bash
python -m CAPLA.core.Prepare_URV_V3B_CAPLA_Dataset \
  --urv-v3b-dir CAPLA/data/urv_dataset_v3b \
  --source-dataset-dir CAPLA/data/urv_dataset \
  --out-dir CAPLA/data/urv_dataset_v3b_prepared \
  --feature-mode copy
```

Available feature modes:

| Mode | Behaviour |
|---|---|
| `copy` | Copies feature files into the prepared dataset. More robust. |
| `symlink` | Creates symbolic links to save disk space. More fragile if source paths move. |

The current GUI intentionally focuses on experiments rather than data regeneration. The preparation backend remains available from the command line.

---

## 5. Original Checkpoint

The original pretrained CAPLA checkpoint is stored locally in:

```text
CAPLA/models/pretrained/best_model.pt
```

It is used by:

- pretrained-only evaluation;
- pretrained versus fine-tuned comparison;
- split-specific fine-tuning experiments.

The checkpoint is not required for from-scratch training.

Because `.pt` files can be large, they should normally remain local and be excluded from Git unless the repository policy explicitly requires versioning them.

---

## 6. Backend Adaptation

The adapted backend under `CAPLA/core/` was cleaned to work with the Molecular Analysis System.

### Import cleanup

The original adapted files referenced the previous location:

```text
TFM_Implementation.CAPLA
```

Those imports were replaced by:

```text
CAPLA.core
```

### Path cleanup

The code was aligned with the new structure:

```text
CAPLA/core/
CAPLA/data/
CAPLA/models/
CAPLA/outputs/
CAPLA/original/src/
```

### Clean model adapter

`CAPLA/core/model_adapter.py` provides a clean model-loading layer and avoids relying directly on the original research scripts.

This matters because the original code contains analysis-oriented side effects inside forward passes, including NumPy exports. Those behaviours are useful during exploratory research but inappropriate for a GUI production workflow.

### Headless plot generation

The plotting backend must remain non-interactive for subprocess execution:

```python
import matplotlib
matplotlib.use("Agg")
```

This avoids Qt Wayland plugin warnings when scatter plots are generated without displaying a Matplotlib window.

---

## 7. GUI Integration

CAPLA is integrated into the PySide6 menu bar through:

```text
CAPLA/ui/menus/menu_CAPLA.py
```

The final CAPLA menu contains:

```text
CAPLA
├── Validate Prepared Dataset
├── Train Official Split(s) From Scratch
├── Hyperparameter Search
├── Evaluate Model on Prepared Dataset
└── Compare Pretrained vs Fine-Tuned
```

### Why subprocess workers are used

CAPLA operations run through:

```text
CAPLA/workers.py
```

Each GUI action launches a background `QThread`, which starts a separate Python subprocess.

```text
PySide6 GUI
    └── QThread
        └── Python subprocess
            └── CAPLA CLI module
```

This design was chosen because it provides:

- a responsive GUI during long-running operations;
- live log forwarding to the GUI console;
- clearer isolation of backend exceptions;
- compatibility with CPU-only and CUDA environments;
- easier migration to a dedicated Conda environment if required later.

The subprocesses set:

```text
PYTHONUNBUFFERED=1
MPLBACKEND=Agg
```

so logs are streamed immediately and plots are generated headlessly.

---

## 8. CLI Validation

Before GUI integration, the backend was validated directly from the command line.

### 8.1 Prepared-dataset validation

```bash
python -m CAPLA.core.Run_URV_V3B_5Splits_CAPLA \
  --mode debug \
  --device cpu \
  --batch-size 2
```

Validated output:

```text
CAPLA debug OK
  device: cpu
  valid rows: 378
  output_shape: [2]
```

This confirms:

- successful loading of all 378 rows;
- readable global and pocket feature matrices;
- valid SMILES encoding;
- correct model instantiation;
- a successful CPU forward pass;
- valid output shape.

### 8.2 From-scratch smoke test

```bash
python -m CAPLA.core.Run_URV_V3B_5Splits_CAPLA \
  --mode all \
  --splits 1 \
  --device cpu \
  --epochs 1 \
  --batch-size 8 \
  --early-stopping-rounds 1 \
  --min-epochs-before-stopping 1 \
  --output-dir ../outputs/smoke_from_scratch \
  --models-dir ../models/smoke_from_scratch
```

Validated output included:

```text
Training split 01 | train=245 valid=57 test=76 | device=cpu
Finished split 01 | test_RMSE=3.9928 | test_Pearson=0.6720
```

These values are **not** scientific results. The run used a single epoch. Its purpose was to validate:

- train/validation/test loading;
- backward propagation;
- optimizer updates;
- checkpoint writing;
- predictions;
- metrics;
- scatter plots;
- summary files.

### 8.3 Pretrained versus fine-tuned smoke test

```bash
python -m CAPLA.core.Run_CAPLA_Pretrained_vs_Finetuned_URV_V3B \
  --mode all \
  --splits 1 \
  --device cpu \
  --epochs 1 \
  --batch-size 8 \
  --early-stopping-rounds 1 \
  --min-epochs-before-stopping 1 \
  --results-dir ../outputs/smoke_pretrained_vs_finetuned \
  --models-dir ../models/smoke_finetuned
```

Validated output included:

```text
Dataset valid rows: 378
Split 1: pretrained test RMSE=1.288285
Finished split 1: pre_RMSE=1.288285, ft_RMSE=0.994707, delta_RMSE=-0.293579
Comparison completed
```

Again, the one-epoch fine-tuning result is a pipeline check, not a final benchmark.

---

## 9. From-Scratch Training

The main from-scratch entrypoint is:

```text
CAPLA/core/Run_URV_V3B_5Splits_CAPLA.py
```

### Example: full official five-split run

```bash
python -m CAPLA.core.Run_URV_V3B_5Splits_CAPLA \
  --mode all \
  --splits all \
  --device auto \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.0001 \
  --weight-decay 0.01 \
  --early-stopping-rounds 25 \
  --min-epochs-before-stopping 20 \
  --output-dir ../outputs/from_scratch \
  --models-dir ../models/from_scratch
```

### Methodology

For each official split:

1. a fresh CAPLA model is initialized;
2. only the official training subset is used for optimization;
3. the official validation subset is used for checkpoint selection and early stopping;
4. the best validation checkpoint is reloaded;
5. the official test subset is evaluated once;
6. split-specific outputs are saved independently.

A model trained on split 1 is never reused as the starting point for split 2.

### Generated artifacts

```text
CAPLA/outputs/from_scratch/
├── Summary_5splits.csv
├── Aggregate_metrics.json
└── split_01/
    ├── validation/
    │   ├── Predictions_CAPLA_validation.csv
    │   ├── Metrics_CAPLA_validation.csv
    │   └── Scatter_CAPLA_validation.png
    ├── test/
    │   ├── Predictions_CAPLA_test.csv
    │   ├── Metrics_CAPLA_test.csv
    │   └── Scatter_CAPLA_test.png
    ├── train_history.csv
    ├── split_summary.json
    └── training_config.json
```

Models are saved separately:

```text
CAPLA/models/from_scratch/
└── split_01/
    └── CAPLA_URV_v3b_split01.pt
```

---

## 10. Evaluation of a CAPLA Checkpoint

The inference entrypoint is:

```text
CAPLA/core/Predict_CAPLA.py
```

The GUI action is:

```text
CAPLA
└── Evaluate Model on Prepared Dataset
```

It evaluates a selected checkpoint against the complete prepared dataset and writes:

```text
Predictions_CAPLA.csv
Metrics_CAPLA.csv
Scatter_CAPLA.png
```

Typical use cases:

- evaluate the original pretrained checkpoint;
- inspect a from-scratch model;
- inspect an HPO-selected model;
- evaluate a fine-tuned model.

---

## 11. Pretrained Versus Fine-Tuned Comparison

The comparison entrypoint is:

```text
CAPLA/core/Run_CAPLA_Pretrained_vs_Finetuned_URV_V3B.py
```

The GUI action is:

```text
CAPLA
└── Compare Pretrained vs Fine-Tuned
```

### Methodology

For each selected official split:

1. the original checkpoint is loaded;
2. pretrained-only predictions are computed on the official test subset;
3. the original checkpoint is loaded again from disk;
4. the reloaded model is fine-tuned on the official train subset;
5. the best epoch is selected using the official validation subset;
6. the fine-tuned model is evaluated on the same official test subset;
7. both sets of metrics are compared.

The original checkpoint is reloaded independently for every split. A fine-tuned model from split 1 is never reused for split 2.

### Generated artifacts

```text
CAPLA/outputs/pretrained_vs_finetuned/
├── Comparison_per_split.csv
├── Comparison_aggregate.json
├── Comparison_report.md
└── split_01/
    ├── pretrained_original/
    └── finetuned_from_original/
```

Fine-tuned checkpoints are saved under:

```text
CAPLA/models/finetuned/
```

---

## 12. Hyperparameter Search

The CAPLA HPO pipeline was added in:

```text
CAPLA/core/capla_hyperparameter_search.py
```

The GUI dialog is:

```text
CAPLA/ui/dialogs/hyperparameter_search_capla_dialog.py
```

The GUI action is:

```text
CAPLA
└── Hyperparameter Search
```

### Search space

The current implementation searches over:

```text
learning rate
batch size
weight decay
```

The grid is Cartesian. For example:

```text
Learning rates : 0.00005,0.0001
Batch sizes    : 8,16
Weight decays  : 0.0,0.01
```

produces:

```text
2 × 2 × 2 = 8 trials
```

### Methodological rule

The HPO pipeline uses only:

```text
train.csv
valid.csv
```

It never evaluates or selects configurations using official `test.csv` subsets.

This avoids test leakage. The test subsets must remain untouched until the final evaluation stage.

### Selection rule

The best configuration is selected using:

```text
1. minimum mean validation RMSE
2. maximum mean validation Pearson in case of an RMSE tie
```

### Trial isolation

Every trial creates a fresh CAPLA model. Every selected tuning split is trained independently.

### HPO smoke test

The GUI HPO action was validated using:

```text
Tuning split(s)                  : 1
Device                           : cpu
Maximum epochs per trial         : 1
Early-stopping rounds            : 1
Minimum epochs before stopping   : 1
Learning rates                   : 0.0001
Batch sizes                      : 8
Weight decays                    : 0.01
```

The smoke test generated the expected model, history, summary and configuration files.

### Recommended first real search

For the current CPU-only workstation, start with:

```text
Tuning split(s)                  : 1
Device                           : cpu
Maximum epochs per trial         : 50
Early-stopping rounds            : 10
Minimum epochs before stopping   : 5
Learning rates                   : 0.00005,0.0001
Batch sizes                      : 8,16
Weight decays                    : 0.0,0.01
```

After selecting the best configuration, run the standard five-split from-scratch training workflow using the selected hyperparameters.

### HPO outputs

```text
CAPLA/outputs/hpo/
├── latest_run.json
└── capla_hpo_YYYYMMDD_HHMMSS/
    ├── search_config.json
    ├── capla_hpo_trials.csv
    ├── best_config_capla.json
    ├── best_config_capla.yaml
    ├── best_models/
    └── trial_0001/
        ├── split_01/
        │   ├── train_history.csv
        │   └── split_summary.json
        └── trial_summary.json
```

Trial models are stored under:

```text
CAPLA/models/hpo/
└── capla_hpo_YYYYMMDD_HHMMSS/
    └── trial_0001/
        └── split_01/
            └── CAPLA_HPO_trial_0001_split01.pt
```

---

## 13. GUI Workflow

Start the application from the repository root:

```bash
python main.py
```

Recommended workflow:

```text
1. Validate Prepared Dataset
2. Hyperparameter Search
3. Train Official Split(s) From Scratch with the selected configuration
4. Evaluate selected models
5. Compare Pretrained vs Fine-Tuned
```

The dataset-preparation backend should only be used when the prepared dataset must be rebuilt.

---

## 14. CPU and CUDA Behaviour

The current workstation does not have a GPU. CAPLA therefore runs on CPU.

Use:

```text
device = auto
```

or:

```text
device = cpu
```

`auto` is recommended because it uses CPU on the current machine and can automatically use CUDA on a compatible GPU machine later.

If CUDA is explicitly requested on a system without an available GPU, the backend raises an error instead of silently falling back to CPU. This behaviour is intentional.

### Practical consequence

Smoke tests are fast enough on CPU, but full five-split training and large HPO grids can be slow. Avoid launching large experiments before confirming the selected configuration and expected output paths.

---

## 15. Validated State

| Feature | Status |
|---|---|
| Prepared URV v3b dataset loading | Validated |
| 378 valid complexes detected | Validated |
| Global feature matrices | Validated |
| Pocket feature matrices | Validated |
| SMILES encoding | Validated |
| CPU forward pass | Validated |
| From-scratch backward pass | Validated |
| From-scratch checkpoint export | Validated |
| Original pretrained checkpoint loading | Validated |
| Pretrained-only inference | Validated |
| Split-specific fine-tuning | Validated |
| Comparative report generation | Validated |
| GUI subprocess execution | Validated |
| GUI log forwarding | Validated |
| CAPLA HPO smoke test | Validated |
| CUDA execution | Not tested on the current CPU-only workstation |
| Raw DSSP / interaction-JSON feature generation | Not available in the received package |

---

## 16. Git Recommendations

Generated outputs and binary checkpoints should normally remain outside version control.

Recommended `.gitignore` additions:

```gitignore
# CAPLA generated artifacts
CAPLA/outputs/**
CAPLA/models/**/*.pt

# Keep folders when needed
!CAPLA/models/.gitkeep
!CAPLA/models/pretrained/.gitkeep
!CAPLA/models/from_scratch/.gitkeep
!CAPLA/models/finetuned/.gitkeep
!CAPLA/models/hpo/.gitkeep

# Python caches
__pycache__/
*.pyc
```

If the original pretrained checkpoint must be distributed separately, document its expected location:

```text
CAPLA/models/pretrained/best_model.pt
```

---

## 17. Reference

The integrated model is based on:

> Jin, Z., Wu, T., Chen, T., Pan, D., Wang, X., Xie, J., Quan, L., and Lyu, Q.
> **CAPLA: improved prediction of protein-ligand binding affinity by a deep learning approach based on a cross-attention mechanism.**
> *Bioinformatics*, 39(2), 2023.

---

## 18. Summary

The CAPLA integration is operational.

The current implementation provides:

- a cleaned CAPLA backend;
- preservation of the original research code;
- support for the official URV v3b five-split protocol;
- prepared-dataset validation;
- from-scratch training;
- checkpoint evaluation;
- pretrained versus fine-tuned comparison;
- a leakage-safe HPO pipeline;
- PySide6 menu integration;
- background subprocess execution with live GUI logs;
- traceable output files for every workflow.

The next experimental step is to run a real HPO search, select the best configuration using validation metrics only, and then train the final models on all five official splits.
