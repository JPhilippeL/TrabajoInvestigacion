# DEAttentionDTA — URV v3b GUI Integration

## 1. Purpose

This module integrates **DEAttentionDTA** into the Molecular Analysis System GUI while preserving the upstream repository as a read-only reference.

DEAttentionDTA is a deep-learning model for protein–ligand binding-affinity prediction based on dynamic embeddings and self-attention. In this project, the model is applied to the official URV v3b dataset of SARS-CoV-2 Mpro inhibitors and predicts the experimental **pIC50** value associated with each protein–ligand complex.

The integration is designed around five principles:

1. preserve the original DEAttentionDTA repository without rewriting it;
2. use the official URV v3b train/validation/test splits;
3. keep dataset preparation reproducible but outside the normal GUI workflow;
4. prevent test-set leakage during hyperparameter search;
5. expose clear GUI actions for validation, training, evaluation, fine-tuning and HPO.

---

## 2. Final module structure

The expected structure inside `TrabajoInvestigacion/` is:

```text
DEAttentionDTA/
├── __init__.py
├── workers.py
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── common.py
│   ├── workflows.py
│   ├── hyperparameter_search.py
│   ├── Prepare_URV_Positions_From_V2_Dataset.py
│   ├── Run_URV_5Splits.py
│   └── Run_URV_Finetune_Pretrained.py
│
├── data/
│   ├── urv_dataset_v3b/
│   │   ├── Info.csv
│   │   └── Splits/
│   │       ├── train_index_folder.txt
│   │       ├── valid_index_folder.txt
│   │       └── test_index_folder.txt
│   │
│   └── urv_dataset_v3b_prepared/
│       ├── affinity_all.csv
│       ├── seq_data_all.csv
│       ├── split_manifest.csv
│       ├── reports/
│       │   ├── dropped_rows.csv
│       │   ├── position_report.csv
│       │   └── position_report.json
│       └── splits/
│           ├── split_01/
│           ├── split_02/
│           ├── split_03/
│           ├── split_04/
│           └── split_05/
│
├── models/
│   ├── pretrained/
│   │   └── DEAttentionDTA.pt
│   ├── from_scratch/
│   ├── finetuned/
│   └── hpo/
│
├── original/
│   ├── LICENSE
│   ├── README.md
│   ├── environment.yml
│   ├── data/
│   ├── pre-code/
│   ├── src/
│   │   └── bestmodel.py
│   └── src-v2/
│
├── outputs/
│   ├── debug/
│   │   ├── prepared_dataset/
│   │   └── pretrained/
│   ├── from_scratch/
│   ├── predictions/
│   ├── pretrained_vs_finetuned/
│   └── hpo/
│
└── ui/
    ├── dialogs/
    └── menus/
```

### What is preserved

The upstream source code is stored under:

```text
DEAttentionDTA/original/
```

The GUI integration loads the upstream model architecture dynamically from:

```text
DEAttentionDTA/original/src/bestmodel.py
```

The original implementation is not rewritten. The adapted URV runners, GUI workflows and HPO logic live under `DEAttentionDTA/core/`.

---

## 3. Dataset and preparation logic

### 3.1 Official URV v3b dataset

The GUI workflows use the official URV v3b dataset and its five predefined splits. The prepared dataset is expected at:

```text
DEAttentionDTA/data/urv_dataset_v3b_prepared/
```

Each split contains six CSV files:

```text
splits/split_XX/
├── seq_train.csv
├── affinity_train.csv
├── seq_valid.csv
├── affinity_valid.csv
├── seq_test.csv
└── affinity_test.csv
```

### 3.2 Why a preparation step exists

DEAttentionDTA requires protein-sequence, ligand-SMILES and pocket-related inputs. The official URV v3b `Info.csv` does not directly contain all pocket information expected by the model.

The script:

```text
DEAttentionDTA/core/Prepare_URV_Positions_From_V2_Dataset.py
```

keeps URV v3b as the official dataset and reconstructs the required `Position` and `Pocket` values using MPro-URV Version2 structural files.

The extraction order is:

1. attempt to recover residue positions from `Interaction/<PDB_ID>_ligand.json`;
2. otherwise compute a distance-based pocket from `Complex/ALIGNED/<PDB_ID>.cif`;
3. discard samples for which no valid pocket can be reconstructed;
4. document discarded samples in `reports/dropped_rows.csv`.

### 3.3 Why preparation is not exposed as a normal GUI button

The prepared dataset is already included in the module. Dataset preparation is therefore not part of ordinary GUI usage.

This is deliberate. Re-running preparation can overwrite a validated prepared dataset and change the usable-sample set. The script remains available for reproducibility, but the normal GUI workflow begins with validation of the existing prepared dataset.

### 3.4 Manual preparation command

Only run this when the prepared dataset must be regenerated.

From the module directory:

```bash
cd DEAttentionDTA

python core/Prepare_URV_Positions_From_V2_Dataset.py \
  --urv-dir data/urv_dataset_v3b \
  --urv-v2-dir "/absolute/path/to/MPro-URV_Version2" \
  --out-dir data/urv_dataset_v3b_prepared \
  --distance-cutoff 4.5
```

The preparation report is written to:

```text
DEAttentionDTA/data/urv_dataset_v3b_prepared/reports/position_report.json
```

---

## 4. GUI entry points

The `DEAttentionDTA` dashboard card exposes:

```text
DEAttentionDTA
├── Validation
│   ├── Validate Prepared Dataset
│   └── Validate Pretrained Checkpoint
├── Train
├── Search
├── Evaluate
└── Compare
```

The hidden internal menu used by the frontend stores the same workflows as:

```text
DEAttentionDTA
├── Train Official Split(s) From Scratch
├── Hyperparameter Search
├── Evaluate Checkpoint on Official Split(s)
├── Compare Pretrained vs Fine-Tuned
└── Advanced Validation
    ├── Validate Prepared Dataset
    └── Validate Pretrained Checkpoint
```

The `Validation` dashboard button opens the `Advanced Validation` submenu. It does not directly execute a validation workflow.

---

## 5. Common fields

Several dialogs reuse the same concepts.

### 5.1 Paths

Paths entered in the GUI are resolved relative to the root of `TrabajoInvestigacion/` unless an absolute path is provided.

Recommended default paths:

| Purpose                      | Path                                                 |
| ---------------------------- | ---------------------------------------------------- |
| Prepared dataset             | `DEAttentionDTA/data/urv_dataset_v3b_prepared`       |
| Pretrained checkpoint        | `DEAttentionDTA/models/pretrained/DEAttentionDTA.pt` |
| Dataset-validation output    | `DEAttentionDTA/outputs/debug/prepared_dataset`      |
| Checkpoint-validation output | `DEAttentionDTA/outputs/debug/pretrained`            |
| From-scratch models          | `DEAttentionDTA/models/from_scratch`                 |
| From-scratch outputs         | `DEAttentionDTA/outputs/from_scratch`                |
| HPO models                   | `DEAttentionDTA/models/hpo`                          |
| HPO outputs                  | `DEAttentionDTA/outputs/hpo`                         |
| Evaluation outputs           | `DEAttentionDTA/outputs/predictions`                 |
| Fine-tuned models            | `DEAttentionDTA/models/finetuned`                    |
| Fine-tuning outputs          | `DEAttentionDTA/outputs/pretrained_vs_finetuned`     |

### 5.2 Device

| Value  | Meaning                                                          |
| ------ | ---------------------------------------------------------------- |
| `auto` | Use CUDA when available; otherwise use CPU.                      |
| `cuda` | Force GPU execution. The operation fails if CUDA is unavailable. |
| `cpu`  | Force CPU execution. Useful for short validation checks.         |

### 5.3 Official splits

The adapted URV workflow uses five predefined splits.

Accepted values:

| Value   | Meaning                     |
| ------- | --------------------------- |
| `all`   | Run splits `1,2,3,4,5`.     |
| `*`     | Alias of `all`.             |
| `1`     | Run split 1 only.           |
| `1,3,5` | Run only the listed splits. |

For smoke tests, use split `1`. For final experiments, use `all` unless a specific protocol requires otherwise.

### 5.4 Checkpoint fold

The `Checkpoint fold` field is relevant when the selected checkpoint contains multiple fold-specific state dictionaries under a top-level `fold_state_dicts` key.

Accepted values:

| Value         | Meaning                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `matching`    | Use the checkpoint fold whose number matches the currently processed URV split. |
| `match`       | Alias of `matching`.                                                            |
| `same`        | Alias of `matching`.                                                            |
| `first`       | Always use the first available checkpoint fold.                                 |
| `0`           | Alias of `first`.                                                               |
| `1`, `2`, ... | Explicitly select one checkpoint fold by number.                                |

For checkpoints containing a single `state_dict`, this field has no practical effect because there is only one set of weights to load.

There is no ensemble mode in this integration. The code loads one state dictionary at a time, matching the behavior already implemented by the source workflow.

### 5.5 Compatibility mode

The checkbox:

```text
Allow non-strict checkpoint loading
```

loads weights with `strict=False` instead of `strict=True`.

Leave it disabled by default. Enable it only after running `Validate Pretrained Checkpoint` and verifying the compatibility report. Non-strict loading is a diagnostic fallback, not a default configuration.

### 5.6 Optimization fields

| Field                   | Meaning                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `Maximum epochs`        | Maximum number of training epochs. Training can stop earlier through early stopping.       |
| `Batch size`            | Number of samples processed per optimization step. Lower values reduce memory consumption. |
| `Learning rate`         | AdamW optimizer step size.                                                                 |
| `Weight decay`          | AdamW L2-style regularization coefficient.                                                 |
| `Early-stopping rounds` | Number of consecutive non-improving validation epochs tolerated before stopping.           |
| `Random seed`           | Base seed used for reproducibility. The runners derive split-specific seeds from it.       |
| `DataLoader workers`    | Number of subprocesses used for loading batches. Use `0` for the safest default.           |

---

## 6. Button reference

## 6.1 Validation → Validate Prepared Dataset

### Purpose

Check that the prepared URV dataset can be read and that a forward pass through the DEAttentionDTA architecture succeeds.

This operation does not train the model and does not load the pretrained checkpoint.

### Fields

| Field              | Meaning                                            | Recommended value                               |
| ------------------ | -------------------------------------------------- | ----------------------------------------------- |
| `Prepared dataset` | Directory containing the prepared split CSV files. | `DEAttentionDTA/data/urv_dataset_v3b_prepared`  |
| `Output directory` | Directory where the debug report is written.       | `DEAttentionDTA/outputs/debug/prepared_dataset` |
| `Device`           | Runtime device.                                    | `cpu` for validation                            |
| `Batch size`       | Number of rows used in the forward pass.           | `2`                                             |

### Expected artifact

```text
DEAttentionDTA/outputs/debug/prepared_dataset/debug_report.json
```

The report includes the device, output shape, target shape, NaN/Inf checks and the number of rows loaded from split 1.

---

## 6.2 Validation → Validate Pretrained Checkpoint

### Purpose

Verify that the pretrained checkpoint can be loaded into the DEAttentionDTA architecture and that a forward pass succeeds after loading the weights.

### Fields

| Field                                 | Meaning                                              | Recommended value                                    |
| ------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| `Prepared dataset`                    | Prepared URV v3b dataset.                            | `DEAttentionDTA/data/urv_dataset_v3b_prepared`       |
| `Checkpoint`                          | Checkpoint to validate.                              | `DEAttentionDTA/models/pretrained/DEAttentionDTA.pt` |
| `Output directory`                    | Directory where the compatibility report is written. | `DEAttentionDTA/outputs/debug/pretrained`            |
| `Checkpoint fold`                     | Fold selector used for aggregate checkpoints.        | `matching`                                           |
| `Device`                              | Runtime device.                                      | `cpu` for validation                                 |
| `Batch size`                          | Number of rows used in the forward pass.             | `2`                                                  |
| `Allow non-strict checkpoint loading` | Allow missing or unexpected state-dict keys.         | Disabled                                             |

### Expected artifact

```text
DEAttentionDTA/outputs/debug/pretrained/debug_pretrained_report.json
```

The report documents the loaded checkpoint format, selected fold, compatibility counts, output shape and NaN/Inf checks.

---

## 6.3 Train → Train Official Split(s) From Scratch

### Purpose

Train one independent DEAttentionDTA model from random initialization for each selected official URV split.

For each split:

1. train on the official training subset;
2. monitor the official validation subset;
3. keep the state with the best validation loss;
4. evaluate the retained model on validation and test subsets;
5. save metrics, predictions, plots and the trained model bundle.

The test subset is evaluated only after model selection within the split.

### Fields

| Field                   | Meaning                                               | Recommended starting value                     |
| ----------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| `Prepared dataset`      | Prepared URV v3b dataset.                             | `DEAttentionDTA/data/urv_dataset_v3b_prepared` |
| `Results directory`     | Metrics, predictions, histories and plots.            | `DEAttentionDTA/outputs/from_scratch`          |
| `Models directory`      | Saved model bundles.                                  | `DEAttentionDTA/models/from_scratch`           |
| `Official splits`       | Splits to train.                                      | `all` for final training; `1` for a smoke test |
| `Device`                | Runtime device.                                       | `cuda` when available                          |
| `Batch size`            | Batch size for training and evaluation.               | `16`                                           |
| `DataLoader workers`    | Data-loading subprocesses.                            | `0`                                            |
| `Random seed`           | Reproducibility seed.                                 | `990721`                                       |
| `Maximum epochs`        | Maximum training duration per split.                  | `100`                                          |
| `Learning rate`         | AdamW learning rate.                                  | `0.0001`                                       |
| `Weight decay`          | AdamW regularization.                                 | `0.0`                                          |
| `Early-stopping rounds` | Stop after this many non-improving validation epochs. | `15`                                           |

### Main artifacts

```text
DEAttentionDTA/models/from_scratch/
└── split_XX/
    └── DEAttentionDTA_URV_v3b_splitXX.pt

DEAttentionDTA/outputs/from_scratch/
├── Summary_5splits.csv
├── Aggregate_metrics.json
└── split_XX/
    ├── train_history.csv
    ├── Predictions_validation.csv
    ├── Predictions_test.csv
    ├── Metrics_validation.csv
    ├── Metrics_test.csv
    ├── Scatter_validation.png
    ├── Scatter_test.png
    └── split_summary.json
```

Scatter plots are generated when Matplotlib is available.

---

## 6.4 Search → Hyperparameter Search

### Purpose

Search for a suitable from-scratch training configuration without using official test subsets.

Each trial trains the selected tuning split or splits using only:

```text
official train subset → optimization
official validation subset → trial comparison
```

Official test subsets are never read during HPO.

### Selection rule

Trials are ranked by:

1. minimum mean validation RMSE;
2. maximum mean validation Pearson as a tie-breaker.

### Fields

| Field                            | Meaning                                                                   | Recommended starting value                                               |
| -------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `Prepared dataset`               | Prepared URV v3b dataset.                                                 | `DEAttentionDTA/data/urv_dataset_v3b_prepared`                           |
| `Trial models root`              | Root folder for models produced by individual trials.                     | `DEAttentionDTA/models/hpo`                                              |
| `Results root`                   | Root folder for HPO reports.                                              | `DEAttentionDTA/outputs/hpo`                                             |
| `Tuning split(s)`                | Official split or splits used for search.                                 | `1` for initial HPO; use more splits only when computationally justified |
| `Device`                         | Runtime device.                                                           | `cuda`                                                                   |
| `Maximum epochs per trial`       | Maximum epochs for each split in each trial.                              | `50`                                                                     |
| `Early-stopping rounds`          | Stop a trial split after consecutive non-improving validation epochs.     | `10`                                                                     |
| `Minimum epochs before stopping` | Prevent early stopping before this epoch.                                 | `5`                                                                      |
| `Minimum improvement`            | Required validation-RMSE decrease to count as improvement.                | `0.0`                                                                    |
| `Gradient clipping norm`         | Clip gradient norm when greater than zero. Use `0.0` to disable clipping. | `0.0`                                                                    |
| `DataLoader workers`             | Data-loading subprocesses.                                                | `0`                                                                      |
| `Random seed`                    | Base seed for reproducibility.                                            | `42`                                                                     |
| `Learning rates`                 | Comma-separated learning-rate candidates.                                 | `0.00005,0.0001`                                                         |
| `Batch sizes`                    | Comma-separated batch-size candidates.                                    | `8,16`                                                                   |
| `Weight decays`                  | Comma-separated regularization candidates.                                | `0.0,0.01`                                                               |

### Number of trials

The number of trials is the Cartesian product of the search-space lists.

Example:

```text
2 learning rates × 2 batch sizes × 2 weight decays = 8 trials
```

If two tuning splits are selected, each of the eight trials trains two independent models and aggregates their validation metrics.

### Main artifacts

```text
DEAttentionDTA/outputs/hpo/
├── latest_run.json
└── deattentiondta_hpo_YYYYMMDD_HHMMSS/
    ├── search_config.json
    ├── deattentiondta_hpo_trials.csv
    ├── best_config_deattentiondta.yaml
    ├── best_config_deattentiondta.json
    ├── best_models/
    └── trial_XXXX/
        ├── trial_summary.json
        ├── split_summaries.csv
        └── split_XX/
            ├── train_history.csv
            ├── split_summary.json
            └── DEAttentionDTA_HPO_trial_XXXX_splitXX.pt
```

Failed trials write:

```text
error_traceback.txt
```

inside the corresponding trial directory.

---

## 6.5 Evaluate → Evaluate Checkpoint on Official Split(s)

### Purpose

Evaluate an existing checkpoint without training. This is the zero-shot evaluation workflow.

For each selected split, the model is evaluated on:

```text
validation subset
and
test subset
```

The training subset is loaded only as part of dataset construction and is not used for optimization.

### Fields

| Field                                 | Meaning                                  | Recommended value                                    |
| ------------------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `Checkpoint`                          | Checkpoint to evaluate.                  | `DEAttentionDTA/models/pretrained/DEAttentionDTA.pt` |
| `Prepared dataset`                    | Prepared URV v3b dataset.                | `DEAttentionDTA/data/urv_dataset_v3b_prepared`       |
| `Results directory`                   | Metrics, predictions and plots.          | `DEAttentionDTA/outputs/predictions`                 |
| `Official splits`                     | Splits to evaluate.                      | `all` or a subset such as `1`                        |
| `Checkpoint fold`                     | Fold selector for aggregate checkpoints. | `matching`                                           |
| `Device`                              | Runtime device.                          | `cuda` when available                                |
| `Batch size`                          | Evaluation batch size.                   | `16`                                                 |
| `DataLoader workers`                  | Data-loading subprocesses.               | `0`                                                  |
| `Allow non-strict checkpoint loading` | Compatibility fallback.                  | Disabled                                             |

### Main artifacts

```text
DEAttentionDTA/outputs/predictions/
├── Summary_5splits_zero_shot_pretrained.csv
├── Aggregate_metrics_zero_shot_pretrained.json
└── split_XX/
    ├── Metrics_validation_zero_shot.csv
    ├── Predictions_validation_zero_shot.csv
    ├── Scatter_validation_zero_shot.png
    ├── Metrics_test_zero_shot.csv
    ├── Predictions_test_zero_shot.csv
    ├── Scatter_test_zero_shot.png
    └── zero_shot_summary.json
```

---

## 6.6 Compare → Compare Pretrained vs Fine-Tuned

### Purpose

Compare the original pretrained checkpoint against a fine-tuned version on the official URV splits.

For each selected split:

1. load the original pretrained checkpoint;
2. evaluate it before fine-tuning;
3. fine-tune on the official training subset;
4. retain the best model according to validation loss;
5. evaluate the retained model after fine-tuning;
6. save before/after metrics and the fine-tuned model.

Each split is initialized independently from the original checkpoint. The fine-tuned model from split 1 is never reused as the initialization for split 2. This avoids contamination between splits.

### Fields

| Field                                 | Meaning                                                 | Recommended starting value                           |
| ------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| `Prepared dataset`                    | Prepared URV v3b dataset.                               | `DEAttentionDTA/data/urv_dataset_v3b_prepared`       |
| `Original checkpoint`                 | Starting pretrained checkpoint.                         | `DEAttentionDTA/models/pretrained/DEAttentionDTA.pt` |
| `Results directory`                   | Before/after metrics, predictions and plots.            | `DEAttentionDTA/outputs/pretrained_vs_finetuned`     |
| `Models directory`                    | Saved fine-tuned bundles.                               | `DEAttentionDTA/models/finetuned`                    |
| `Official splits`                     | Splits to process.                                      | `all` for final comparison; `1` for a smoke test     |
| `Checkpoint fold`                     | Fold selector for aggregate checkpoints.                | `matching`                                           |
| `Device`                              | Runtime device.                                         | `cuda` when available                                |
| `Batch size`                          | Batch size for fine-tuning and evaluation.              | `16`                                                 |
| `DataLoader workers`                  | Data-loading subprocesses.                              | `0`                                                  |
| `Random seed`                         | Reproducibility seed.                                   | `990721`                                             |
| `Allow non-strict checkpoint loading` | Compatibility fallback.                                 | Disabled                                             |
| `Maximum epochs`                      | Maximum fine-tuning duration per split.                 | `150`                                                |
| `Learning rate`                       | AdamW learning rate.                                    | `0.00005`                                            |
| `Weight decay`                        | AdamW regularization.                                   | `0.0`                                                |
| `Early-stopping rounds`               | Stop after consecutive non-improving validation epochs. | `25`                                                 |

### Main artifacts

```text
DEAttentionDTA/models/finetuned/
└── split_XX/
    └── DEAttentionDTA_URV_v3b_pretrained_finetuned_splitXX.pt

DEAttentionDTA/outputs/pretrained_vs_finetuned/
├── Summary_5splits_pretrained_finetuned.csv
├── Aggregate_metrics_pretrained_finetuned.json
└── split_XX/
    ├── finetune_history.csv
    ├── finetune_summary.json
    ├── Metrics_validation_before_finetune.csv
    ├── Predictions_validation_before_finetune.csv
    ├── Scatter_validation_before_finetune.png
    ├── Metrics_test_before_finetune.csv
    ├── Predictions_test_before_finetune.csv
    ├── Scatter_test_before_finetune.png
    ├── Metrics_validation_after_finetune.csv
    ├── Predictions_validation_after_finetune.csv
    ├── Scatter_validation_after_finetune.png
    ├── Metrics_test_after_finetune.csv
    ├── Predictions_test_after_finetune.csv
    └── Scatter_test_after_finetune.png
```

---

## 7. Metrics

The adapted URV workflows report the following primary regression metrics:

| Metric    | Meaning                                                           | Direction         |
| --------- | ----------------------------------------------------------------- | ----------------- |
| `RMSE`    | Root mean squared error between predicted and experimental pIC50. | Lower is better.  |
| `MAE`     | Mean absolute error between predicted and experimental pIC50.     | Lower is better.  |
| `Pearson` | Pearson correlation coefficient between predictions and labels.   | Higher is better. |

Training histories also store validation loss. HPO uses validation RMSE as its primary selection criterion and validation Pearson only as a tie-breaker.

---

## 8. Methodological safeguards

### 8.1 No test leakage during HPO

Hyperparameter search uses official training and validation subsets only. Test subsets are excluded from every HPO trial.

### 8.2 Independent split training

From-scratch training creates a new model for each official split.

Fine-tuning reloads the original pretrained checkpoint independently for each official split.

### 8.3 Validation before long runs

Before any full training or HPO execution, run both validation actions. This avoids wasting GPU time because of a broken path, malformed checkpoint or invalid prepared dataset.

---

## 9. Recommended first-run procedure

Start the GUI from the project root:

```bash
cd ~/Studies/stage/TrabajoInvestigacion
conda activate gui_app
python main.py
```

Then run these checks in order.

### Step 1 — Validate prepared dataset

```text
DEAttentionDTA → Validation → Validate Prepared Dataset
```

Recommended values:

```text
Prepared dataset: DEAttentionDTA/data/urv_dataset_v3b_prepared
Output directory: DEAttentionDTA/outputs/debug/prepared_dataset
Device: cpu
Batch size: 2
```

### Step 2 — Validate pretrained checkpoint

```text
DEAttentionDTA → Validation → Validate Pretrained Checkpoint
```

Recommended values:

```text
Prepared dataset: DEAttentionDTA/data/urv_dataset_v3b_prepared
Checkpoint: DEAttentionDTA/models/pretrained/DEAttentionDTA.pt
Output directory: DEAttentionDTA/outputs/debug/pretrained
Checkpoint fold: matching
Device: cpu
Batch size: 2
Allow non-strict checkpoint loading: disabled
```

### Step 3 — Smoke-test from-scratch training

```text
DEAttentionDTA → Train
```

Use:

```text
Official splits: 1
Maximum epochs: 1
Batch size: 4
Device: cuda
```

Use `cpu` only when CUDA is unavailable.

### Step 4 — Smoke-test HPO

```text
DEAttentionDTA → Search
```

Use exactly one trial:

```text
Tuning split(s): 1
Maximum epochs per trial: 1
Early-stopping rounds: 1
Minimum epochs before stopping: 1
Learning rates: 0.0001
Batch sizes: 4
Weight decays: 0.0
Device: cuda
```

### Step 5 — Smoke-test evaluation

```text
DEAttentionDTA → Evaluate
```

Use:

```text
Official splits: 1
Checkpoint: DEAttentionDTA/models/pretrained/DEAttentionDTA.pt
Checkpoint fold: matching
Device: cuda
```

### Step 6 — Smoke-test fine-tuning

```text
DEAttentionDTA → Compare
```

Use:

```text
Official splits: 1
Maximum epochs: 1
Batch size: 4
Checkpoint: DEAttentionDTA/models/pretrained/DEAttentionDTA.pt
Checkpoint fold: matching
Device: cuda
```

Once these smoke tests pass, launch the real HPO and final experiments.

---

## 10. Persistent GUI settings

Dialog values are saved through `QSettings`. When a dialog is opened again, the previous values are restored.

This is convenient during normal use but can preserve stale paths after restructuring the project.

To reset only DEAttentionDTA dialog settings:

```bash
python - <<'PY'
from PySide6.QtCore import QSettings

applications = [
    "DEAttentionDTA_Debug",
    "DEAttentionDTA_DebugPretrained",
    "DEAttentionDTA_Finetune",
    "DEAttentionDTA_HyperparameterSearch",
    "DEAttentionDTA_PrepareDataset",
    "DEAttentionDTA_Evaluation",
    "DEAttentionDTA_TrainOfficialSplits",
]

for application in applications:
    settings = QSettings("ResearchApp", application)
    settings.clear()
    settings.sync()
    print(f"[CLEARED] ResearchApp / {application}")
PY
```

---

## 11. Troubleshooting

### A dialog opens with a long `QLineEdit.__init__` signature error

Cause: a non-string value was restored from `QSettings` into a text field.

Action: clear the DEAttentionDTA settings using the reset command above. For HPO search-space fields, store comma-separated strings such as:

```text
0.00005,0.0001
```

not Python lists.

### The dashboard button does nothing

Check that the callback key used in `DashboardPage` matches the key returned by `MainWindow.build_dashboard_callbacks()`.

Expected DEAttentionDTA keys:

```text
deattentiondta_validate
deattentiondta_train
deattentiondta_search
deattentiondta_evaluate
deattentiondta_compare
```

### The validation submenu opens only one workflow

The dashboard `Validation` callback must open the contextual submenu rather than directly triggering `Validate Prepared Dataset`.

### CUDA was requested but is unavailable

Use:

```text
Device: auto
```

or:

```text
Device: cpu
```

### Prepared files are missing

Check that all five prepared split folders exist under:

```text
DEAttentionDTA/data/urv_dataset_v3b_prepared/splits/
```

Each split must contain six CSV files: train, validation and test sequence/affinity files.

### Checkpoint loading fails in strict mode

Run:

```text
Validation → Validate Pretrained Checkpoint
```

Inspect:

```text
DEAttentionDTA/outputs/debug/pretrained/debug_pretrained_report.json
```

Use non-strict loading only after reviewing missing and unexpected keys.

---

## 12. Upstream reference

The original DEAttentionDTA repository is retained under:

```text
DEAttentionDTA/original/
```

The upstream reference environment is also preserved:

```text
DEAttentionDTA/original/environment.yml
```

It documents the environment used by the original implementation. The integrated GUI uses the existing project environment, typically:

```bash
conda activate gui_app
```

The integrated module does not require rewriting the upstream files.

---

## 13. Summary

The normal usage sequence is:

```text
Validate prepared dataset
→ Validate pretrained checkpoint
→ Run a short smoke test
→ Run HPO without test leakage
→ Train final from-scratch models
→ Evaluate checkpoints
→ Compare pretrained and fine-tuned models
```

Dataset preparation remains available as a manual reproducibility tool, not as a routine GUI action.
