# EGNN Module — Detailed README

## 1. Overview

This README documents the work carried out on the **EGNN module** in the project. Its purpose is not only to describe what the code does, but also to explain **how the module was built, why certain decisions were made, what was intentionally avoided, what is already implemented, and what still needs verification**.

The target reader is someone who:
- did **not** write the code,
- does **not** know the history of the module,
- but needs to understand the structure, current behavior, and integration choices.

The EGNN work was not just about running a model from scripts. The real objective was to transform the original EGNN pipeline into a **GUI-integrated module** that follows the architecture already used in the project for other models.

In other words, the work evolved from:
- isolated Python scripts,

to:
- a reusable software component integrated into the main Qt interface.

---

## 2. Main objective of the EGNN work

The module was developed to allow a user to interact with the EGNN workflow from the GUI without manually editing Python files each time.

The intended GUI actions are:
1. **Generate Data**
2. **Train Model**
3. **Train All Models**
4. **Evaluate Model**
5. **Evaluate All Models (Folder)**

For EGNN, these actions correspond to the following logic:

- **Generate Data**: build graph data from raw protein and ligand files.
- **Train Model**: train EGNN over the predefined splits.
- **Train All Models**: perform a hyperparameter search over multiple EGNN configurations.
- **Evaluate Model**: evaluate one EGNN experiment folder.
- **Evaluate All Models (Folder)**: evaluate several experiment folders and summarize the results.

A critical design decision was made here:

> For EGNN, “Train All Models” does **not** mean training different model families. It means training multiple **hyperparameter configurations** of the EGNN model.

This was done because EGNN is not structured like a multi-architecture model zoo. Reusing the wording from the GUI was acceptable, but the internal meaning had to remain coherent with the real EGNN workflow.

---

## 3. Starting point: what originally existed

The original EGNN code was provided as a set of scripts rather than as a clean module ready for GUI integration.

The original files were conceptually organized like this:

- `04_a_DB_Generation_EGNN.py`
- `04_b_Debug_Graphs.py`
- `04_c_Train_EGNN.py`
- `04_d_Predict_EGNN.py`
- `EGNN.py`

Each file had a distinct role.

### 3.1 `04_a_DB_Generation_EGNN.py`
This script was responsible for generating graph data. It uses the raw biological and structural inputs to build graph objects usable by the model.

### 3.2 `04_b_Debug_Graphs.py`
This script was used to inspect or debug generated graph files. It is useful for developer-level validation, but it is not a central user-facing GUI action.

### 3.3 `04_c_Train_EGNN.py`
This script handled the training process over the predefined splits. It created the model, loaded split data, trained across epochs, applied early stopping, and saved the best weights.

### 3.4 `04_d_Predict_EGNN.py`
This script evaluated the trained models, computed metrics, created plots, and wrote CSV summaries.

### 3.5 `EGNN.py`
This file defined the EGNN architecture itself. One of the most important tunable values identified in this file was `hidden_dim`.

---

## 4. What had to be understood before integrating anything

Before writing GUI code, the original pipeline had to be understood precisely. The key questions were:

- Which parameters are true hyperparameters?
- Which files are actual inputs?
- Which files are outputs?
- Which parts of the scripts are reusable?
- Which parts are hardcoded and must be refactored?

This step was important because without it, the GUI would only be a thin decorative layer over badly understood code.

The objective was to avoid exactly that.

---

## 5. Hyperparameters identified in EGNN

At the beginning, there was a distinction to make between:
- real user-controlled hyperparameters,
- and ordinary variables inside the training loop.

The following hyperparameters were identified as meaningful and worth exposing:

- `lr`
- `batch_size`
- `hidden_dim`
- `epochs`
- `patience`
- `device`
- `seed`

Among these, the first three were the primary tuning targets:

- `lr`: controls optimization step size.
- `batch_size`: controls mini-batch size and indirectly memory use and gradient stability.
- `hidden_dim`: controls model capacity inside the EGNN architecture.

Other values such as:
- `optimizer`
- `criterion`
- `best_rmse`
- `patience_counter`

were **not** treated as user-facing hyperparameters. They are part of the training mechanism, not part of the configuration exposed to the user.

---

## 6. First hyperparameter-search workflow

Before the GUI version of hyperparameter search was stabilized, a first standalone search mechanism was designed.

### 6.1 Why this was done

The first search system had a practical goal:

> explore multiple EGNN configurations **without directly rewriting the original scripts each time**.

The approach used was the following:

1. create a temporary run directory,
2. copy the necessary files,
3. patch the copied files with one set of hyperparameters,
4. run training,
5. run prediction,
6. read metrics,
7. write results to CSV,
8. update the best YAML configuration when necessary.

### 6.2 Why this was a valid strategy at that stage

This method was chosen because it offered several benefits:

- the original files stayed intact,
- trials were isolated,
- the search could be resumed or analyzed afterward,
- the pipeline was reproducible,
- it was easy to log success/failure per trial.

### 6.3 Search space used

The initial search grid was:

- `lr`: `5e-5`, `1e-4`, `5e-4`, `1e-3`
- `hidden_dim`: `32`, `64`, `128`
- `batch_size`: `2`, `4`, `8`

Total number of combinations: **36**.

This grid was deliberately moderate. It was centered around the default baseline values and extended in both smaller and larger directions.

### 6.4 Output files generated

The hyperparameter search generated:

- `egnn_hyperparameter_trials.csv`
- `best_config_egnn.yaml`

The CSV stored one row per trial, including:
- trial identifier,
- learning rate,
- hidden dimension,
- batch size,
- mean RMSE,
- mean Pearson,
- mean Spearman,
- status,
- error message.

The YAML stored the best configuration according to the selection rule.

### 6.5 Selection rule

The selection strategy was:

1. minimize **RMSE**,
2. if equal, maximize **Pearson**.

`Spearman` was also stored even though it was not the primary decision variable. It was kept because it gives useful information about ranking consistency.

---

## 7. Metrics used in EGNN evaluation

Three metrics were kept in the pipeline:

- **RMSE**
- **Pearson**
- **Spearman**

### 7.1 RMSE
RMSE is the primary regression metric. It measures the average prediction error magnitude. Lower is better.

### 7.2 Pearson
Pearson measures linear correlation between predicted and real values. Higher is better.

### 7.3 Spearman
Spearman measures rank consistency. It is useful when one wants to know whether the ordering of samples is preserved, even if the exact numeric values are noisy.

### 7.4 Why all three were kept
Keeping only RMSE would give an incomplete picture. A model may have decent RMSE but poor correlation structure, or the opposite. The three metrics together provide a more robust interpretation.

---

## 8. Reading the original EGNN outputs

Once the original scripts were executed correctly, the outputs showed that the training/evaluation pipeline was split-based.

Typical outputs included:

- `Models_EGNN/split_00/best_model.pt`
- `Models_EGNN/split_01/best_model.pt`
- ...

and in results:

- `metrics_per_split.csv`
- `metrics_summary.csv`
- `scatter_Split 00.png`
- `scatter_Split 01.png`
- ...
- `scatter_global.png`
- `egnn_labels.npy`
- `egnn_preds.npy`

This made it clear that the correct interpretation of performance is not based on one split only, but on the aggregated summary across all splits.

That point shaped later design choices in the GUI integration, especially for the evaluation functions.

---

## 9. Why the architecture had to change for the GUI

A decisive correction was made during the conversation.

At first, the work was drifting toward a generic refactor focused on internal code cleanliness. That was not the real target.

The real target was this:

> integrate EGNN into the existing GUI the same way other project modules are integrated.

That means the correct problem is not:
- “How do I make EGNN nicer as a script package?”

but:
- “How do I make EGNN behave like a real module in the project interface?”

This distinction matters, because it changed several decisions:

- use the same modular structure as existing GUI modules,
- expose real user parameters through dialogs,
- use `QThread` workers,
- connect everything through the main menu system.

---

## 10. GUI architecture chosen for EGNN

The module was structured to match the existing GUI architecture used elsewhere in the project.

Target structure:

```text
EGNN/
├── Core/
│   ├── egnn_generate_data.py
│   ├── egnn_model.py
│   ├── egnn_trainer.py
│   ├── egnn_tester.py
│   ├── egnn_metrics.py
│   └── egnn_hyperparameter_search.py
├── ui/
│   ├── dialogs/
│   │   ├── generate_data_dialog.py
│   │   ├── train_egnn_dialog.py
│   │   ├── batch_train_egnn_dialog.py
│   │   ├── test_egnn_dialog.py
│   │   └── batch_test_egnn_dialog.py
│   └── menus/
│       ├── __init__.py
│       └── menu_EGNN.py
├── utils/
│   └── constants.py
└── workers.py
```

This split of responsibilities is intentional.

### 10.1 `Core/`
Contains the actual scientific and computational logic.

### 10.2 `ui/dialogs/`
Contains the windows that let the user provide files, directories, and hyperparameters.

### 10.3 `workers.py`
Contains Qt worker threads that run long jobs without freezing the interface.

### 10.4 `ui/menus/`
Contains the EGNN menu added to the main menu bar.

### 10.5 `utils/constants.py`
Centralizes defaults such as folder names and default values.

This separation was chosen because it prevents the GUI layer from becoming a mess of mixed responsibilities.

---

## 11. What the dialogs were designed to expose

A very important design correction was made here.

At one point, the implementation started drifting toward asking for a generic `dataset_dir`. That would have been lazy and misleading.

The decision was then corrected:

> dialogs must expose the **real files and directories actually needed by the action**, not an artificial abstract root path.

This follows the same spirit as the existing module used as reference.

### 11.1 Generate Data dialog
The Generate Data action should ask for:

- `pIC50.txt`
- `Ligand_SDF` directory
- `Protein_PDB` directory
- graph output directory
- `cutoff_edges`
- `cutoff_protein`

This reflects the real data-generation pipeline.

### 11.2 Train Model dialog
The training action should ask for:

- graph directory,
- train split file,
- validation split file,
- test split file,
- model output base directory,
- learning rate,
- batch size,
- hidden dimension,
- epochs,
- patience,
- device,
- seed.

### 11.3 Train All Models dialog
For EGNN, this was explicitly mapped to hyperparameter search. It should ask for:

- graph directory,
- split files,
- models root,
- results root,
- temp run directory,
- lists of learning rates,
- lists of hidden dimensions,
- lists of batch sizes,
- epochs,
- patience,
- device,
- seed.

### 11.4 Evaluate Model dialog
This action should ask for:

- graph directory,
- test split file,
- models directory,
- results directory,
- batch size,
- hidden dimension,
- device.

### 11.5 Evaluate All Models (Folder) dialog
This action should ask for:

- graph directory,
- test split file,
- models root,
- results root,
- batch size,
- hidden dimension,
- device.

---

## 12. Why `QSettings` was used

The dialogs use `QSettings` so that recent values are preserved.

This is not cosmetic. It matters for usability because:
- the same split files are reused many times,
- graph directories are reused,
- model output directories are reused,
- users should not be forced to browse the same paths every time.

This also helps avoid local hardcoded paths inside the Python code itself.

---

## 13. Meaning of the cutoff parameters

The generation dialog includes two parameters:

- `cutoff_edges`
- `cutoff_protein`

These were explicitly clarified during the work.

### 13.1 `cutoff_protein`
This is used to decide which protein atoms are retained around the ligand before building the graph.

If the value is too small:
- only a small local neighborhood is kept,
- useful context may be lost.

If the value is too large:
- the local environment becomes bigger,
- but more noise may also be introduced.

### 13.2 `cutoff_edges`
This is used after node selection to connect nodes into a graph based on inter-atomic distance.

If the value is too small:
- the graph becomes sparse.

If the value is too large:
- the graph becomes denser.

### 13.3 Conceptual distinction
In short:

- `cutoff_protein` controls **which protein atoms enter the graph**,
- `cutoff_edges` controls **how the kept nodes are connected**.

This distinction is fundamental and was kept explicit in the GUI because these parameters do not play the same role.

---

## 14. Worker-thread design

The EGNN GUI integration uses worker threads because the following tasks are too heavy to run directly in the GUI thread:

- graph generation,
- training,
- hyperparameter search,
- evaluation.

The worker layer uses Qt thread classes such as:

- `DBGenerationThread`
- `TrainThread`
- `TrainAllModelsThread`
- `TestThread`
- `TestAllModelsThread`

The logic is always the same:

1. gather parameters in a dialog,
2. validate the required fields,
3. disable the main window,
4. launch a worker thread,
5. run the corresponding Core function,
6. emit a success or error signal,
7. re-enable the GUI.

This design was chosen to preserve interface responsiveness.

---

## 15. Menu integration

The file `menu_EGNN.py` is the entry point that makes the module visible to the user.

Its role is to:
- create the EGNN menu,
- create the five GUI actions,
- open the correct dialog for each action,
- launch the correct worker,
- log success or error messages.

The current intended actions are:

- `Generate Data`
- `Train Model`
- `Train All Models`
- `Evaluate Model`
- `Evaluate All Models (Folder)`

A language cleanup was also necessary. At one point, the menu inherited Spanish labels from the reference module. This was corrected because the menu should be coherent with the rest of the code and with the intended presentation.

So one explicit cleanup decision was:

> keep the EGNN menu in English.

This avoids mixing English Python naming with Spanish visible menu actions.

---

## 16. Core refactoring decisions

The original EGNN scripts were not designed for GUI workers. They had to be refactored into callable functions.

### 16.1 Refactoring `generate_data(...)`
The data-generation code had to move away from a script-style `main` and become a callable function.

This function was redesigned to receive the real files and directories needed for generation.

A key point here is that this function should not rely on absolute user-specific paths. It should take all required paths as parameters.

### 16.2 Refactoring `train(...)`
The training logic had to be extracted from the original training script and rewritten as a function that:
- receives graph directory,
- receives split files,
- receives model output directory,
- receives training hyperparameters,
- runs the split-based training procedure,
- returns the output directory.

### 16.3 Refactoring `test_model(...)`
The evaluation logic had to be rewritten as a function that:
- loads the correct split models,
- evaluates them,
- computes metrics,
- saves per-split and summary outputs,
- returns summary metrics.

### 16.4 Adding `test_all_models_in_folder(...)`
This function was introduced because the GUI includes an action for evaluating all models in a folder. This was not just a copy of the single-model evaluation: it required iterating over multiple experiment folders and summarizing them into a final CSV.

### 16.5 Refactoring hyperparameter search for GUI use
The original hyperparameter search used script copying and patching. For GUI integration, the better version is to call the internal training and evaluation functions directly.

So the GUI-compatible search function should:
- generate all parameter combinations,
- run training,
- run evaluation,
- compare metrics,
- write trial CSV and best YAML,
- return a dictionary summary.

---

## 17. Environment-related problems encountered

The integration work exposed several environment-related issues.

### 17.1 Wrong Python environment
At some point, tests were launched in one environment while the GUI itself was run in another.

This created import errors because the module dependencies were not installed in the environment actually used by the interface.

### 17.2 Missing packages
The following types of errors were encountered:

- missing `numpy`
- missing `pandas`
- missing `Bio` / `biopython`

This led to a clear conclusion:

> If the GUI imports EGNN code, then the GUI environment must contain all EGNN dependencies.

For this project, that means the `gui_app` environment must include at least:
- `biopython`
- `pandas`
- `torch`
- `pyyaml`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `rdkit`
- possibly `torch-geometric`

This is an integration constraint, not an optional luxury.

---

## 18. Bugs encountered during integration

Several bugs were encountered. They are worth documenting because they reveal where the module is fragile.

### 18.1 YAML key mismatch
A `KeyError: 'paths'` occurred when the hyperparameter-search script expected a certain YAML schema that did not match the actual file. This showed that config-file structure must be validated carefully.

### 18.2 Indentation error in data generation
An `IndentationError` occurred in `egnn_generate_data.py`. This was fixed, but it showed that script-to-function refactoring must be syntax-checked before higher-level integration.

### 18.3 Local fake paths used during testing
A test was run with placeholder paths like `/vrai/chemin/vers/...`, which obviously caused permission or file-creation errors. This was not a logic error in the module, but it confirmed that real file selection through the GUI is necessary.

### 18.4 Parent-window attribute mismatch
A GUI integration error occurred because the menu bar and menu object were not sharing the parent reference exactly as expected. This was fixed by passing and storing the correct parent object.

### 18.5 Naming and language inconsistency
The menu text initially appeared in Spanish because the reference module was copied too literally. This was later corrected. The same principle should be applied consistently across the dialogs if full language coherence is desired.

---

## 19. What is already done

The following has been clearly established or implemented conceptually:

- the EGNN pipeline has been understood,
- the main hyperparameters have been identified,
- a hyperparameter-search strategy was created,
- output metrics and file organization were analyzed,
- a GUI-compatible module structure was chosen,
- dialogs were designed around real data dependencies,
- worker threads were defined for asynchronous execution,
- menu integration was written,
- the role of split-based outputs was clarified,
- the meaning of `cutoff_edges` and `cutoff_protein` was clarified,
- the role of `Train All Models` as hyperparameter search was explicitly decided.

This is not a vague direction anymore. The intended software architecture is now explicit.

---

## 20. What is not fully finished or still needs verification

This part must be stated honestly.

### 20.1 Some Core implementations still need full end-to-end verification
The overall structure and expected signatures are defined, but the final code paths still need to be checked carefully against the exact original EGNN script logic.

That means the following must still be verified in practice:
- every internal helper import,
- every parameter name,
- every split-file assumption,
- every output folder assumption,
- every path naming convention.

### 20.2 The GUI layer can exist before full pipeline validation
It is possible for the menu and dialogs to open correctly while the training or evaluation still fails deeper in the Core layer. This means GUI-level success is not enough.

### 20.3 The debug script is not a full GUI feature
`04_b_Debug_Graphs.py` is not currently a central GUI action. It remains more of a developer utility.

### 20.4 Batch evaluation assumes a folder structure
The folder-level evaluation action assumes that each experiment folder has a layout compatible with the expected EGNN output structure. If the training outputs differ, that code must be adapted.

---

## 21. What was intentionally avoided

Several decisions were made specifically to avoid bad outcomes.

### 21.1 Avoid hardcoded absolute local paths
One of the strongest constraints was that the code should not depend on a machine-specific path such as:

- `/home/mohamed/...`

This was avoided because the code must remain usable on another machine. That is why dialogs expose files and folders directly, and why `QSettings` is used to remember paths between runs.

### 21.2 Avoid rewriting the whole GUI framework
The goal was never to redesign the whole application. The EGNN work had to fit the existing project architecture.

### 21.3 Avoid pretending EGNN is a multi-architecture module
It would have been easy to fake a “multiple models” concept by copying the structure of another module too literally. That would have been misleading. Instead, `Train All Models` was mapped to the only coherent meaning for EGNN: hyperparameter search.

### 21.4 Avoid touching original scripts more than necessary
The first hyperparameter-search design was based on wrappers and patched copies precisely because direct modification of the original scripts was to be minimized.

---

## 22. How the module should be tested properly

Testing should be done in layers.

### 22.1 GUI-level test
Check that:
- the main window opens,
- the EGNN menu appears,
- all five actions are visible,
- each dialog opens,
- each `Select...` button works,
- the `OK` / `Cancel` flow behaves correctly.

### 22.2 Worker-level test
Check that:
- clicking `OK` starts the correct worker,
- the GUI is disabled during the job,
- it is re-enabled at the end,
- errors are logged instead of freezing the window.

### 22.3 Core-level test
Check actual outputs:

#### Generate Data
Expected:
- files appear in `Graphs_EGNN`

#### Train Model
Expected:
- `split_00`, `split_01`, ... folders created
- `best_model.pt` files produced

#### Evaluate Model
Expected:
- `metrics_per_split.csv`
- `metrics_summary.csv`
- scatter figures
- prediction arrays

#### Train All Models
Expected:
- trial directories
- `egnn_hyperparameter_trials.csv`
- `best_config_egnn.yaml`

#### Evaluate All Models (Folder)
Expected:
- one result folder per experiment
- one final summary CSV

This layered testing strategy is important because otherwise one ends up mixing GUI bugs, worker bugs, and scientific-logic bugs into one indistinguishable mess.

---

## 23. Suggested future improvements

Several improvements are possible and realistic.

### 23.1 Full consistency audit
A final pass should check:
- imports,
- parameter names,
- output names,
- folder conventions,
- dialog key names,
- worker call signatures.

### 23.2 Better input validation in dialogs
Dialogs could validate:
- missing files,
- wrong file extensions,
- nonexistent folders,
- malformed hyperparameter lists,
- invalid numeric ranges.

### 23.3 Progress reporting
Long jobs should emit progress signals for:
- split number,
- epoch number,
- trial number.

### 23.4 Better GUI result display
Instead of only writing logs and files, the interface could display directly:
- best RMSE,
- best hyperparameter configuration,
- location of saved plots,
- summary tables.

### 23.5 Reuse the same design for EDNN
A large part of the GUI architecture created here can be reused for EDNN later:
- same dialog logic,
- same worker logic,
- same menu integration style,
- same file-based outputs.

---

## 24. Final conclusion

This EGNN module is not just a set of scripts anymore. The work done here established the transition toward a real integrated project component.

The most important result of this work is not a single file or a single function. It is the fact that the module now has:

- a clear architecture,
- a defined GUI entry point,
- identified user-facing hyperparameters,
- a consistent interpretation of batch training,
- a path toward reproducible evaluation and hyperparameter search,
- and an explicit distinction between what is already implemented and what still requires final validation.

The essential idea behind all the decisions made here is the following:

> The goal was not only to make EGNN run, but to make EGNN understandable, controllable, and integrable for users of the project GUI.

That is what this module is supposed to become, and this README is meant to document that process honestly.
