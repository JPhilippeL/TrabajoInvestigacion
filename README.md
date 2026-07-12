# Molecular Analysis System — Backend Documentation

## Project Overview

This repository contains backend modules for protein-ligand binding-affinity prediction experiments, mainly around SARS-CoV-2 Mpro / MPro-URV datasets. It brings together graph-based models, sequence-based models, structure-aware models, and matrix-based models.

The backend code supports data generation and preprocessing, feature generation, model training, hyperparameter search, checkpoint evaluation, metrics, and reports. The root-level overview here is intentionally concise; model-specific README files remain the best source for implementation details.

## General Backend Workflow

```text
Generate Data -> Search -> Train -> Evaluate
```

| Stage | Purpose |
|---|---|
| Generate Data | Convert raw molecular data into model-specific inputs such as graph objects, sequence files, matrix features, or prepared split files. |
| Search | Run hyperparameter search using validation metrics only. |
| Train | Train the selected model on official or configured train/validation/test splits. |
| Evaluate | Load trained checkpoints or model bundles and compute predictions, metrics, and reports. |

`Train` can also be used independently with a manually selected configuration, for example for smoke tests or controlled experiments. However, for a complete experimental protocol, hyperparameter search should normally be performed before the final training stage.

## Installation

The main environment definition is `environment.yml`. The commands below use `gui_app` as the environment name because that is the shared project convention. The name can be changed if needed, but commands in this README assume `gui_app`.

```bash
conda env create -n gui_app -f environment.yml
conda activate gui_app
```

For an existing environment:

```bash
conda env update -n gui_app -f environment.yml
conda activate gui_app
```

Some PyTorch Geometric and CUDA-related packages may require compatible wheels depending on the machine, installed PyTorch version, and CUDA runtime.

## Environment Checks

After activating the environment, run:

```bash
python -c "import torch; print(torch.__version__)"
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "import torch_geometric; print('PyG OK')"
python -c "import numpy, pandas, sklearn; print('Scientific stack OK')"
```

Optional tools for specific structural or charge-generation workflows:

```bash
which mkdssp || which dssp
which pdb2pqr || which pdb2pqr30
which obabel
```

These tools are needed only by workflows that regenerate secondary-structure, pocket, or charge-based features.

## Project Structure

```text
TrabajoInvestigacion/
├── environment.yml
├── README.md
├── README_BACKEND.md
├── EGNN/
├── EDNN/
├── DeepDTA/
├── WideDTA/
├── DCML/
├── CAPLA/
├── DEAttentionDTA/
├── GNNs/
├── models/
└── ...
```

Each model folder contains its own backend logic, data preparation, training/search/evaluation scripts, outputs, and documentation. Generated data and results are usually stored in model-specific folders such as `data/`, `models/`, `outputs/`, `results/`, `Graphs_*`, `Models_*`, or `Results_*`.

## Dataset Note

Raw datasets such as MPro-URV may not be committed because of size, confidentiality, or path-specific constraints. Backend scripts and functions usually require explicit dataset roots, prepared dataset roots, feature roots, checkpoint paths, or output roots.

A common raw MPro-v2-like structure is:

```text
MPro-URV_Version2/
├── pIC50.txt
├── Ligand/
├── Protein/
└── Splits/
```

Avoid hardcoded machine-specific absolute paths in reusable scripts, configuration files, and documentation.

## Model Modules

### EGNN

EGNN is a graph-based model using molecular and protein graph data. Its data generation uses MPro-URV-style inputs such as `pIC50.txt`, ligand SDF files, and protein PDB files. Graph construction includes cutoffs for local protein neighborhoods and graph connectivity, so geometric and structural information are part of the input representation.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: generated graph quality depends on raw structural files, cutoff settings, and compatible PyTorch Geometric objects.

Detailed documentation: [EGNN/README.md](EGNN/README.md)

### EDNN

EDNN is an edge-aware graph neural model related to EGNN. It uses PyTorch Geometric graph objects with node features, positions, edge indices, optional edge attributes, labels, and batch vectors.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: graph fields and edge features must match the model contract; graph output folders should remain separate from EGNN outputs.

Detailed documentation: [EDNN/README.md](EDNN/README.md)

### DeepDTA

DeepDTA is a sequence-based model using ligand SMILES and protein sequences. Its compatible dataset layout uses `ligands_can.txt`, `proteins.txt`, and `Y`. The MPro-URV converter uses RDKit canonicalization and filtering, so a filtered MPro-compatible subset may be produced when ligands exceed the original model constraints.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: fixed sequence encodings and maximum lengths are inherited from the original DeepDTA setup.

Detailed documentation: [DeepDTA/README.md](DeepDTA/README.md)

### WideDTA

WideDTA is a sequence/subsequence model using ligand, protein, and motif-style branches. Its MPro-URV-compatible data includes `ligands_can.txt`, `proteins.txt`, `motif2.txt`, and `Y`. The current `motif2` handling is a technical baseline aligned with protein data rather than a full biological motif extraction pipeline.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: older `wide.pt` checkpoints may be incompatible when architecture dimensions are refactored.

Detailed documentation: [WideDTA/README.md](WideDTA/README.md)

### DCML

DCML is a matrix-based model using scikit-learn `GradientBoostingRegressor` on precomputed feature matrices. Its prepared format includes a feature ZIP containing NumPy matrices and a label NPY file. Supported feature modes include `distance_only`, `real_charge`, and `full`.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: charge-generation workflows may require PDB2PQR plus RDKit/OpenBabel-compatible ligand processing.

Detailed documentation: [DCML/README.md](DCML/README.md)

### CAPLA

CAPLA is a structure-aware model using global protein features, binding-pocket features, and ligand SMILES. The adapted workflow uses official URV v3b splits, with hyperparameter selection based on validation metrics only. True structural feature regeneration may require DSSP or `mkdssp`.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: the prepared CAPLA dataset may reuse existing global and pocket matrices; regeneration of true structural features depends on external structural-processing assets.

Detailed documentation: [CAPLA/Readme.md](CAPLA/Readme.md)

### DEAttentionDTA

DEAttentionDTA is an attention-based model using encoded protein sequence, ligand SMILES, and pocket representation. The adapted workflow uses official URV v3b splits and preserves checkpoint compatibility with the maintained architecture.

Backend workflow status: Generate / Train / Search / Evaluate.

Important limitations and dependencies: HPO must avoid test leakage; checkpoint loading depends on architecture compatibility and prepared input encodings.

Detailed documentation: [DEAttentionDTA/README.md](DEAttentionDTA/README.md)

## Modules Pending Documentation

The following modules need contributor-owned documentation. The notes below are intentionally conservative.

### CheapNet

Conservative note: model-related files are present, but no detailed module README was used for this overview.

TODO: add architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation usage.
TODO: add expected outputs.
TODO: add limitations and dependencies.

### GIGN

Conservative note: model-related files are present, but no detailed module README was used for this overview.

TODO: add architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation usage.
TODO: add expected outputs.
TODO: add limitations and dependencies.

### GraphDTA

Conservative note: model-related files are present, but no detailed module README was used for this overview.

TODO: add architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation usage.
TODO: add expected outputs.
TODO: add limitations and dependencies.

### PLANET

Conservative note: model-related files are present, but no detailed module README was used for this overview.

TODO: add architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation usage.
TODO: add expected outputs.
TODO: add limitations and dependencies.

### URVDEEPDTAF

Conservative note: a similarly named project folder exists, but no detailed module README was used for this overview.

TODO: add architecture summary.
TODO: add dataset requirements.
TODO: add training/search/evaluation usage.
TODO: add expected outputs.
TODO: add limitations and dependencies.

### Molecule/GNN Utilities

Conservative note: shared graph and molecular utility folders are present, but no detailed module README was used for this overview.

TODO: add utility scope summary.
TODO: add dataset requirements.
TODO: add usage examples.
TODO: add expected outputs.
TODO: add limitations and dependencies.

## Results and Outputs

Common backend outputs include:

- generated feature files;
- graph files;
- checkpoints;
- predictions CSV files;
- metrics CSV, JSON, or YAML files;
- scatter plots;
- training histories;
- HPO trial CSV files;
- `best_config` YAML or JSON files;
- dataset generation reports.

Exact filenames and folder layouts are model-specific.

## Experimental Methodology

- HPO must use validation metrics only.
- Test metrics must not be used for hyperparameter selection.
- Official folds should be used when available.
- Smoke tests are only pipeline checks, not final scientific results.
- Keep output folders tied to the code and configuration used for the run.
- Avoid data leakage between train, validation, and test splits.

## Troubleshooting

| Symptom | Backend check |
|---|---|
| Import errors | Confirm the active conda environment and run the environment checks above. |
| Missing RDKit | Recreate or update the conda environment from `environment.yml`. |
| PyTorch/PyG/`torch_sparse` mismatch | Check PyTorch, CUDA, PyTorch Geometric, and extension wheel compatibility. |
| Missing DSSP or `mkdssp` | Install the structural-processing tool only for workflows that require it. |
| Missing PDB2PQR | Install PDB2PQR only for charge-generation workflows that require it. |
| Missing OpenBabel | Install OpenBabel only for ligand conversion or charge workflows that require it. |
| Missing raw dataset files | Verify dataset root paths and expected raw file layout. |
| CUDA unavailable | Use CPU for small checks or install a compatible CUDA-enabled PyTorch stack. |
| Suspiciously good results | Check for split contamination, target leakage, duplicate samples, and HPO using test metrics. |
| Suspiciously bad results | Check feature/label alignment, split files, checkpoint compatibility, and preprocessing reports. |

## Documentation Policy

`README_BACKEND.md` is the high-level backend overview. Model-specific README files remain the authoritative source for technical details. Undocumented modules should be completed by their responsible contributor. Generated data, checkpoints, and results should normally not be committed unless required by the project protocol.

## Final Status Table

| Module | Documentation status | Backend workflow | Notes |
|---|---|---|---|
| EGNN | Available | Generate / Train / Search / Evaluate | Graph model |
| EDNN | Available | Generate / Train / Search / Evaluate | Edge-aware graph model |
| DeepDTA | Available | Generate / Train / Search / Evaluate | Sequence model |
| WideDTA | Available | Generate / Train / Search / Evaluate | Word/motif sequence model |
| DCML | Available | Generate / Train / Search / Evaluate | Matrix-based model |
| CAPLA | Available | Generate / Train / Search / Evaluate | Structure-aware model |
| DEAttentionDTA | Available | Generate / Train / Search / Evaluate | Attention-based model |
| CheapNet | TODO | Pending documentation | Colleague documentation pending |
| GIGN | TODO | Pending documentation | Colleague documentation pending |
| GraphDTA | TODO | Pending documentation | Colleague documentation pending |
| PLANET | TODO | Pending documentation | Colleague documentation pending |
| URVDEEPDTAF | TODO | Pending documentation | Colleague documentation pending |
