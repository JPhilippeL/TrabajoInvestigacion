# DCML GUI Module

This folder contains the clean DCML integration for the Molecular Analysis System GUI.

DCML in this integration is not a PyTorch Geometric graph model. It trains a scikit-learn `GradientBoostingRegressor` on precomputed DCML feature matrices:

- `feature.zip`: ZIP file containing exactly one `.npy` 2D matrix.
- `label.npy`: one-dimensional NumPy array containing the target values.

## Main files

```text
DCML/
├── Core/
│   ├── common.py
│   ├── data_utils.py
│   ├── metrics_utils.py
│   ├── dcml_trainer.py
│   ├── dcml_tester.py
│   ├── dcml_hyperparameter_search.py
│   ├── dcml_feature_generation.py
│   └── dcml_results.py
├── ui/
│   ├── dialogs/
│   │   ├── train_dcml_dialog.py
│   │   ├── test_dcml_dialog.py
│   │   └── hyperparameter_search_dcml_dialog.py
│   └── menus/
│       └── menu_DCML.py
├── configs/
│   ├── default_dcml.yaml
│   └── search_space_dcml.yaml
└── workers.py
```

## GUI integration

In `ui/menu_bar.py`, add:

```python
from DCML.ui.menus.menu_DCML import MenuDCML
```

Then inside `MenuBar.__init__`, after the other model menus:

```python
self.menu_DCML = MenuDCML(self.parent)
self.addMenu(self.menu_DCML)
```

## Core usage

Train:

```python
from DCML.Core.dcml_trainer import train_dcml

summary = train_dcml(
    train_feature_zip="data/train_feature.zip",
    train_label_npy="data/train_label.npy",
    output_model="DCML/results/DCML.pt",
    output_dir="DCML/results/train",
    cast_float32=True,
    seed=42,
    hyperparameters={
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.01,
        "min_samples_split": 2,
        "subsample": 0.7,
        "max_features": "sqrt",
        "loss": "squared_error",
    },
)
```

Evaluate:

```python
from DCML.Core.dcml_tester import test_dcml

summary = test_dcml(
    model_pt="DCML/results/DCML.pt",
    feature_zip="data/test_feature.zip",
    label_npy="data/test_label.npy",
    output_dir="DCML/results/predict",
    dataset_name="test",
    cast_float32=True,
)
```

Hyperparameter search:

```python
from DCML.Core.dcml_hyperparameter_search import run_hyperparameter_search

results = run_hyperparameter_search(
    train_feature_zip="data/train_feature.zip",
    train_label_npy="data/train_label.npy",
    validation_feature_zip="data/test_feature.zip",
    validation_label_npy="data/test_label.npy",
    models_root="DCML/results/hpo_models",
    results_root="DCML/results/hpo",
    n_estimators_values=[100, 300],
    max_depth_values=[3, 6],
    learning_rate_values=[0.01, 0.05],
    subsample_values=[0.7, 1.0],
    max_features_values=["sqrt", None],
    loss_values=["squared_error"],
)
```

## CPU note

The `device` parameter is accepted for GUI consistency, but DCML runs on CPU because the backend is scikit-learn.
