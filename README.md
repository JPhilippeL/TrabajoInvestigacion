# Molecular Analysis System — Backend Documentation

## Authors

**Mohamed Abdessalame** & **Mohamed El-Boukhiari**

## Overview

In this project, we integrated several deep learning models into a Graphical User Interface (GUI) to facilitate the workflow of generating datasets, training models, making predictions, and performing hyperparameter optimization through a unified interface.

The integrated models are:

- GraphDTA (GIN)
- GraphDTA (GCN)
- GraphDTA (GCN-GAT)
- GraphDTA (GAT)
- GIGN
- CheapNet
- DEAttentionDTA
- URVDeepDTAF
- CAPLA
- PLANET
- DCML
- WideDTA
- DeepDTA
- EGNN
- EDNN

Each model provides the following features:

- **Data generation**, which adapts any compatible database to the MPRO structure required by the model.
- **Model training**.
- **Prediction** using trained models.
- **Hyperparameter tuning**, which searches for the best hyperparameter combination to obtain the highest-performing model.

## Installation

```bash
conda env create -n gui_app -f environment.yml
conda activate gui_app
```

## Architecture

Due to differences in the implementation of the integrated models, the backend is divided into two distinct architectures.

### Architecture 1: MVC-Based Architecture

The following models follow a traditional **Model–View–Controller (MVC)** architecture:

- DEAttentionDTA
- CAPLA
- DCML
- DeepDTA
- WideDTA
- EGNN
- EDNN

Each model is organized into its own package with the following structure:

```
MODEL_NAME/
├── core/
├── ui/
└── workers.py
```

For example:

```
CAPLA/
├── core/
├── ui/
└── workers.py
```

### Architecture 2: Design Pattern & Five-Tier Architecture

The remaining models follow a more modular architecture based on several software design patterns combined with a **five-tier architecture**.

The main directories are:

- `core`
- `facade`
- `data_pipeline`
- `model`
- `menu_ui`

The following models follow the architecture:
- GraphDTA (GIN)
- GraphDTA (GCN)
- GraphDTA (GCN-GAT)
- GraphDTA (GAT)
- GIGN
- CheapNet
- Planet