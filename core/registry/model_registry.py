from facade_pattern.cheapnet_facade import CheapNetFacade

MODEL_REGISTRY = {
    "cheapnet": {
        "display_name": "CheapNet",
        "facade": CheapNetFacade(),
        "tasks": {
            "generate_data": "Generate dataset",
            "train": "Train model",
            "predict": "Predict",
        },
    },
}
