from core.registry.model_registry import MODEL_REGISTRY
from core.strategies.data_generation_strategy import DataGenerationStrategy
from core.strategies.prediction_strategy import PredictStrategy
from core.strategies.train_strategy import TrainStrategy


class JobFactory:
    @staticmethod
    def create_strategy(model_name: str, task_name: str):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_entry = MODEL_REGISTRY[model_name]
        facade = model_entry["facade"]

        if task_name == "generate_data":
            return DataGenerationStrategy(facade)

        if task_name == "train":
            return TrainStrategy(facade)

        if task_name == "predict":
            return PredictStrategy(facade)

        raise ValueError(f"Unknown task: {task_name}")
