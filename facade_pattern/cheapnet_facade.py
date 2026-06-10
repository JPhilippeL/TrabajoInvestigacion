from models.cheapnet.CheapnetDataGeneration import CheapnetDataGeneration
from models.cheapnet.CheapnetPredictor import CheapNetPredictor
from models.cheapnet.CheapNetTrainer import CheapNetTrainer


class CheapNetFacade:
    def train(self, config, log_callback=None, progress_callback=None):
        trainer = CheapNetTrainer(config)
        return trainer.train(log_callback=log_callback)

    def predict(self, config, log_callback=None, progress_callback=None):
        predictor = CheapNetPredictor(config)
        return predictor.predict(log_callback=log_callback)

    def generate_data(self, config, log_callback=None, progress_callback=None):
        generator = CheapnetDataGeneration(config)
        return generator.build_graphs_pt(log_callback=log_callback)
