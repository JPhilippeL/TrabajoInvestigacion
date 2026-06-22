from data_pipeline.planet_graph import PlanetGraph
from models.planet.PlanetPredictor import PlanetPredictor
from models.planet.PlanetTrainer import PlanetTrainer


class PlanetFacade:
    def train(self, config, log_callback=None, progress_callback=None):
        trainer = PlanetTrainer(config)
        return trainer.train(log_callback=log_callback)

    def predict(self, config, log_callback=None, progress_callback=None):
        predictor = PlanetPredictor(config)
        return predictor.predict(log_callback=log_callback)

    def generate_data(self, config, log_callback=None, progress_callback=None):
        generator = PlanetGraph(config)
        return generator.build_data(log_callback=log_callback)
