from data_pipeline.dta_graph import DTAGraph
from models.graphdta.DTAPredictor import DTAPredictor
from models.graphdta.DTATrainer import DTATrainer


class DTAFacade:
    def train(self, config, log_callback=None, progress_callback=None):
        trainer = DTATrainer(config)
        return trainer.train(log_callback=log_callback)

    def predict(self, config, log_callback=None, progress_callback=None):
        predictor = DTAPredictor(config)
        return predictor.predict(log_callback=log_callback)

    def generate_data(self, config, log_callback=None, progress_callback=None):
        generator = DTAGraph(config)
        return generator.generate_all_dta_dta(log_callback=log_callback)
