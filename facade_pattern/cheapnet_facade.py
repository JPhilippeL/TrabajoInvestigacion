from data_pipeline.graph_ligand_protein import LPGraph3D
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
        generator = LPGraph3D(config)
        return generator.build_graphs_pt(log_callback=log_callback)
