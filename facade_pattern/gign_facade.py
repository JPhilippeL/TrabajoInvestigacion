from data_pipeline.graph_ligand_protein import LPGraph3D
from models.gign.GIGNPredictor import GIGNPredictor
from models.gign.GIGNTrainer import GIGNTrainer


class GIGNFacade:
    def train(self, config, log_callback=None, progress_callback=None):
        trainer = GIGNTrainer(config)
        return trainer.train(log_callback=log_callback)

    def predict(self, config, log_callback=None, progress_callback=None):
        predictor = GIGNPredictor(config)
        return predictor.predict(log_callback=log_callback)

    def generate_data(self, config, log_callback=None, progress_callback=None):
        generator = LPGraph3D(config)
        return generator.build_graphs_pt(log_callback=log_callback)
