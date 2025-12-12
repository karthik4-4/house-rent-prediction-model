from House_Rent_Prediction.config.configuration import ConfigurationManager
from House_Rent_Prediction.components.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(model_training_config)
        model_training.train_poly()
        model_training.train_random_forest()
        model_training.train_xgboost()
        model_training.train_mlp()