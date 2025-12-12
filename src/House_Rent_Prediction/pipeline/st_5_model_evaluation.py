from House_Rent_Prediction.config.configuration import ConfigurationManager
from House_Rent_Prediction.components.model_evaluation import ModelEvaluation
from House_Rent_Prediction import logger

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluation(model_eval_config)
        model_eval.log_into_mlflow()