from House_Rent_Prediction.config.configuration import ConfigurationManager
from House_Rent_Prediction.components.data_transformation import DataTransformation
from House_Rent_Prediction import logger

class DatatransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transform_config()
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.preprocess_data()