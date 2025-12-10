from House_Rent_Prediction.config.configuration import ConfigurationManager
from House_Rent_Prediction.components.data_validation import DataValidation
from House_Rent_Prediction import logger

class DatavalidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        data_validation.validate_all_columns()