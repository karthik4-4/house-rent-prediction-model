from House_Rent_Prediction.config.configuration import ConfigurationManager
from House_Rent_Prediction.components.data_ingestion import DataIngestion
from House_Rent_Prediction import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()       
