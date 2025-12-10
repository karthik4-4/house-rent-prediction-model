from pathlib import Path
from House_Rent_Prediction.utils.common import read_yaml,create_directories
from House_Rent_Prediction.entity.config_entity import DataIngestionConfig,DataValidationConfig
from House_Rent_Prediction.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH,SCHEMA_FILE_PATH

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH,schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir = Path(config.root_dir),
            source_URL = config.source_URL,
            local_data_file = Path(config.local_data_file),
            unzip_dir= Path(config.unzip_dir)
        )
    
    def get_data_validation_config(self)->DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir = Path(config.root_dir),
            source_path = Path(config.source_path),
            status_file = config.status_file,
            data_schema = self.schema
        )

