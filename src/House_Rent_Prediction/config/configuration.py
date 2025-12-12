from pathlib import Path
from House_Rent_Prediction.utils.common import read_yaml,create_directories
from House_Rent_Prediction.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainingConfig,ModelEvaluationConfig
from House_Rent_Prediction.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH,SCHEMA_FILE_PATH

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH,params_filepath = PARAMS_FILE_PATH, schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir = Path(config.root_dir),
            source_URL = config.source_URL,
            local_data_file = Path(config.local_data_file),
            unzip_dir= Path(config.unzip_dir),
        )
    
    def get_data_validation_config(self)->DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir = Path(config.root_dir),
            source_path = Path(config.source_path),
            status_file = config.status_file,
            data_schema = self.schema,
        )
    
    def get_data_transform_config(self)->DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
        )


    def get_model_training_config(self)->ModelTrainingConfig:
        config = self.config.model_training
        create_directories([config.root_dir])
        params = self.params.Polynomial_regression
        schema = self.schema.TARGET_COLUMN

        return ModelTrainingConfig(
            root_dir = Path(config.root_dir),
            train_data_path = Path(config.train_data_path),
            model_name = config.model_name,
            degree = params.degree,
            target_columns = schema.name,
        )

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        params = self.params.Polynomial_regression
        schema = self.schema.TARGET_COLUMN


        return ModelEvaluationConfig(
            root_dir = Path(config.root_dir),
            test_data_path = Path(config.test_data_path),
            model_path = Path(config.model_path),
            metric_file_name = Path(config.metric_file_name),
            target_column = schema.name,
            all_params = params,
        )
