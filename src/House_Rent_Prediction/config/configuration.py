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
        schema = self.schema.TARGET_COLUMN

        return DataTransformationConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            target_column = schema.name,
        )


    def get_model_training_config(self)->ModelTrainingConfig:
        config = self.config.model_training
        create_directories([config.root_dir])
        poly_params = self.params.Polynomial_regression
        random_params = self.params.Random_Forest
        xgboost_params = self.params.XGBoost
        mlp_params = self.params.MLP


        return ModelTrainingConfig(
            X_train_path = Path(config.X_train_path),
            X_test_path = Path(config.X_test_path),
            y_train_path = Path(config.y_train_path), 
            y_test_path = Path(config.y_test_path),
            root_dir = Path(config.root_dir),

            model_1 = config.model_1,       # Polynomial
            degree = poly_params.degree,

            
            model_2 = config.model_2,        # Random Forest
            rf_n_estimators = random_params.rf_n_estimators,
            rf_max_depth = random_params.rf_max_depth,
            rf_min_samples_split = random_params.rf_min_samples_split,
            rf_min_samples_leaf = random_params.rf_min_samples_leaf,


            model_3 = config.model_3,         # XGBoost
            xgb_n_estimators = xgboost_params.xgb_n_estimators,
            xgb_learning_rate = xgboost_params.xgb_learning_rate,
            xgb_max_depth = xgboost_params.xgb_max_depth,
            xgb_subsample = xgboost_params.xgb_subsample,
            xgb_colsample_bytree = xgboost_params.xgb_colsample_bytree,
            xgb_reg_lambda = xgboost_params.xgb_reg_lambda,

            model_4 = config.model_4,         # MLP
            mlp_hidden_layers = mlp_params.mlp_hidden_layers,
            mlp_activation = mlp_params.mlp_activation,
            mlp_solver = mlp_params.mlp_solver,
            mlp_learning_rate = mlp_params.mlp_learning_rate,
            mlp_max_iter = mlp_params.mlp_max_iter,

        )

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        poly_params = self.params.Polynomial_regression
        random_params = self.params.Random_Forest
        xgboost_params = self.params.XGBoost
        mlp_params = self.params.MLP


        return ModelEvaluationConfig(
            root_dir = Path(config.root_dir),
            X_test_path = Path(config.X_test_path),
            y_test_path = Path(config.y_test_path),
            poly_reg_path = Path(config.poly_reg_path),
            polynomial_feat_extr_path = Path(config.polynomial_feat_extr_path),
            random_forest_path = Path(config.random_forest_path),
            xgboost_path = Path(config.xgboost_path),
            mlp_path = Path(config.mlp_path),
            metric_file_name = Path(config.metric_file_name),
            poly_reg_params = poly_params,
            random_forest_parmas = random_params,
            xg_boost_params = xgboost_params,
            mlp_params = mlp_params,
        )
