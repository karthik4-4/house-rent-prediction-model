import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_squared_error
from House_Rent_Prediction.utils.common import save_json
from House_Rent_Prediction.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config= ModelEvaluationConfig):
        self.config = config

        self.test_set = pd.read_csv(self.config.test_data_path)
        self.X_test = self.test_set.drop(columns=[self.config.target_column])
        self.y_test = self.test_set[self.config.target_column]

        self.model_map = {
            "Polynomial_regressor": {
                "model_path": self.config.poly_reg_path,
                "params": self.config.poly_reg_params,
                "transformer": self.config.polynomial_feat_extr_path
            },
            "Random_Forest": {
                "model_path": self.config.random_forest_path,
                "params": self.config.random_forest_parmas,
                "transformer": None
            },
            "XGBoost": {
                "model_path": self.config.xgboost_path,
                "params": self.config.xg_boost_params,
                "transformer": None
            },
            "MLP_Regressor": {
                "model_path": self.config.mlp_path,
                "params": self.config.mlp_params,
                "transformer": None
            }
        }


    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return rmse, r2

    def predict_model(self, model_path, transformer_path):
        model = joblib.load(model_path)

        if transformer_path:
            transformer = joblib.load(transformer_path)
            X_transformed = transformer.transform(self.X_test)
        else:
            X_transformed = self.X_test

        predictions = model.predict(X_transformed)
        return self.evaluate(self.y_test, predictions)


    def log_model_to_mlflow(self, model_name, model_path, params, rmse, r2):
        mlflow.log_params(params)
        mlflow.log_metric(f"{model_name}_rmse", rmse)
        mlflow.log_metric(f"{model_name}_r2", r2)

        mlflow.sklearn.log_model(
            sk_model=joblib.load(model_path),
            artifact_path=model_name
        )


    def log_into_mlflow(self):
        scores = {}

        with mlflow.start_run():

            for model_name, cfg in self.model_map.items():

                rmse, r2 = self.predict_model(
                    model_path=cfg["model_path"],
                    transformer_path=cfg["transformer"]
                )

                print(f'{model_name} r2_score is {r2}')

                scores[model_name] = {"rmse": rmse, "r2": r2, "path": str(cfg['model_path'])}

                self.log_model_to_mlflow(
                    model_name=model_name,
                    model_path=cfg["model_path"],
                    params=cfg["params"],
                    rmse=rmse,
                    r2=r2
                )

            save_json(path=self.config.metric_file_name, data=scores)
