import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from House_Rent_Prediction.utils.common import save_json 
from sklearn.metrics import r2_score,mean_squared_error
from House_Rent_Prediction.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config = ModelEvaluationConfig):
        self.config = config

    def evalute(self,y_true,y_pred):
        rmse = np.sqrt(mean_squared_error(y_true,y_pred))
        r2 = r2_score(y_true,y_pred)

        return rmse, r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        polynomial_feat = joblib.load(self.config.polynomial_feat_extr)

        X_test = test_data.drop(columns=[self.config.target_column],axis=1)
        y_test = test_data[[self.config.target_column]]

        
        pred = model.predict(polynomial_feat.transform(X_test))

        (rmse,r2) = self.evalute(y_test,pred)

        scores = {"rmse_score": rmse, "r2_score": r2}
        save_json(path = self.config.metric_file_name,data = scores)