import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from House_Rent_Prediction.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self,config=ModelTrainingConfig):
        self.config = config

    def train_poly(self):
        train_data = pd.read_csv(self.config.train_data_path)

        X_train = train_data.drop(columns=[self.config.target_columns],axis=1)
        y_train = train_data[[self.config.target_columns]]

        poly = PolynomialFeatures(degree=4)
        poly_feat = poly.fit_transform(X_train)
        lin_reg = LinearRegression()
        lin_reg.fit(poly_feat,y_train)

        joblib.dump(lin_reg,os.path.join(self.config.root_dir,self.config.model_name))

    def train_random_forest(self):
        pass


    def train_xgboost(self):
        pass

    def train_mlp(self):
        pass