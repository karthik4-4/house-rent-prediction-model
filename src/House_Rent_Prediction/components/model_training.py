import numpy as np
import pandas as pd
import os
import joblib

from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from House_Rent_Prediction.entity.config_entity import ModelTrainingConfig
from House_Rent_Prediction import logger


class ModelTraining:
    def __init__(self, config=ModelTrainingConfig):
        self.config = config

        self.X_train = joblib.load(self.config.X_train_path)
        self.X_test = joblib.load(self.config.X_test_path)
        self.y_train = joblib.load(self.config.y_train_path) 
        self.y_test = joblib.load(self.config.y_test_path) 
        
    def train_poly(self):
        logger.info('Starting Polynomial_regrssor')
        print(self.X_train.shape)
        print(self.X_test.shape)
        print(self.y_train.shape)
        poly = PolynomialFeatures(degree=self.config.degree)
        poly_fea = poly.fit_transform(self.X_train)

        lin_reg = SGDRegressor()
        lin_reg.fit(poly_fea, self.y_train)

        y_pred = lin_reg.predict(poly.transform(self.X_test))
        print("r2-score: ",r2_score(self.y_test,y_pred))

        joblib.dump(poly, os.path.join(self.config.root_dir, "polynomial_feat_extr.joblib"))
        joblib.dump(lin_reg, os.path.join(self.config.root_dir, self.config.model_1))

        logger.info('Trained and dumpted poly_reg successfully')


    def train_random_forest(self):
        logger.info('Starting Random_forest_regrssor')
        rf = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=42
        )

        rf.fit(self.X_train, self.y_train)

        y_pred = rf.predict(self.X_test)
        print("r2-score: ",r2_score(self.y_test,y_pred))

        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_2))

        logger.info('Trained and dumpted Random_forest_regrssor successfully')


    def train_xgboost(self):
        logger.info('Starting xgboost')
        xgb = XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_lambda=self.config.xgb_reg_lambda,
            random_state=42
        )

        xgb.fit(self.X_train, self.y_train)

        y_pred = xgb.predict(self.X_test)
        print("r2-score: ",r2_score(self.y_test,y_pred))

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_3))

        logger.info('Trained and dumpted xgboost successfully')


    def train_mlp(self):
        hidden_layers = tuple(self.config.mlp_hidden_layers)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=self.config.mlp_activation,
            solver=self.config.mlp_solver,
            learning_rate_init=self.config.mlp_learning_rate,
            max_iter=self.config.mlp_max_iter,
            random_state=42
        )
        mlp.fit(self.X_train, self.y_train)

        y_pred = mlp.predict(self.X_test)
        print("r2-score: ",r2_score(self.y_test,y_pred))

        joblib.dump(mlp, os.path.join(self.config.root_dir, self.config.model_4))

        logger.info('Trained and mlp successfully')
