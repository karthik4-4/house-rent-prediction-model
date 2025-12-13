import os
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_URL : str
    local_data_file : Path
    unzip_dir : Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir : Path
    source_path : Path
    status_file : Path
    data_schema : dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    data_path : Path
    target_column: str

@dataclass(frozen=True)
class ModelTrainingConfig:
    X_train_path: Path
    X_test_path: Path
    y_train_path: Path 
    y_test_path: Path
    root_dir: str

    # Polynomial
    model_1: str
    degree: int

    # Random Forest
    model_2: str
    rf_n_estimators: Optional[int]
    rf_max_depth: int
    rf_min_samples_split: int
    rf_min_samples_leaf: int

    # XGBoost
    model_3: str
    xgb_n_estimators: int
    xgb_learning_rate: float
    xgb_max_depth: int
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_reg_lambda: float

    # MLP
    model_4: str
    mlp_hidden_layers: List[int]
    mlp_activation: str
    mlp_solver: str
    mlp_learning_rate: float
    mlp_max_iter: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    poly_reg_path: Path
    polynomial_feat_extr_path: Path
    random_forest_path: Path
    xgboost_path: Path
    mlp_path: Path
    metric_file_name: Path
    poly_reg_params: dict
    random_forest_parmas: dict
    xg_boost_params: dict
    mlp_params: dict
    target_column: str








