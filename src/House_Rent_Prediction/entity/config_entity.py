import os
from pathlib import Path
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

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    model_name: str
    degree:int
    target_columns: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path 
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    all_params : dict







