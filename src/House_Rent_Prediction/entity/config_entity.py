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





