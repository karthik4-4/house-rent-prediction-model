import os
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,  # minimum level to show
    format="%(asctime)s - %(levelname)s - %(message)s"
)

project_name = "House-Rent-Prediction-model"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir = filepath.parent
    filename = filepath.name

    if filedir != Path(''):
        if not filedir.exists():
            os.makedirs(filedir,exist_ok=True)
            logging.info(f'Created directory: {filedir} for file: {filename}')
        else:
            logging.info(f'Directory already exists: {filedir}')

    if not filepath.exists():
        with open(filepath,'w') as f:
            pass
        logging.info(f'Creating file: {filepath}')
    
    elif os.path.getsize(filepath)==0:
        logging.info(f'File already exists but it is empty: {filepath}')
    
    else:
        logging.info(f'File already exists and not empty: {filepath}')

