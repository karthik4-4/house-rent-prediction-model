import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from House_Rent_Prediction import logger
from House_Rent_Prediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config : DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data, test_size = 0.2, random_state=42)

        train_path = os.path.join(self.config.root_dir,'train.csv')
        test_path = os.path.join(self.config.root_dir,'test.csv')

        train.to_csv(train_path,index=False)
        test.to_csv(test_path,index=False)

        logger.info(f"Splitted data into train and test")
        logger.info(f'Train set shape : {train.shape}')
        logger.info(f'Test set shape : {test.shape}')

        return train_path, test_path
    
    



