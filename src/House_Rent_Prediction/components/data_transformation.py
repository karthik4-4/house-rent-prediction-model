import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from House_Rent_Prediction import logger
from House_Rent_Prediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config : DataTransformationConfig):
        self.config = config

    
    def preprocess_data(self):
        data = pd.read_csv(self.config.data_path)

        data = data.drop_duplicates()
        logger.info(f'dropped duplicates...')


        for col in ['area','floor']:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            data = data[(data[col] >= lower) & (data[col] <= upper)]
      
        logger.info(f'Removed Outliers...')
        
        logger.info("Removing unwanted columns")
        data = data.drop(columns = ['Homeowners Association', 'property_tax','fire_insurance', 'total_amount'],axis=1)

        data = pd.get_dummies(data,columns=['City', 'pets', 'furnished'], drop_first=True,dtype=int)
        logger.info(f'Encoded categorical variables...')

        train, test = train_test_split(data, test_size = 0.2, random_state=42)
        logger.info(f'Splitted data into train and test...')

        st_scaler = StandardScaler()
        scaled_train = st_scaler.fit_transform(train)
        scaled_test = st_scaler.transform(test)
        train = pd.DataFrame(scaled_train, columns=train.columns)
        test = pd.DataFrame(scaled_test, columns=test.columns)
        logger.info(f'Applied Standard Scaler...')

        train_path = os.path.join(self.config.root_dir,'train.csv')
        test_path = os.path.join(self.config.root_dir,'test.csv')    
        train.to_csv(train_path,index=False)
        test.to_csv(test_path,index=False)   
        logger.info(f'Train set shape : {train.shape}')
        logger.info(f'Test set shape : {test.shape}')
