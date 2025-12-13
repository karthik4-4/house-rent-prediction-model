import numpy as np
import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from House_Rent_Prediction import logger
from House_Rent_Prediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config : DataTransformationConfig):
        self.config = config

    
    def preprocess_data(self):
        data = pd.read_csv(self.config.data_path)

        logger.info(f'handling duplicates')
        data = data.drop_duplicates()

        logger.info("dropping columns")
        cols = ['Homeowners Association', 'property_tax','fire_insurance', 'total_amount']
        data = data.drop(columns = cols)

        logger.info(f'Splitted data into train_set and test-set...')
        X = data.drop(columns = [self.config.target_column])
        y = data[self.config.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y , test_size = 0.2, random_state=42
        )


        def remove_outliers_iqr(train_df, test_df, cols):
            train_df = train_df.copy()
            test_df = test_df.copy()

            for col in cols:
                q1 = train_df[col].quantile(0.25)
                q3 = train_df[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                train_df = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)]
                test_df = test_df[(test_df[col] >= lower) & (test_df[col] <= upper)]

            return train_df, test_df
        
        logger.info(f'handling outliers..')
        X_train, X_test = remove_outliers_iqr(X_train, X_test, ['area','floor'])
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]
        
        X_test.to_csv(
            os.path.join(self.config.root_dir, 'ref.csv'),
            index=False)
                
        logger.info(f'Scaling and encoding...')
        numeric_cols = X_train.select_dtypes(include='number').columns
        categorical_cols = X_train.select_dtypes(include='object').columns
        
        numeric_pipeline = Pipeline(steps=[        
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])


        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        preprocessor_path = Path("artifacts/preprocessor.joblib")

        joblib.dump(preprocessor, os.path.join(self.config.root_dir,'preprocessor.joblib'))
        joblib.dump(X_train,os.path.join(self.config.root_dir, "X_train.joblib"))
        joblib.dump(X_test, os.path.join(self.config.root_dir, "X_test.joblib"))
        joblib.dump(y_train, os.path.join(self.config.root_dir , "y_train.joblib"))
        joblib.dump(y_test, os.path.join(self.config.root_dir, "y_test.joblib"))

        logger.info(f'preprocessing is done..')
        logger.info(f'X_train shape : {X_train.shape}\n X_test shape: {X_test.shape}')
        
        
