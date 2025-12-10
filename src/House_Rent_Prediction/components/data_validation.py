import numpy as np
import pandas as pd
from House_Rent_Prediction import logger
from House_Rent_Prediction.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self,config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)->bool:
        try:
            validation_status = True

            data = pd.read_csv(self.config.source_path)
            columns = data.columns

            cols_of_schema = self.config.data_schema.COLUMNS.keys()
            
            print(columns)
            print(cols_of_schema)

            for col in columns:
                if col not in cols_of_schema and col!=self.config.data_schema.TARGET_COLUMN.name:
                    print(f'{col} is not matched')
                    validation_status = False
                    break
                else:
                    print(f'{col} is matched')

            with open(self.config.status_file,'w') as f:
                f.write(f'Validation Status : {validation_status}')
                logger.info(f'validation completed')

            return validation_status
        
        except Exception as e:
            raise e
