from House_Rent_Prediction.pipeline.st_1_data_ingestion import DataIngestionPipeline
from House_Rent_Prediction.pipeline.st_2_data_validation import DatavalidationPipeline
from House_Rent_Prediction.pipeline.st_3_data_transformation import DatatransformationPipeline
from House_Rent_Prediction import logger

if __name__ == '__main__':
    try:
        logger.info(f'{"#"*4} Starting Data Ingestion {"#"*4}')
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f'{"#"*4} completed Data Ingestion stage {"#"*4}')
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f'{"#"*4} Starting Data validation {"#"*4}')
        obj = DatavalidationPipeline()
        obj.main()
        logger.info(f'{"#"*4} completed Data validation stage {"#"*4}')
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f'{"#"*4} Starting Data transformation {"#"*4}')
        obj = DatatransformationPipeline()
        obj.main()
        logger.info(f'{"#"*4} completed Data transformation stage {"#"*4}')
    except Exception as e:
        logger.exception(e)
        raise e
