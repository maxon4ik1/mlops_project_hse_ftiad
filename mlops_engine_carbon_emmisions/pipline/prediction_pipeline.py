import os
import sys

import numpy as np
import pandas as pd
from mlops_engine_carbon_emmisions.entity.config_entity import MlopsProjectPredictorConfig
from mlops_engine_carbon_emmisions.entity.s3_estimator import ModelEstimator
from mlops_engine_carbon_emmisions.exception import MlopsProjectException
from mlops_engine_carbon_emmisions.logger import logging
from mlops_engine_carbon_emmisions.utils.main_utils import read_yaml_file
from pandas import DataFrame


class Data:
    def __init__(self,
                country,
                date,
            	  sector,
                timestamp
                ):
        
        try:
            self.country = country
            self.date = date
            self.sector = sector
            self.timestamp = timestamp


        except Exception as e:
            raise MlopsProjectException(e, sys) from e

    def get_input_data_frame(self)-> DataFrame:
        try:
            
            input_dict = self.get_data_as_dict()
            return DataFrame(input_dict)
        
        except Exception as e:
            raise MlopsProjectException(e, sys) from e


    def get_data_as_dict(self):

        try:
            input_data = {
                "country": [self.country],
                "date": [self.date],
                "sector": [self.sector],
                "timestamp": [self.timestamp]
            }

            logging.info("Created data dict")

            return input_data

        except Exception as e:
            raise MlopsProjectException(e, sys) from e

class Classifier:
    def __init__(self,prediction_pipeline_config: MlopsProjectPredictorConfig = MlopsProjectPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MlopsProjectException(e, sys)


    def predict(self, dataframe) -> str:
        try:
            logging.info("Entered predict method of Classifier class")
            model = ModelEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MlopsProjectException(e, sys)