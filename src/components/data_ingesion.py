from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import InitiateDataTransformation
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataIngesionConfig:
    notebook_data = r'notebook_credit_card_default\data.csv'
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'


class InitiateDataIngesion:
    def __init__(self):
        self.__config = DataIngesionConfig
        os.makedirs(os.path.dirname(self.config.data), exist_ok=True)
    
    def get_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.__config.notebook_data)
            data.to_csv(self.__config.data, index=False)
            train_data, test_data = train_test_split(data, random_state=41, train_size=0.75)
            train_data.to_csv(self.__config.train_data, index=False)
            test_data.to_csv(self.__config.test_data, index=False)
            logging.info('Data fetched successfully and saved train and test data')
            return data, train_data, test_data
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    data, train_data, test_data = InitiateDataIngesion().get_data()
    x_train_array, x_test_array, y_train, y_test = InitiateDataTransformation().transform_data(data, train_data, test_data)
