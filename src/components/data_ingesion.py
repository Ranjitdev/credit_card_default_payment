from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import sys


@dataclass
class DataIngesionConfig:
    notebook_data = r'notebook_credit_card_default\data.csv'
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'


class InitiateDataIngesion:
    def __init__(self):
        self.config = DataIngesionConfig
        os.makedirs(os.path.dirname(self.config.data), exist_ok=True)
    
    def get_data(self):
        try:
            data = pd.read_csv(self.config.notebook_data)
            data.to_csv(self.config.data)
            train_data, test_data = train_test_split(data, random_state=42, train_size=0.75)
            train_data.to_csv(self.config.train_data)
            test_data.to_csv(self.config.test_data)
            logging.info('Data fetched successfully and saved train and test data')
            return data, train_data, test_data
        except Exception as e:
            raise CustomException(e, sys)
