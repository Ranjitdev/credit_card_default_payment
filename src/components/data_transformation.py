from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from typing import List
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_obj, load_obj
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataTransformationConfig:
    notebook_data = r'notebook_credit_card_default\data.csv'
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'
    preprocessor = r'artifacts\preprocessor.pkl'



class InitiateDataTransformation:
    def __init__(self):
        self.__config = DataTransformationConfig
        self._target = 'next_month'
        self._categorical_features = ['sex', 'marrige']
        self._numerical_features = ['total_credit', 'bill1_sep', 'paid1_sep', 'bill2_aug', 'paid2_aug', 'bill3_jul', 
                                 'paid3_jul', 'bill4_jun', 'paid4_jun', 'bill5_may', 'paid5_may', 'bill6_apr', 'paid6_apr']
    
    def data_preprocessor(self, data):
        try:
            num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
            cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())
            transformer_obj = ColumnTransformer([
                ('numerical', num_pipe, self._numerical_features),
                ('categorical', cat_pipe, self._categorical_features)
            ], remainder='passthrough')
            
            preprocessor_obj = transformer_obj.fit(data)
            save_obj(obj=preprocessor_obj, path=self.__config.preprocessor)
            
            preprocessor = load_obj(self.__config.preprocessor)
            logging.info('Successfully Created Preprocessor Model')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform_data(self, data=None, train_data=None, test_data=None):
        try:
            if data is None:
                data = pd.read_csv(self.__config.data)
            if train_data is None:
                train_data = pd.read_csv(self.__config.train_data)
            if test_data is None:
                test_data = pd.read_csv(self.__config.test_data)
            
            x_train = train_data.drop(self._target, axis=1)
            x_test = test_data.drop(self._target, axis=1)
            y_train = train_data[self._target]
            y_test = test_data[self._target]
            
            preprocessor = self.data_preprocessor(x_train)
            x_train_array = preprocessor.transform(x_train)
            x_test_array = preprocessor.transform(x_test)
            return x_train_array, x_test_array, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

