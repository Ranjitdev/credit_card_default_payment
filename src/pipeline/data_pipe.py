from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import InitiateDataTransformation
from src.components.model_trainer import InitiateModelTraining
from src.utils import save_obj, load_obj
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataPipeConfig:
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'
    preprocessor = r'artifacts\preprocessor.pkl'
    model = r'artifacts\model.pkl'
    scores = r'artifacts\models_scores.json'
    predicted = r'artifacts\predicted.csv'


class DataPipe:
    def __init__(self) -> None:
        self.config = DataPipeConfig
        
    def transformer_pipe(self, user_input) -> List[float]:
        preprocessor = load_obj(self.config.preprocessor)
        client_data = preprocessor.transform(user_input)
        return client_data
    
    def predict_default(self, data: dict) -> str:
        user_input = pd.DataFrame([data])
        model = load_obj(self.config.model)
        client_data = self.transformer_pipe(user_input)
        prediction = model.predict(client_data)
        if prediction == 0:
            return 'No'
        else:
            return 'Yes'
    
    def predict_multiple(self, data: pd.DataFrame) -> List[float]:
        model = load_obj(self.config.model)
        csv_file = data.read().decode('utf-8')
        binary_file = io.StringIO(csv_file)
        df = pd.read_csv(binary_file).drop('next_month', axis=1)
        client_data = self.transformer_pipe(df)
        predictions = model.predict(client_data)
        pred_array = pd.DataFrame(predictions)
        final_file = pd.concat((df, pred_array), axis=1, ignore_index=True)
        final_file.columns = list(df.columns) + ['next_month']
        return final_file