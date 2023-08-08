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
    def __init__(self, user_input: dict) -> None:
        self.user_input = pd.DataFrame([user_input])
        self.config = DataPipeConfig
        
    def main_pipe(self) -> List[float]:
        preprocessor = load_obj(self.config.preprocessor)
        client_data = preprocessor.transform(self.user_input)
        return client_data
    
    def predict_default(self):
        model = load_obj(self.config.model)
        client_data = self.main_pipe()
        prediction = model.predict(client_data)
        if prediction == 0:
            return 'No'
        else:
            return 'Yes'