from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import sys
import dill
import warnings
warnings.filterwarnings('ignore')


def save_obj(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        dill.dump(obj, file)
        logging.info(f'File {path} saved successfully')


def load_obj(path: str):
    with open(path, 'rb') as file:
        file_obj = dill.load(file)
        logging.info(f'File {path} loaded successfully')
        return file_obj
  