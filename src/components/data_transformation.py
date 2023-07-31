from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import sys


@dataclass
class DataTransformationConfig:
    notebook_data = r'notebook_credit_card_default\data.csv'
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'


class InitiateDataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig
