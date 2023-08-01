from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    notebook_data = r'notebook_credit_card_default\data.csv'
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'
    preprocessor = r'artifacts\preprocessor.pkl'
    model = r'artifacts\model.pkl'
    models = {
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier()
        }
    params = {
        'LogisticRegression': {
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky'],
            'max_iter': [100, 250, 500, 750, 1000]
        },
        'KNeighborsClassifier': {
            'n_neighbors' : [5,9,13,15],
            'weights' : ['uniform','distance'],
            'metric' : ['minkowski','euclidean','manhattan']
        },
        'SVC': {
            # 'C': [0.1, 1, 10, 100, 1000], 
            # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'DecisionTreeClassifier': {
            # 'criterion': ['gini', 'entropy', 'log_loss'],
            # 'splitter': ['best','random'],
            'max_depth': range(5, 15, 3),
            'min_samples_split': range(8, 16, 2),
            'min_samples_leaf': range(5, 15, 3),
            # 'max_features': ['sqrt','log2']
        },
        'RandomForestClassifier': {
                'n_estimators': [25, 50, 75, 100],
                'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': range(5, 15, 3),
            'min_samples_split': range(8, 16, 2),
            'min_samples_leaf': range(5, 15, 3),
            # 'max_features': ['sqrt','log2']
        },
        'GradientBoostingClassifier': {
            # 'n_estimators': [25, 50, 75, 100],
            # 'loss':['log_loss', 'exponential'],
            # 'criterion':['friedman_mse','squared_error'],
            'max_depth': range(5, 15, 3),
            'min_samples_split': range(8, 16, 2),
            'min_samples_leaf': range(5, 15, 3),
            # 'max_features': ['sqrt','log2']
            # 'learning_rate': [1,0.5,.1, .01, .05, .001],
            # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        },
        'AdaBoostClassifier': {
            # 'n_estimators': [25, 50, 75, 100],
            # 'learning_rate': [1,0.5,.1, .01, .05, .001]
        }
}


class InitiateModelTraining:
    def __init__(self, x_train_array, x_test_array, y_train, y_test):
        self.__config = ModelTrainerConfig
        self.x_train_array = x_train_array
        self.x_test_array = x_test_array
        self.y_train = y_train
        self.y_test = y_test
        
    def model_trainer(self):
        try:
            result = {}
            for i in self.__config.models:
                model = self.__config.models[i]
                param = self.__config.params[i]
                gs = GridSearchCV(model, param, scoring='accuracy', n_jobs=-1, verbose=2, cv=5, error_score='raise')
                gs.fit(self.x_train_array, self.y_train)
                model.set_params(**gs.best_params_)
                clf = model.fit(self.x_train_array, self.y_train)
                
                pred_train = clf.predict(self.x_train_array)
                train_score = np.round(accuracy_score(self.y_train, pred_train)*100, 2)
                
                pred_test = clf.predict(self.x_test_array)
                test_score = np.round(accuracy_score(self.y_test, pred_test)*100, 2)
                
                result[str(i)] = [train_score, test_score, gs.best_params_]
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def evaluate_models():
        pass
