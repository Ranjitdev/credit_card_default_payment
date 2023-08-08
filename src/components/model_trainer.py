from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj, load_obj
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    notebook_data = r'notebook\data.csv'
    data = r'artifacts\data.csv'
    train_data = r'artifacts\train.csv'
    test_data = r'artifacts\test.csv'
    preprocessor = r'artifacts\preprocessor.pkl'
    model = r'artifacts\model.pkl'
    scores = r'artifacts\models_scores.json'
    models = {
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        # 'SVC': SVC(),
        # 'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        # 'AdaBoostClassifier': AdaBoostClassifier()
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
            # 'min_samples_split': range(8, 16, 2),
            # 'min_samples_leaf': range(5, 15, 3),
            # 'max_features': ['sqrt','log2']
        },
        'RandomForestClassifier': {
            # 'n_estimators': [25, 50, 75, 100],
            # 'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': range(5, 15, 3),
            'min_samples_split': range(8, 16, 2),
            'min_samples_leaf': range(5, 15, 3),
            'max_features': ['sqrt','log2']
        },
        'GradientBoostingClassifier': {
            # 'n_estimators': [25, 50, 75, 100],
            # 'loss':['log_loss', 'exponential'],
            # 'criterion':['friedman_mse','squared_error'],
            'max_depth': range(5, 15, 3),
            'min_samples_split': range(8, 16, 2),
            'min_samples_leaf': range(5, 15, 3),
            'max_features': ['sqrt','log2'],
            # 'learning_rate': [1,0.5,.1, .01, .05, .001],
            # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        },
        'AdaBoostClassifier': {
            # 'n_estimators': [25, 50, 75, 100],
            # 'learning_rate': [1,0.5,.1, .01, .05, .001]
        }
}


class InitiateModelTraining:
    def __init__(self, x_train_array, x_test_array, y_train, y_test) -> None:
        self.__config = ModelTrainerConfig
        self.x_train_array = x_train_array
        self.x_test_array = x_test_array
        self.y_train = y_train
        self.y_test = y_test
    
    def train_model(self, best_model: str, best_score: float, **best_param: dict) -> None:
        try:
            model = self.__config.models[best_model]
            model.set_params(**best_param)
            clf = model.fit(self.x_train_array, self.y_train)
            save_obj(obj=clf, path=self.__config.model)
            logging.info('Model saved successfully')
            test_model = load_obj(self.__config.model)
            test_model.predict(self.x_test_array)
            logging.info('Model testing done successfully')
        except Exception as e:
            raise CustomException(e, sys)
        
    def evaluate_models(self):
        try:
            start_total_time = datetime.now()
            result = {}
            for i in self.__config.models:
                start_model_time = datetime.now()
                model = self.__config.models[i]
                param = self.__config.params[i]
                gs = GridSearchCV(model, param, scoring='accuracy', n_jobs=8, verbose=2, cv=5, error_score='raise')
                gs.fit(self.x_train_array, self.y_train)
                model.set_params(**gs.best_params_)
                clf = model.fit(self.x_train_array, self.y_train)
                
                pred_train = clf.predict(self.x_train_array)
                train_score = np.round(accuracy_score(self.y_train, pred_train)*100, 2)
                
                pred_test = clf.predict(self.x_test_array)
                test_score = np.round(accuracy_score(self.y_test, pred_test)*100, 2)
                
                result[str(i)] = [train_score, test_score, gs.best_params_]
                end_model_time = datetime.now()
                logging.info(
                    f'Model {str(i)} Train score {train_score} Test score {test_score} Time taken {end_model_time-start_model_time}'
                    )
            end_total_time = datetime.now()
            logging.info(f'Evaluation of models done successfully total time taken {end_total_time-start_total_time}')
            return result
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def evaluate_scores(result: dict) -> Tuple[str, float, str]:
        try:
            scores = pd.DataFrame(
                result, index=['Train score', 'Test score', 'Best parameter']).sort_values(by='Test score', axis=1, ascending=False)
            scores.to_json(r'artifacts\model_scores.json')
            best_score = scores.iloc[:, 0]['Test score']
            best_model = scores.columns[0]
            best_param = scores.iloc[:, 0]['Best parameter']
            logging.info(f'Evaluation of score done Best Model {best_model} with Score {best_score}')
            return str(best_model), float(best_score), best_param
        except Exception as e:
            raise CustomException(e, sys)
