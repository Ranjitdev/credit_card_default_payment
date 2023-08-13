from src.components.data_ingesion import InitiateDataIngesion
from src.components.data_transformation import InitiateDataTransformation
from src.components.model_trainer import InitiateModelTraining
from src.pipeline.data_pipe import DataPipe
from src.exception import CustomException
from src.logger import logging
from flask import Flask, flash, request, redirect, url_for, render_template
import pandas as pd
import sys


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            total_credit = request.form.get('total_credit')
            sex = request.form.get('sex')
            education = request.form.get('education')
            marrige = request.form.get('marrige')
            age = request.form.get('age')
            pay1_sep = request.form.get('pay1_sep')
            bill1_sep = request.form.get('bill1_sep')
            paid1_sep = request.form.get('paid1_sep')
            pay2_aug = request.form.get('pay2_aug')
            bill2_aug = request.form.get('bill2_aug')
            paid2_aug = request.form.get('paid2_aug')
            pay3_jul = request.form.get('pay3_jul')
            bill3_jul = request.form.get('bill3_jul')
            paid3_jul = request.form.get('paid2_aug')
            pay4_jun = request.form.get('pay4_jun')
            bill4_jun = request.form.get('bill4_jun')
            paid4_jun = request.form.get('paid4_jun')
            pay5_may = request.form.get('pay5_may')
            bill5_may = request.form.get('bill5_may')
            paid5_may = request.form.get('paid5_may')
            pay6_apr = request.form.get('pay6_apr')
            bill6_apr = request.form.get('bill6_apr')
            paid6_apr = request.form.get('paid6_apr')
            
            user_input = {
                'total_credit': total_credit,
                'sex': sex,
                'education': education,
                'marrige': marrige,
                'age': age,
                'pay1_sep': pay1_sep,
                'bill1_sep': bill1_sep,
                'paid1_sep': paid1_sep,
                'pay2_aug': pay2_aug,
                'bill2_aug': bill2_aug,
                'paid2_aug': paid2_aug,
                'pay3_jul': pay3_jul,
                'bill3_jul': bill3_jul,
                'paid3_jul': paid3_jul,
                'pay4_jun': pay4_jun,
                'bill4_jun': bill4_jun,
                'paid4_jun': paid4_jun,
                'pay5_may': pay5_may,
                'bill5_may': bill5_may,
                'paid5_may': paid5_may,
                'pay6_apr': pay6_apr,
                'bill6_apr': bill6_apr,
                'paid6_apr': paid6_apr
            }
            
            
            prediction = DataPipe().predict_default(data=user_input)
            logging.info('Got the data from web and sented predicted result to web')
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            raise CustomException(e, sys)
    else:
        return str(request.method) + ' is wrong method'

@app.route('/multi_entry',  methods=['POST', 'GET'])
def multi_entry():
    try:
        csv_file = request.files.get('file')
        output_data = DataPipe().predict_multiple(data=csv_file)
        output_data = output_data.to_html(classes='table table-bordered table-hover', index=False)
        return render_template('excel_data.html', raw_data=output_data)
    except Exception as e:
            raise CustomException(e, sys)

@app.route('/excel_data',  methods=['POST', 'GET'])
def raw_data():
    try:
        data = pd.read_csv('artifacts/data.csv').drop('next_month', axis=1)
        data_html = data.to_html(classes='table table-bordered table-hover', index=False)
        return render_template('excel_data.html', raw_data=data_html)
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/retrain_model',  methods=['POST', 'GET'])
def retrain_model():
    try:
        data, train_data, test_data = InitiateDataIngesion().get_data()
        x_train_array, x_test_array, y_train, y_test = InitiateDataTransformation().transform_data(data, train_data, test_data)
        result = InitiateModelTraining(x_train_array, x_test_array, y_train, y_test).evaluate_models()
        best_model, best_score, best_param = InitiateModelTraining.evaluate_scores(result)
        InitiateModelTraining(x_train_array, x_test_array, y_train, y_test).train_model(best_model, best_score, **best_param)
        return render_template('index.html', training_done='Model Trained Successfully')
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/scores', methods=['POST', 'GET'])
def scores():
    try:
        data = pd.read_json('artifacts/model_scores.json')
        scores = data.to_html(index=True)
        return render_template('scores.html', scores=scores)
    except Exception as e:
        raise CustomException(e, sys)


if __name__=='__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')