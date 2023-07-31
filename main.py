from src.components.data_ingesion import InitiateDataIngesion
from flask import Flask
import requests


InitiateDataIngesion().get_data()

app = Flask(__name__)

@app.route('/homepage', methods=['POST', 'GET'])
def homepage():
    return 'Homepage'


if __name__=='__main__':
    app.run(debug=True, port=8000)