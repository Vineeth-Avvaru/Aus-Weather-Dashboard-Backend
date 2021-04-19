from flask import Flask
from flask import request, jsonify
import pandas as pd
import numpy as np
import json
from json import JSONEncoder
from data_cleaning import process_data

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# To make np.ndarray JSON Serializable
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Data cleaning
columns, values = process_data("WeatherAUS.csv") 

def sendInitialData():
    weatherData = {"columns" : columns.tolist(), "data": values.tolist()}
    return weatherData

@app.route('/getWeatherData')
@cross_origin()
def sendWeatherData():
    return sendInitialData()

app.run()