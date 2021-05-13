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

# Data cleaning
columns, values ,sampled_values, locRainfallIndex= process_data("WeatherAUS.csv",True) 

def sendInitialData():
    weatherData = {"columns" : columns.tolist(), "data": values.tolist(),"sampled_data" : sampled_values.tolist(), "loc_rainfall_index":locRainfallIndex}
    return weatherData

@app.route('/getWeatherData')
@cross_origin()
def sendWeatherData():
    return sendInitialData()

app.run()