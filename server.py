from flask import Flask
from flask import request, jsonify
import pandas as pd
import numpy as np
import json
from json import JSONEncoder

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

def sendInitialData():
    weatherData = {"data": "Hiii I am weather"}
    return weatherData

@app.route('/getWeatherData')
@cross_origin()
def sendWeatherData():
    return sendInitialData()

app.run()