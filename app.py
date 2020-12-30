import numpy as np
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify, render_template
import os

# app name
app = Flask(__name__)
# model
model = load('pipe_sklearn.joblib') 

def get_pred(x):
    labels = ['rejected', 'granted']
    X = np.array([x])
    y = model.predict(X)[0]
    return labels[y]
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = get_pred([x for x in request.form.values()])
    return render_template('index.html', output=f"Your application is {prediction}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port, use_reloader=False)
