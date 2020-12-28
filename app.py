import numpy as np
import pandas as pd
from pycaret.classification import load_model, predict_model
from flask import Flask, request, jsonify, render_template
import os

# app name
app = Flask(__name__)
model = load_model('Datathon 3\loan-prediction\loans')
def get_pred(x):
    labels = ['rejected', 'granted']
    X = pd.DataFrame([x], columns=['Gender', 'Married', 'Dependents','Education','Self_Employed',
                         'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
                         'Credit_History','Property_Area'])
    y = predict_model(model, X)['Label'].values[0]
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