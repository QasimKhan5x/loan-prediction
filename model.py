import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')
rf = load_model('Datathon 3\loan-prediction\loans')
predict_new = predict_model(rf, data=test_data)
print(predict_new.head())

