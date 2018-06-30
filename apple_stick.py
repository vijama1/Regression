import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
dframe=pd.read_csv('Apple.csv')
training_data=dframe[['Open','High','Low','Adj Close','Volume']]
training_target=dframe[['Close']]

regr = linear_model.LinearRegression()
regr.fit(training_data,training_target)
y_pred = regr.predict([[172.229996,173.919998,171.699997,171.777618,22431600]])
print(y_pred)
print('Coefficients: \n', regr.coef_)
print(mean_squared_error([[172.440002]], y_pred))
