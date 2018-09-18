#importing the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score

#reading the csv files
dframe=pd.read_csv('Summary of Weather.csv')

training_data=dframe[['MinTemp','MeanTemp']]
#training_target on the closing values
training_target=dframe[['MaxTemp']]

#creating the linear regression object
regr = linear_model.LinearRegression()
#fitting the data
regr.fit(training_data,training_target)
max=float(input("Enter the Maximum Temprarure"))
mean=float(input("Enter the Mean Temprature"))

#predicting the values
y_pred = regr.predict([[max,mean]])
print("Predicted value is: ")
print(y_pred)
#printing the regression coefficients
print('Coefficients: \n', regr.coef_)
#printing the mean squarred error
#print('Mean squarred error: \n',mean_squared_error([[43.66]], y_pred))
#print('Variance score: %.2f' % regr.score(43.66, y_pred))
