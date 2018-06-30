#importing the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score

#reading the csv files
dframe=pd.read_csv('Apple.csv')

#training_data on the open,high,low,adjacent close,volume
training_data=dframe[['Open','High','Low','Adj Close','Volume']]
#training_target on the closing values
training_target=dframe[['Close']]

#creating the linear regression object
regr = linear_model.LinearRegression()
#fitting the data
regr.fit(training_data,training_target)
opening=float(input("Enter the opening price: "))
high=float(input("Enter the highest price: "))
low=float(input("Enter the lowest price: "))
adj_close=float(input("Enter the adjacent day closing price: "))
volume=int(input("Enter the volumes of the shares: "))
#predicting the values
y_pred = regr.predict([[opening,high,low,adj_close,volume]])
print("Predicted value is: ")
print(y_pred)
#printing the regression coefficients
print('Coefficients: \n', regr.coef_)
#printing the mean squarred error
print('Mean squarred error: \n',mean_squared_error([[172.440002]], y_pred))
