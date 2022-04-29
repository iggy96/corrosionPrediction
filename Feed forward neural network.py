# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 20:17:22 2020

@author: oseho
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as model_selection
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
##################################################libraries for dataset scripts##############
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
#Plot each input against target
# use the list to select a subset of the original DataFrame
feature_cols = ['T', 'DO', 'S','pH','ORP']
X = data[feature_cols]
# print the first 5 rows of X
X.head()
# check the type and shape of X
C = X.shape
print(type(X))
print(X.shape)
# select a Series from the DataFrame
y = data['CR']
# print the first 5 values
y.head()
# check the type and shape of Y
print(type(y))

input_shape = (5,)
#visualization to view correlation between features of dataset
#sns.pairplot(data[["T", "DO", "S", "pH","ORP","CR"]], diag_kind="kde")

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.90,test_size=0.10, random_state=1)
# define model
model = Sequential()
model.add(Dense(5, input_shape=input_shape, activation='linear'))
model.add(Dense(1, activation='linear'))
# compile the model
model.compile(optimizer='adam', loss='mse')
# fit the model
model.fit(X_train, y_train, epochs=3000, batch_size=32, verbose=0)
y_pred = model.predict(X_test)
###########convert y_pred to 1D array
ini_array1 = np.array(y_pred)   
y_pred = ini_array1.flatten() 
print("y_pred: ", y_pred)
#######################################
pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs((true - pred)/pred)
percentage_error = error * 100
score = abs (r2_score(true, pred))
mse = mean_squared_error(true,pred)
mae = mean_absolute_error(true, pred)
print("R2:{0:.3f}, MSE:{1:.2f}, MAE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,mae,np.sqrt(mse)))
################ visualization #####################################
l = list(range(5)) #index numbers for x axis
l
plt.plot(l, y_pred, label = "Predicted values") 
plt.plot(l, y_test, label = "True values") 
plt.plot(l, error, label = "error") 
# naming the x axis 
plt.xlabel('trials') 
# naming the y axis 
plt.ylabel('true and predicted values') 
# giving a title to my graph 
plt.title('Feed Forward Neural Network') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show()

### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('Feed forward neural network prediction.csv')

