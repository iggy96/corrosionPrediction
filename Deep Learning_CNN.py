# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:11:33 2020

@author: oseho
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 01:52:40 2020

@author: oseho
"""

from defined_libraries import* 
##################################################libraries for dataset scripts##############
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten



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
X = np.array (X)
X = X.reshape(X.shape[0],X.shape[1],1)
print (X.shape)
# check the type and shape of X
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
model.add(Conv1D(5, 2, activation="relu", input_shape=(5, 1)))
model.add(Flatten())
model.add(Dense(20, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse')
# fit the model
model.fit(X_train, y_train, epochs=900, batch_size=32, verbose=0)
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
plt.title('NN with back propagation but no activation fucntion visualization') 
# show a legend on the plot 
plt.legend() 
# function to show the plot 
plt.show()



