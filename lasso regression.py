# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 03:34:16 2020

@author: oseho
"""


import time
from time import strftime
from datetime import datetime 
from time import gmtime

def start_time_():    
    #import time
    start_time = time.time()
    return(start_time)

def end_time_():
    #import time
    end_time = time.time()
    return(end_time)

def Execution_time(start_time_,end_time_):
   #import time
   #from time import strftime
   #from datetime import datetime 
   #from time import gmtime
   return(strftime("%H:%M:%S",gmtime(int('{:.0f}'.format(float(str((end_time-start_time))))))))

start_time = start_time_()
###############################################################################

from defined_libraries import* 
from feature_set import*

data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
#Plot each input against target
#sns.pairplot(data, x_vars=['T', 'DO', 'S','pH','ORP'], y_vars='CR', size=7, aspect=0.7, kind='reg')
# use the list to select a subset of the original DataFrame
feature_cols = ['T', 'DO', 'S','pH','ORP']
X = data[feature_cols]
# print the first 5 rows of X
X.head()
# check the type and shape of X
print(type(X))
print(X.shape)
# select a Series from the DataFrame
y = data['CR']
# print the first 5 values
y.head()
# check the type and shape of Y
print(type(y))
print(y.shape)
#####################

model=(Lasso(alpha=2.0, random_state=1)).fit(X_train, y_train)
y_pred = model.predict(X_test)
print("y_pred:",y_pred)

pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs((true - pred)/pred)
percentage_error = error * 100
score = model.score(X_test,true)
mse = mean_squared_error(true,pred)
mae = metrics.mean_absolute_error(true, pred)
print("Alpha:{0:.2f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(model.alpha, score, mse, np.sqrt(mse)))
################ visualization #####################################
l = list(range(5)) #index numbers for x axis
l
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(l, y_pred, color =  "g", label = "Predicted values")
lns2 = ax.plot(l, y_test,color = "r", label = "True values")
ax2 = ax.twinx()
lns3 = ax2.plot(l, percentage_error, label = '% error')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("trials")
ax.set_ylabel(r"true and predicted values ($μA/cm^2$)")
ax2.set_ylabel(r"% error")
plt.title('Lasso regression') 
plt.show()
### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('Lasso regression prediction.csv')

##############################################################################################
[i for i in range(0,100000000)]
############################################################################## your code here #
from defined_libraries import* 
from feature_set import*

data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
data.tail()
data.shape
#Plot each input against target
#sns.pairplot(data, x_vars=['T', 'DO', 'S','pH','ORP'], y_vars='CR', size=7, aspect=0.7, kind='reg')
# use the list to select a subset of the original DataFrame
feature_cols = ['T', 'DO', 'S','pH','ORP']
X = data[feature_cols]
# print the first 5 rows of X
X.head()
# check the type and shape of X
print(type(X))
print(X.shape)
# select a Series from the DataFrame
y = data['CR']
# print the first 5 values
y.head()
# check the type and shape of Y
print(type(y))
print(y.shape)
#####################

model=(Lasso(alpha=2.0, random_state=1)).fit(X_train, y_train)
y_pred = model.predict(X_test)
print("y_pred:",y_pred)

pred = y_pred
true = y_test
################ Performance Evaluation #######################################
error = abs((true - pred)/pred)
percentage_error = error * 100
score = model.score(X_test,true)
mse = mean_squared_error(true,pred)
mae = metrics.mean_absolute_error(true, pred)
print("Alpha:{0:.2f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(model.alpha, score, mse, np.sqrt(mse)))
################ visualization #####################################
l = list(range(5)) #index numbers for x axis
l
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(l, y_pred, color =  "g", label = "Predicted values")
lns2 = ax.plot(l, y_test,color = "r", label = "True values")
ax2 = ax.twinx()
lns3 = ax2.plot(l, percentage_error, label = '% error')

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("trials")
ax.set_ylabel(r"true and predicted values ($μA/cm^2$)")
ax2.set_ylabel(r"% error")
plt.title('Lasso regression') 
plt.show()
### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('Lasso regression prediction.csv')

##############################################################################################
end_time = end_time_()
print("Execution_time is :", Execution_time(start_time,end_time))

from guppy import hpy; h=hpy()
h.heap()