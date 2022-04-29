# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:42:36 2020

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

#Importing the libraries
from defined_libraries import* 
from feature_set import*

# build poly kernel based model
svr_poly = (SVR(kernel='poly', C=2, degree=3, epsilon=0.0009923))
svr_poly.fit(X_train, y_train)
y_pred = svr_poly.predict(X_test)

################ Performance Evaluation #######################################
def rmse():
	return sqrt(mean_squared_error(y_test, y_pred))
def mse():
    return (mean_squared_error(y_test,y_pred))
def R2():
    return abs (r2_score(y_test, y_pred))

print('RMSE %.3f' % (rmse()))
print('MSE %.3f' % (mse()))
print('R2 %.3f' % (R2()))
error = abs((y_test - y_pred)/y_pred)
percentage_error = (error*100)

################ visualization #####################
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
plt.title('Support Vector Regression: Polynomial Kernel') 
plt.show()
### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('SVR Poly kernel prediction.csv')

##############################################################################################
[i for i in range(0,100000000)]
###################################
#Importing the libraries
from defined_libraries import* 
from feature_set import*

# build poly kernel based model
svr_poly = (SVR(kernel='poly', C=2, degree=3, epsilon=0.0009923))
svr_poly.fit(X_train, y_train)
y_pred = svr_poly.predict(X_test)

################ Performance Evaluation #######################################
def rmse():
	return sqrt(mean_squared_error(y_test, y_pred))
def mse():
    return (mean_squared_error(y_test,y_pred))
def R2():
    return abs (r2_score(y_test, y_pred))

print('RMSE %.3f' % (rmse()))
print('MSE %.3f' % (mse()))
print('R2 %.3f' % (R2()))
error = abs((y_test - y_pred)/y_pred)
percentage_error = (error*100)

################ visualization #####################
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
plt.title('Support Vector Regression: Polynomial Kernel') 
plt.show()
### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('SVR Poly kernel prediction.csv')
##############################################################################################
end_time = end_time_()
print("Execution_time is :", Execution_time(start_time,end_time))
