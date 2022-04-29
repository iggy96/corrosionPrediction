# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:49:06 2020

@author: oseho
"""
from defined_libraries import* 
from feature_set import*
import numpy as np

# build huber model
huber = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001, 
                       warm_start=False, fit_intercept=True,
                       tol=1e-05).fit(X_train, y_train)

# predict with built model
y_pred = huber.predict(X_test)

############################ performance evaluation parameters #####################
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

