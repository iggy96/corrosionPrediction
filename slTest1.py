# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:17:09 2020

@author: oseho
"""



from defined_libraries import* 
from feature_set import*


ensemble = SuperLearner()
ensemble.add([(AdaBoostRegressor(DecisionTreeRegressor(max_depth= 12), random_state=1)),
              (xgb.XGBRegressor(max_depth=6,random_state=1)),
              (Lasso(alpha=2.0, random_state=1)),
              (SVR(kernel='poly', C=2, degree=3, epsilon=0.0001)),
              (LinearRegression()),
              (Ridge(alpha=0.0006, normalize=True))])
ensemble.add_meta(LinearRegression())
ensemble.fit(X_train, y_train)
print(ensemble.data) # summarize base learners
y_pred = ensemble.predict(X_test) # evaluate meta model

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
mape = (np.mean(percentage_error))