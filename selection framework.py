# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 03:40:50 2020

@author: oseho
"""

# Genetic Algorithm applied to develop selection framework for super learner

from defined_libraries import* 
from feature_set import*




def mse():
    return (mean_squared_error(y_test,y_pred))
def R2():
    return abs (r2_score(y_test, y_pred))


# Xgb Hyperparamter
rf_params = {'max_depth': sp_randint(0,8)} # Define the hyperparameter configuration space
n_iter_search=20 #number of iterations is set to 20, you can increase this number if time permits
clf = xgb.XGBRegressor(random_state=1)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
c1 = Random.best_params_
val1 = list(c1.values())[0]
####################
model1 = xgb.XGBRegressor(max_depth =val1,random_state=1)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
xgbMSE = mse()
xgbR2 = R2()


############## Lasso
rf_params = {'alpha': sp_randint(0,3)}
n_iter_search=20
clf = Lasso(random_state=1)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
#####################
c2 = Random.best_params_
val2 = list(c2.values())[0] #value of hyperparameter
####################
model2 = Lasso(alpha =val2,random_state=1)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
lassoMSE = mse()
lassoR2 = R2()



################ AdaBoost 
rf_params = {'learning_rate': sp_randint(1,2)}
n_iter_search=20
clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth= 12), random_state=1)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
#####################
c3 = Random.best_params_
val3 = list(c3.values())[0]
####################
model3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth= 12),
                           learning_rate=val3,random_state=1)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
adaBoostMSE = mse()
adaBoostR2 = R2()



################ Decision Tree 
rf_params = {'max_depth':sp_randint(1,6)}
n_iter_search=20
clf = DecisionTreeRegressor(random_state=1)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
#####################
c4 = Random.best_params_
val4 = list(c4.values())[0]
####################
model4 = DecisionTreeRegressor(max_depth=val4,random_state=1)
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)
dtrMSE = mse()
dtrR2 = R2()



################ light GBM 
rf_params = {'num_leaves':sp_randint(1,50)}
n_iter_search=20
clf = lgb.LGBMRegressor(random_state=1)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
#####################
c5 = Random.best_params_
val5 = list(c5.values())[0]
####################
model5 = lgb.LGBMRegressor(num_leaves=val5,random_state=1)
model5.fit(X_train, y_train)
y_pred = model5.predict(X_test)
lgbmMSE = mse()
lgbmR2 = R2()



################ huber GBM 
rf_params = {'alpha':stats.uniform(0,1)}
n_iter_search=20
clf = HuberRegressor(epsilon=1.35, max_iter=100,warm_start=False,
                     fit_intercept=True,tol=1e-05)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
#####################
c6 = Random.best_params_
val6 = list(c6.values())[0]
####################
model6 = HuberRegressor(epsilon=1.35, max_iter=100, alpha=val6, 
                       warm_start=False, fit_intercept=True,
                       tol=1e-05)
model6.fit(X_train, y_train)
y_pred = model6.predict(X_test)
huberMSE = mse()
huberR2 = R2()



# linear reg
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
linregMSE = mse()
linregR2 = R2()



# Ridge Regression
rf_params = {'alpha':stats.uniform(0,0.0007)}
n_iter_search=20
clf = Ridge(normalize=True, random_state=1)
# Set the hyperparameters of GA    
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,
                            cv=3,random_state =1,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
# transfer hypeparamter value to numVal
#####################
c7 = Random.best_params_
val70 = list(c7.values())[0]
####################
model7 = Ridge(alpha=val70, normalize=True,random_state=1)
model7.fit(X_train, y_train)
y_pred = model7.predict(X_test)
ridgeMSE = mse()
ridgeR2 = R2()



# SVR Linear Kernel
rf_params = {'C': stats.uniform(0,100),"epsilon":stats.uniform(0,1)}
n_iter_search=20
clf = SVR(kernel='linear')
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
c8 = Random.best_params_
val80 = list(c8.values())[0]
val81 = list(c8.values())[1]
####################
model8 = SVR(kernel='linear',C=val80, epsilon=val81)
model8.fit(X_train, y_train)
y_pred = model8.predict(X_test)
svrlinMSE = mse()
svrlinR2 = R2()



# SVR Polynomial Kernel
rf_params = {'C': sp_randint(2,3),
             "epsilon":stats.uniform(0,0.0001)}
n_iter_search=20
clf = SVR(kernel='poly',degree=3)
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='neg_mean_squared_error')
Random.fit(X, y)
print(Random.best_params_)
print("MSE:"+ str(-Random.best_score_))
c9 = Random.best_params_
val90 = list(c9.values())[0]
val91 = list(c9.values())[1]
####################
model9 = SVR(kernel='poly',C=val90,degree=3,epsilon=val91)
model9.fit(X_train, y_train)
y_pred = model9.predict(X_test)
svrpolyMSE = mse()
svrpolyR2 = R2()



# threshold (0.8) for MSE
collMSE = dict([('svrpolyMSE',svrpolyMSE), ('svrlinMSE',svrlinMSE),
                   ('ridgeMSE',ridgeMSE),('linregMSE',linregMSE), ('lassoMSE',lassoMSE),
                   ('huberMSE',huberMSE),('dtrMSE',dtrMSE), ('xgbMSE',xgbMSE), 
                   ('lgbmMSE',lgbmMSE), ('adaBoostMSE',adaBoostMSE)])
results1 = dict((k, v) for k, v in collMSE.items() if v < 0.8)
cd1 = (sorted(results1.items(),key=operator.itemgetter(1),reverse=False))
print (list(iter(results1)))

# threshold 80% or 0.8 for R2
#collR2 = dict([('svrpolyR2',svrpolyR2), ('svrlinR2',svrlinR2),
#                   ('ridgeR2',ridgeR2),('linregR2',linregR2), ('lassoR2',lassoR2),
#                   ('huberR2',huberR2),('dtrR2',dtrR2), ('xgbR2',xgbR2), 
#                   ('lgbmR2',lgbmR2), ('adaBoostR2',adaBoostR2)])

#results2 = dict((k, v) for k, v in collR2.items() if v > 0.8)
#cd2 = (sorted(results2.items(),key=operator.itemgetter(1),reverse=False))
#print (list(iter(results2)))

"""
This framework gives us an idea of the algorithms to use for the super model
The tweaked hyperparameters of the chosen algorithms serve as a baseline
in the super learner model. With this framework, less tweaking is done on
the hyperparameters of the algorithms within the super learner
e.g. slTest2 is based on the qualified algorithms (and their hyperparameters) from
this framework but has an MSE = 0.278; R2 = 0.940
slTest1 is based on the qualified algorithms (and their hyperparameters) from
this framework but their hyperparamters are further tweaked to produce 
MSE = 0.278; R2 = 0.949
"""