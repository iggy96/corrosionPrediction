# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 02:18:26 2020

@author: oseho
"""
#Importing the libraries
from defined_libraries import* 
from feature_set import*

################ Performance Evaluation #######################################
model=(xgb.XGBRegressor(max_depth=7,random_state=1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pred = y_pred
true = y_test

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
plt.title('Boosting XGBM Ensemble Method') 
plt.show()

### Saving result in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('Boosting XGBM Ensemble Method prediction.csv')

