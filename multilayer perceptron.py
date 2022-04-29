# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:16:10 2020

@author: oseho
"""

#Importing the libraries
from defined_libraries import* 
from feature_set import*

# build MLP model
regr = MLPRegressor(hidden_layer_sizes=(10, 10),activation='relu',
                       alpha=0.00001, batch_size='auto', random_state=1).fit(X_train, y_train)

y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)

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

################ visualization ###############################################
l = list(range(8)) #index numbers for x axis
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
plt.title('Linear regression') 
plt.show()

d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('multilayer perceptron predictions.csv')


