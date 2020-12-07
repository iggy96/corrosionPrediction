
from defined_libraries import* 
from feature_set import*


ensemble = SuperLearner()
ensemble.add([(xgb.XGBRegressor(max_depth=7,random_state=1)),
              (Lasso(alpha=2.0, random_state=1)),
              (SVR(kernel='poly', C=2, degree=3, epsilon=0.0009923)),
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

l = list(range(5)) #index numbers for x axis
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
plt.title('Super Learner') 
plt.show()

import seaborn as sns
sns.regplot(y_pred, y_test)
# save results in csv file
d = {'y_test':y_test, 'y_pred':y_pred,'error':error,'percentage error':percentage_error}
prediction = pd.DataFrame(d, columns=None).to_csv('superLearnerThesis.csv')