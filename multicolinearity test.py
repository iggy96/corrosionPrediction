from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sklearn.model_selection as model_selection
from sklearn import metrics




data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')
data.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
X = data.iloc[:,:-1]
k = calc_vif(X)
print (k)
###########################33
data2 = data.copy()
data2['T_pH'] = data.apply(lambda x: x['T'] - x['pH'],axis=1)
X1 = data2.drop(['T','pH','CR'],axis=1)
k1 = calc_vif(X1)
print (k1)
print (X1)
#################
