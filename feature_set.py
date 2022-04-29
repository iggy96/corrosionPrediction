# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:26:02 2020

@author: oseho
"""




from defined_libraries import* 

data = pd.read_csv('training data_including_test_data_corrosion_rate_confirmation.csv')

data.head() # check the first five values of the full dataset

data.tail() # check the last five values of the full dataset

data.shape  # check the shape of the full dataset

data_describe = data.describe() # statistical description, only for numeric values

feature_cols = ['T', 'DO', 'S','pH','ORP']

X = data[feature_cols]

X.head()    # print the first 5 rows of X

print(type(X))  # check the data type of X

print(X.shape)  # check the shape of X

y = data['CR']  # select a Series from the DataFrame

y.head()        # print the first 5 values

print(type(y))  # check the data type of y

print(y.shape)  # check the shape of y

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.90,test_size=0.10, random_state = 1)

