# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:14:31 2020

@author: oseho
"""

import numpy as np

import pandas as pd

from math import sqrt

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from mlens.ensemble import SuperLearner

from sklearn.metrics import r2_score

import sklearn.model_selection as model_selection

import matplotlib.pyplot as plt

from sklearn.ensemble import VotingRegressor

from sklearn import metrics

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import HuberRegressor

import xgboost as xgb

from sklearn.linear_model import Lasso, LassoCV

