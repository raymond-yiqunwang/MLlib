import numpy as np
import pandas as pd
import sys
import operator
import math

import xgboost
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#data = pd.read_csv("../../data/data_all_float.csv", header=0, index_col=None, sep=';')

# drop 'CTE, linear' for more instances to study
drop_feature = ['Density', 'CTE, linear']
for ifeature in drop_feature:
    data = data.drop(labels=ifeature, axis=1)


stdscale = StandardScaler()

elements = ['Iron, Fe', 'Carbon, C', 'Sulfur, S', 'Silicon, Si', 'Phosphorous, P', 'Manganese, Mn', 'Chromium, Cr', 'Nickel, Ni', 'Molybdenum, Mo', 'Copper, Cu']
    
target = 'Thermal Conductivity'
# drop instances with NaN
drop_instance = []
for idx in data.index:
    if math.isnan(data.loc[idx, target]):
        drop_instance.append(idx)
data_loc = data.drop(drop_instance)
X = data_loc[elements].values
y = data_loc[target].values

print('Shape of dataset:')
print(X.shape)

# standardize features
X = stdscale.fit_transform(X)
y = stdscale.fit_transform(y.reshape(-1, 1))

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

train_sizes, train_scores, valid_scores = learning_curve(xgboost.XGBRegressor(), X, y, train_sizes=list(range(20, 490, 10)), cv=5)
out = []
for ii in range(len(train_sizes)):
    out.append([train_sizes[ii], np.mean(train_scores[ii]), np.mean(valid_scores[ii])])

dat = pd.DataFrame(out)
dat.to_csv("./tmp.dat", sep=' ', header=False, index=False)
