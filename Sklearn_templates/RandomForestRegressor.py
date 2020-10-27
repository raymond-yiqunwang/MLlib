import numpy as np
import pandas as pd
import sys
import operator

from pmlb import regression_dataset_names, fetch_data

from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#dataset = sys.argv[1]

#input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

dataset = []

cnt = 0
for reg_data in regression_dataset_names:
    X, y = fetch_data(reg_data, return_X_y=True, local_cache_dir='../dataset')
    if (X.shape[0] > 100 or X.shape[1] > 10): continue
    dataset.append(reg_data)
    cnt += 1

print('There are in total %d datasets' %cnt)

hyper_params = [{
    'n_estimators': (10, 100, 1000),
    'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
    'max_features': ('sqrt','log2',None),
}]

for data_name in dataset:
    X, y = fetch_data(data_name, return_X_y=True)
    stdscal = StandardScaler()
    X = stdscal.fit_transform(X)
    y = stdscal.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=5)

    regressor = ensemble.RandomForestRegressor()

    grid_clf = GridSearchCV(regressor,cv=5,param_grid=hyper_params,
                            verbose=0,n_jobs=1,scoring='r2')

    grid_clf.fit(X_train,y_train.ravel())

    train_score_mse = mean_squared_error(stdscal.inverse_transform(y_train),stdscal.inverse_transform(grid_clf.predict(X_train)))
    test_score_mse = mean_squared_error(stdscal.inverse_transform(y_test),stdscal.inverse_transform(grid_clf.predict(X_test)))

    sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

    # print results
    out_text = '\t'.join([data_name.split('/')[-1][:-7], 'random forest',
                          str(sorted_grid_params).replace('\n',','), str(train_score_mse), str(test_score_mse)])

    print(out_text)
    sys.stdout.flush()
