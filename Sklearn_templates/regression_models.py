import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt


data = pd.read_csv("../UCI_repo/Concrete/Concrete_Data.csv", 
                   sep=',', header=0, index_col=None)

# define X, y
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
# standardize data
stdscal = StandardScaler()
X = stdscal.fit_transform(X)
y = stdscal.fit_transform(y.reshape(-1, 1))
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=123)

#algo = "linear_regression"
#algo = "linear_SVR"
#algo = "SGD_regressor"
#algo = "kernel_ridge"
#algo = "lasso_lars"
#algo = "random_forest"
#algo = "MLP"
#algo = "GradientBoosting"
#algo = "AdaBoost"
algo = "XGBoost"

if algo == "linear_regression":
    # linear regression
    from sklearn import linear_model
    regressor = linear_model.LinearRegression()
    hyper_params = [{
        'fit_intercept': (True,),
    }]
elif algo == "linear_SVR":
    # linear SVR
    from sklearn import svm
    regressor = svm.LinearSVR(max_iter=10000)
    hyper_params = [{
        'C': (1e-06,1e-04,0.1,1,),
        'loss' : ('epsilon_insensitive','squared_epsilon_insensitive',),
    }]
elif algo == "SGD_regressor":
    # regularized loss with SGD
    from sklearn import linear_model
    regressor = linear_model.SGDRegressor()
    hyper_params = [{
        'alpha': (1e-06,1e-04,0.01,1,),
        'penalty': ('l2','l1','elasticnet',),
    }]
elif algo == "kernel_ridge":
    # kernel ridge
    from sklearn import kernel_ridge
    regressor = kernel_ridge.KernelRidge()
    hyper_params = [{
        'kernel': ('linear', 'poly','rbf','sigmoid',),
        'alpha': (1e-4,1e-2,0.1,1,),
        'gamma': (0.01,0.1,1,10,),
    },]
elif algo == "lasso_lars":
    # Lasso LARS
    from sklearn import linear_model
    regressor = linear_model.LassoLars()
    hyper_params = [{
        'alpha': (1e-04,0.001,0.01,0.1,1,),
    }]
elif algo == "random_forest":
    # random forest
    from sklearn import ensemble
    regressor = ensemble.RandomForestRegressor()
    hyper_params = [{
        'n_estimators': (10, 100, 1000),
        'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
        'max_features': ('sqrt','log2',None),
    }]
elif algo == "MLP":
    # multi-layer perceptron
    from sklearn.neural_network import MLPRegressor
    regressor = MLPRegressor()
    hyper_params = [{
        'activation' : ('logistic', 'tanh', 'relu',),
        'solver' : ('lbfgs','adam','sgd',),
        'learning_rate' : ('constant', 'invscaling', 'adaptive',),
    }]
elif algo == "GradientBoosting":
    # gradient boosting
    from sklearn import ensemble
    regressor = ensemble.GradientBoostingRegressor()
    hyper_params = [{
        'n_estimators': (10, 100, 1000,),
        'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
        'max_features': ('sqrt','log2',None,),
    }]
elif algo == "AdaBoost":
    # AdaBoost
    from sklearn import ensemble
    regressor = ensemble.AdaBoostRegressor()
    hyper_params = [{
        'learning_rate' : (0.01, 0.1, 1.0, 10.0,),
        'n_estimators' : (10, 100, 1000,),
    }]
elif algo == "XGBoost":
    # XGBoost
    import xgboost
    regressor = xgboost.XGBRegressor()
    hyper_params = [{
        'n_estimators' : (10, 50, 100, 250, 500, 1000,),
        'learning_rate' : (0.0001,0.01, 0.05, 0.1, 0.2,),
        'gamma' : (0,0.1,0.2,0.3,0.4,),
        'max_depth' : (6,),
        'subsample' : (0.5, 0.75, 1,),
    }]
else:
    sys.exit(0)

# grid search with cross-validation
grid_clf = GridSearchCV(regressor, cv=5, param_grid=hyper_params,
                        verbose=0, n_jobs=4, scoring='r2')

grid_clf.fit(X_train,y_train.ravel())

train_score_mse = mean_squared_error(stdscal.inverse_transform(y_train),
                                     stdscal.inverse_transform(grid_clf.predict(X_train)))

test_score_mse = mean_squared_error(stdscal.inverse_transform(y_test),
                                    stdscal.inverse_transform(grid_clf.predict(X_test)))

sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

# print results
out_text = '\t'.join([algo,
                      str(sorted_grid_params).replace('\n',','), 
                      str(train_score_mse), 
                      str(test_score_mse)])

print(out_text)

# plot
fig, ax = plt.subplots()
y_true = stdscal.inverse_transform(y_test)
y_pred = stdscal.inverse_transform(grid_clf.predict(X_test))
ax.scatter(y_true, y_pred)
ax.set_title('TITLE')
ax.set_xlabel('X label')
ax.set_ylabel('y label')
plt.tight_layout()
plt.show()

