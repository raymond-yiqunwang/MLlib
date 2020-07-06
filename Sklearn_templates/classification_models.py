import numpy as np
import pandas as pd
import operator
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv("../UCI_repo/Abalone/abalone.csv",
                   sep=',', header=0, index_col=None)

# print statistics
for icol in data.columns:
    print(icol, data[icol].dtypes)
    if data[icol].dtypes == 'object':
        print(' ', data[icol].unique())
    else:
        print('  min: {:.2f}, max: {:.2f}, mean: {:.2f}, median: {:.2f}'.format( \
                data[icol].min(), data[icol].max(), data[icol].mean(), data[icol].median()))

# define X, y
stdscal = StandardScaler()
X = data.drop('Rings', axis=1)
y = data['Rings'] >= 10.
# one-hot encoding
X = pd.get_dummies(X, columns=['Sex'], prefix=['Sex'])
# standardize data
X.iloc[:, :-3] = stdscal.fit_transform(X.iloc[:, :-3])
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=123)

#algo = "gradient_boosting"
#algo = "KNN"
#algo = "SVC"
#algo = "Gaussian_process"
algo = "random_forest"

if algo == "gradient_boosting":
    # Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier()
    hyper_params = [{
        'n_estimators' : (50, 100),
        'learning_rate' : (0.05, 0.1),
        'max_depth' : (3, 5),
        'min_weight_fraction_leaf': (0.0, 0.25,)
    }]
elif algo == "KNN":
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    hyper_params = [{
        'n_neighbors' : (5, 8),
        'algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute'),
    }]
elif algo == "SVC":
    from sklearn.svm import SVC
    classifier = SVC(probability=True)
    hyper_params = [{
        'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
        'C' : (0.025, 0.5, 1.0)
    }]
elif algo == "Gaussian_process":
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    classifier = GaussianProcessClassifier()
    hyper_params = [{
        'kernel' : (1.0 * RBF(1.0),)
    }]
elif algo == "random_forest":
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    hyper_params = [{
        'max_depth' : (5, 10),
        'n_estimators' : (10, 20),
        'max_features' : (1, 2, 5)
    }]
else:
    sys.exit(0)

# grid search with cross-validation
grid_clf = GridSearchCV(classifier, cv=5, param_grid=hyper_params,
                        verbose=0, n_jobs=4, scoring='roc_auc')
grid_clf.fit(X_train,y_train.ravel())

sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))
train_score_roc = grid_clf.score(X_train, y_train)
test_score_roc = grid_clf.score(X_test, y_test)

# print results
out_text = '\t'.join([algo,
                      str(sorted_grid_params).replace('\n',','), 
                      str(train_score_roc), 
                      str(test_score_roc)])

print(out_text)

# plot ROC curve
fig, ax = plt.subplots()
y_pred = grid_clf.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
ax.plot([0, 1], [0, 1],'r--')
ax.legend(loc = 'lower right')
ax.set_title('Receiver Operating Characteristic')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.tight_layout()
plt.show()

