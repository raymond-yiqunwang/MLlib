import pandas as pd
import numpy as np
import matplotlib
# comment this line if not using Mac OSX
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
import operator


### Question 1: Loading data
def load_data():
    if os.path.isfile('transactions.csv'):
        # speed up reading csv file
        df = pd.read_csv('transactions.csv', header=0, index_col=None)
    else:
        df = pd.read_json('transactions.txt', lines=True)
        df.to_csv('transactions.csv', header=df.columns, index=False)

    # descriptions
    print("\n-- Number of records: {}, number of fields: {}\n".format(df.shape[0], df.shape[1]))
    # drop columns with all null values
    df = df.drop(['echoBuffer', 'merchantCity', 'merchantState',
                  'merchantZip', 'posOnPremises', 'recurringAuthInd'], axis=1)
    # show statistics       
    show_stats = ['availableMoney', 'creditLimit', 'currentBalance', 'transactionAmount']
    for icol in df.columns:
        print(icol, df[icol].dtypes)
        if icol in show_stats:
            print('  min: {:.2f}, max: {:.2f}, mean: {:.2f}, median: {:.2f}'.format(df[icol].min(), \
                                                df[icol].max(), df[icol].mean(), df[icol].median()))
        print('  number of null: {}\n'.format(df[icol].isna().sum()))

    # show categorical features
    print('acqCountry:', df['acqCountry'].unique())
    print('merchantCountryCode:', df['merchantCountryCode'].unique())
    print('cardPresent:', df['cardPresent'].unique())
    print('creditLimit:', df['creditLimit'].unique())
    print('expirationDateKeyInMatch:', df['expirationDateKeyInMatch'].unique())
    print('merchantCategoryCode:', df['merchantCategoryCode'].unique())
    print('transactionType:', df['transactionType'].unique())
    print('posEntryMode:', df['posEntryMode'].unique())
    print('posConditionCode:', df['posConditionCode'].unique())
    print('isFraud:', df['isFraud'].value_counts())

    return df


### Question 2: Plotting data
def plot_data(data):
    # histogram of the processed amounts of each transaction
    transaction_amount = data['transactionAmount']
    trans_data = transaction_amount[transaction_amount <= 600]
    trans_data.hist(bins=10)
    plt.show()
    plt.clf()

    # credit limit
    credit_data = data['creditLimit']
    credit_data.hist(bins=100)
    plt.show()
    plt.clf()

    # transaction time
    transaction_time = data['transactionDateTime']
    trans_counter = Counter()
    for ival in transaction_time.values:
        imonth = ival.split('-')[1]
        trans_counter[imonth] += 1
    trans = pd.DataFrame.from_dict(trans_counter, orient='index').reset_index()
    trans.plot(legend=False, xticks=None)
    plt.show()
    plt.clf()

    # merchant category code
    merchant_code = data['merchantCategoryCode']
    merch = data.groupby('merchantCategoryCode').size()
    merch.plot(kind='pie')
    plt.ylabel("")
    plt.show()
    plt.clf()


### Question 3: Detect duplicate transactions
def duplicate_detection(data):
    
    # filter out all duplicate transactions
    duplicate_features = [
        'accountNumber', 'accountOpenDate', 'acqCountry',
        'cardCVV', 'cardLast4Digits', 'cardPresent',
        'creditLimit', 'currentExpDate', 'customerId', 
        'dateOfLastAddressChange', 'enteredCVV',
        'expirationDateKeyInMatch', 'merchantCategoryCode',
        'merchantCountryCode', 'merchantName', 'posConditionCode',
        'posEntryMode', 'transactionAmount', 'transactionType'
    ]
    dups = data.duplicated(subset=duplicate_features, keep=False)
    
    relevant_features = ['accountNumber', 'availableMoney', 'currentBalance',
                         'transactionAmount', 'transactionDateTime', 'isFraud']
    data_dup = data[dups]
    
    data_debug = data[relevant_features]

    # reversed transactions
    reversed_idx = []
    for idx, irow in data_dup.iterrows():
        # do not consider 0.0 transaction amount
        if irow['transactionAmount'] == 0.: continue
        # loop over the duplicated records
        if (irow['transactionAmount'] == data_debug.iloc[idx+1]['transactionAmount']):
            if (irow['currentBalance'] == data_debug.iloc[idx+2]['currentBalance']) \
               and (irow['availableMoney'] > data_debug.iloc[idx+1]['availableMoney']):
                reversed_idx.append(idx)
    rev_trans = data_debug.iloc[reversed_idx]
    print('reversed transactions: {:.2f}'.format(rev_trans['accountNumber'].nunique() / float(len(reversed_idx))))
    print(rev_trans.describe())
    print(rev_trans['isFraud'].value_counts())
        
    # multi-swipe detection
    multiSwipe_idx = []
    nrows = data_dup.shape[0]
    start = 0
    while start < nrows:
        if data_dup.iloc[start]['transactionAmount'] == 0.:
            start += 1
            continue
        end = start 
        while True:
            if end >= nrows: break
            elif (data_dup['transactionAmount'].iloc[end] != data_dup['transactionAmount'].iloc[start]):
                break
            start_time = data_dup.iloc[start]['transactionDateTime'].split(':')
            end_time = data_dup.iloc[end]['transactionDateTime'].split(':')
            start_sec = int(start_time[1] * 60) + int(start_time[2])
            end_sec = int(end_time[1] * 60) + int(end_time[2])
            if (start_time[0] == end_time[0]) and (end_sec - start_sec) < 180:
                end += 1
            else:
                break
        if (end > start + 1):
            multiSwipe_idx += [i for i in range(start+1, end)]
            start = end 
        else:
            start += 1
    multi_trans = data_dup[relevant_features].iloc[multiSwipe_idx]
    print('multi swipe: {:.2f}'.format(multi_trans['accountNumber'].nunique() / float(len(multiSwipe_idx))))
    print(multi_trans.describe())
    print(multi_trans['isFraud'].value_counts())
            

### Question 4: Model construction for fraud detection
def train_classifier(data):
    
    # primitive features
    """
    # bool
        'expirationDateKeyInMatch', 'cardPresent',
    # categorical
        'merchantCategoryCode', 'merchantCountryCode', 'acqCountry',
        'posConditionCode', 'posEntryMode', 'transactionType',
    # numerical
        'availableMoney', 'currentBalance', 'transactionAmount', 'creditLimit'
    # TODO take time-dependent features into consideration
        'accountOpenDate', 'currentExpDate',
        'dateOfLastAddressChange', 'transactionDateTime'
    """

    data_raw = data[['availableMoney', 'currentBalance', 'transactionAmount', 'creditLimit']]
    data_raw['cvvMatch'] = data['enteredCVV'] == data['cardCVV']
    # one-hot encoding for categorical features
    new_expirationDateKeyInMatch = pd.get_dummies(data['expirationDateKeyInMatch'], dummy_na=True, prefix='expirationDateKeyInMatch')
    new_cardPresent = pd.get_dummies(data['cardPresent'], dummy_na=True, prefix='cardPresent')
    new_posConditionCode = pd.get_dummies(data['posConditionCode'], dummy_na=True, prefix='posConditionCode')
    new_posEntryMode = pd.get_dummies(data['posEntryMode'], dummy_na=True, prefix='posEntryMode')
    new_transactionType = pd.get_dummies(data['transactionType'], dummy_na=True, prefix='transactionType')
    new_merchantCategoryCode = pd.get_dummies(data['merchantCategoryCode'], dummy_na=True, prefix='merchantCategoryCode')
    is_fraud = data['isFraud'].apply(int)
    data_raw = pd.concat([data_raw, new_expirationDateKeyInMatch, new_cardPresent, new_posConditionCode,
                          new_merchantCategoryCode, new_posEntryMode, new_transactionType, is_fraud], axis=1)

    print(data_raw.shape)
    # train test split
    X = data_raw.drop('isFraud', axis=1).values
    y = data_raw['isFraud'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=5)

    # gradient tree boosting with grid search and 5-fold cross-validation
    hyper_params = [{
        'n_estimators' : (50, 100, 200,),
        'learning_rate' : (0.05, 0.1, 0.5,),
        'max_depth' : (3, 5, 8,),
        'min_weight_fraction_leaf': (0.0, 0.25,),
    }]
    classifier = GradientBoostingClassifier()
    grid_clf = GridSearchCV(classifier, cv=5, param_grid=hyper_params,
                            verbose=0, n_jobs=2, scoring='roc_auc')
    grid_clf.fit(X_train, y_train.ravel())

    sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))
    train_score_roc = grid_clf.score(X_train, y_train)
    test_score_roc = grid_clf.score(X_test, y_test)
    # print results
    out_text = '\t'.join(['gradient tree boosting',
                         str(sorted_grid_params).replace('\n',','), str(train_score_roc), str(test_score_roc)])
    print(out_text)

    # plot ROC curve
    y_pred = grid_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    
    ### Question 1: Loading Data
    data = load_data()

    ### Question 2: Ploting Data
    plot_data(data)

    ### Question 3: Duplicate Detection
    duplicate_detection(data)

    ### Question 4: Fraud Detection
    train_classifier(data)


