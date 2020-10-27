import operator
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns; sns.set()

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import *

import matplotlib
import matplotlib.pyplot as plt
# comment this line if using Mac OSX
matplotlib.use('TkAgg')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def read_and_inspect(ipath, muted=False):
    
    data_input = pd.read_csv(ipath)
    # remove duplicates
    data_input.drop_duplicates(inplace=True)

    if muted == True: return data_input

    # print intraday stock info
    for istock in ['STOCKA', 'STOCKB']:
        changes = data_input.loc[(data_input['symbol'] == istock) & (data_input['trdpx'] == 0)]
        print('Stock:', istock)
        print('\t num_changes:\t{:d}'.format(changes.shape[0]))
        # bid info
        print('\t bid range:\t({:.1f}, {:.1f})'.format(changes['bid'].min(axis=0), changes['bid'].max(axis=0)))
        print('\t bid mean:\t{:.1f}'.format(changes['bid'].mean(axis=0)))
        print('\t bid median:\t{:.1f}'.format(changes['bid'].median(axis=0)))
        print('\t bid std:\t{:.1f}'.format(changes['bid'].std(axis=0)))
        print('\t bidsz mean:\t{:.1f}'.format(changes['bidsz'].mean(axis=0)))
        # ask info
        print('\t ask range:\t({:.1f}, {:.1f})'.format(changes['ask'].min(axis=0), changes['ask'].max(axis=0)))
        print('\t ask mean:\t{:.1f}'.format(changes['ask'].mean(axis=0)))
        print('\t ask median:\t{:.1f}'.format(changes['ask'].median(axis=0)))
        print('\t ask std:\t{:.1f}'.format(changes['ask'].std(axis=0)))
        print('\t asksz mean:\t{:.1f}'.format(changes['asksz'].mean(axis=0)))
        print('\t changes/sec:\t{:.1f}'.format(changes.shape[0] // 23400))
        # trade info
        trades = data_input.loc[(data_input['symbol'] == istock) & (data_input['bid'] == 0)]
        print('\t num_trades:\t{:d}'.format(trades.shape[0]))
        print('\t trades/min:\t{:.1f}'.format(trades.shape[0] / 390))
        print('\n')

    return data_input


def pre_processing(data_input, symbol):
    data = data_input.loc[data_input['symbol'] == symbol]
    out = []

    t_window = 1E6
    # sweep through events, linear time operation
    start = 0
    itime = 0
    item_prev = [np.nan] * 7
    while start < data.shape[0]:
        end = start
        while data.iloc[end]['time'] < int(1.8E9)+(itime+1)*int(t_window):
            end += 1
            if (end >= data.shape[0] - 1): break
        if end == start:
            # no events in this time interval
            item = item_prev
        else:
            # changes in the book
            changes = data.iloc[start:end][data['trdpx'] == 0]
            if changes.empty:
                item = item_prev
            else:
                item = changes[['bid','bidsz','ask','asksz']].mean(axis=0).tolist()
            # trades
            trades = data.iloc[start:end][data['trdpx'] != 0]
            if trades.empty:
                item += [0]*3
            else:
                tradeinfo = trades[['trdpx','trdsz','trdsd']].mean(axis=0).tolist()
                relprice = (tradeinfo[0] - item[0]) / (item[2] - item[0])
                relsize = tradeinfo[1] / (item[1] + item[2])
                item += [relprice, relsize, round(tradeinfo[2])]
        out.append(item)
        item_prev = item
        itime += 1
        start = end
   
    features = ['P_bid', 'V_bid', 'P_ask', 'V_ask', 'P_trd', 'V_trd', 'Sd_trd']
    out = pd.DataFrame(out, columns = features)
    out.to_csv("./data_processed.csv", columns=features)
    return out


def run_linear_regression(X, y):

    hyper_params = [{
        'alpha': (0.01, 0.1, 1,),
    }]
   
   # linear regression model
    regressor = linear_model.Ridge(fit_intercept=False)
    grid_clf = GridSearchCV(regressor, cv=5, param_grid=hyper_params,
                            verbose=0,n_jobs=4,scoring='r2')
    grid_clf.fit(X,y.ravel())

    # print results
    r2_train = r2_score(y, grid_clf.predict(X))
    sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))
    outtxt = '\t'.join(['best params:', str(sorted_grid_params), 'r2-train:', str(r2_train)])
    return (grid_clf, r2_train, outtxt)


def feature_selection(features):
    
    select_horizon = True
    select_window = False

    # single point features
    features['P_mid'] = (features['P_ask'] + features['P_bid']) / 2
    features['P_spread'] = features['P_ask'] - features['P_bid']
    features['P_sp_ovr_mid'] = features['P_spread'] / features['P_mid']
    features['V_mid'] = (features['V_ask'] + features['V_bid']) / 2

    ### plot prediction horizon VS each primitive feature
    if select_horizon:
        pred_horizon = [10, 20, 40, 60, 90]
        look_back = [10, 20, 40, 60, 90, 120]
        fig = plt.figure(figsize = (16,8))
        max_r2 = 0
        min_r2 = 0
        r2_scores = []
        for idx, ifeat in enumerate(features.columns):
            level1 = []
            for ihorizon in pred_horizon:
                level2 = []
                for ilook_back in look_back:
                    tmpfeat = features.copy()
                    # get ride of unit for price features
                    pre_shift = tmpfeat.shift(periods=ilook_back, axis=0)
                    non_unit = (tmpfeat - pre_shift) / pre_shift
                    for pfeat in ['P_bid', 'P_ask', 'P_mid', 'P_spread']:
                        tmpfeat[pfeat] = non_unit[pfeat]
                    # feature shift
                    shift_horizon = tmpfeat.shift(periods=ilook_back, axis=0)
                    # target
                    shift_horizon['target'] = tmpfeat['P_mid'].shift(periods=(-1)*ihorizon, axis=0)
                    # prediction
                    out = shift_horizon.dropna()
                    X = out[ifeat].values.reshape(-1, 1)
                    y = out['target'].values.reshape(-1, 1)
                    _, r2_train, _ = run_linear_regression(X, y)
                    if r2_train > max_r2: max_r2 = r2_train
                    if r2_train < min_r2: min_r2 = r2_train
                    level2.append(r2_train)
                level1.append(level2)
            r2_scores.append(level1)
        # plot
        for idx, ihorizon in enumerate(r2_scores):
            plt.subplot(3, 4, idx+1)
            plt.ylim(min_r2-0.002, max_r2+0.002)
            for it, ilook in enumerate(ihorizon):
                handler = plt.scatter(look_back, ilook, s=30, label='pred_horizon={:d}s'.format(pred_horizon[it]))
                plt.title(features.columns[idx])
            if idx == 3: plt.legend(loc='upper right')
            if idx > 6:
                plt.xlabel('look back (s)')
                plt.xticks()
            else:
                plt.xticks([])
            if idx%4 == 0:
                plt.ylabel('r-squared')
                plt.yticks()
            else:
                plt.yticks([])
        plt.show()
        plt.clf()
    

    ### TODO select optimal sliding window size
    if select_window:
        pred_horizon = 60
        tmp_features = features.copy()
        # get ride of unit for price features
        pre_shift = tmp_features['P_bid'].shift(periods=90, axis=0)
        tmp_features['P_bid'] = (tmp_features['P_bid'] - pre_shift) / pre_shift
        pre_shift = tmp_features['P_ask'].shift(periods=120, axis=0)
        tmp_features['P_ask'] = (tmp_features['P_ask'] - pre_shift) / pre_shift
        pre_shift = tmp_features['P_mid'].shift(periods=120, axis=0)
        tmp_features['P_mid'] = (tmp_features['P_mid'] - pre_shift) / pre_shift
        pre_shift = tmp_features['P_spread'].shift(periods=40, axis=0)
        tmp_features['P_spread'] = (tmp_features['P_spread'] - pre_shift) / pre_shift
        tmp_features['target'] = tmp_features['P_mid']

        # feature shifting
        tmp_features['P_bid'] = tmp_features['P_bid'].shift(periods=90, axis=0)
        tmp_features['V_bid'] = tmp_features['V_bid'].shift(periods=90, axis=0)
        tmp_features['P_ask'] = tmp_features['P_ask'].shift(periods=120, axis=0)
        tmp_features['V_ask'] = tmp_features['P_ask'].shift(periods=10, axis=0)
        tmp_features['P_trd'] = tmp_features['P_trd'].shift(periods=10, axis=0)
        tmp_features['V_trd'] = tmp_features['V_trd'].shift(periods=10, axis=0)
        tmp_features['Sd_trd'] = tmp_features['Sd_trd'].shift(periods=10, axis=0)
        tmp_features['P_mid'] = tmp_features['P_mid'].shift(periods=120, axis=0)
        tmp_features['P_spread'] = tmp_features['P_spread'].shift(periods=40, axis=0)
        tmp_features['P_sp_ovr_mid'] = tmp_features['P_sp_ovr_mid'].shift(periods=40, axis=0)
        tmp_features['V_mid'] = tmp_features['V_mid'].shift(periods=60, axis=0)
        
        # target
        target = tmp_features['target'].shift(periods=(-1)*pred_horizon, axis=0)
        # composite features
        window_sizes = [1, 3, 5, 8, 12, 18]
        results = []
        for iwindow in window_sizes:
            composite_mean = tmp_features.rolling(window=iwindow).mean()
            composite_mean['target'] = target
            out = composite_mean.dropna()
            X = out.drop('target', axis=1).values
            y = out['target'].values.reshape(-1, 1)
            _, r2_train, outtxt = run_linear_regression(X, y)
            results.append(r2_train)
        print('sliding window size:', results) 


def gen_features(data, pred_horizon, draw_corr=False):
    
    features = data.copy()
        
    # single point features
    features['P_mid'] = (features['P_ask'] + features['P_bid']) / 2
    features['P_spread'] = features['P_ask'] - features['P_bid']
    features['P_sp_ovr_mid'] = features['P_spread'] / features['P_mid']
    features['V_mid'] = (features['V_ask'] + features['V_bid']) / 2
    
    # get ride of unit for price features
    pre_shift = features['P_bid'].shift(periods=90, axis=0)
    features['P_bid'] = (features['P_bid'] - pre_shift) / pre_shift
    pre_shift = features['P_ask'].shift(periods=120, axis=0)
    features['P_ask'] = (features['P_ask'] - pre_shift) / pre_shift
    pre_shift = features['P_mid'].shift(periods=120, axis=0)
    features['P_mid'] = (features['P_mid'] - pre_shift) / pre_shift
    pre_shift = features['P_spread'].shift(periods=40, axis=0)
    features['P_spread'] = (features['P_spread'] - pre_shift) / pre_shift
    features['target'] = features['P_mid']

    # feature shifting
    features['P_bid'] = features['P_bid'].shift(periods=90, axis=0)
    features['V_bid'] = features['V_bid'].shift(periods=90, axis=0)
    features['P_ask'] = features['P_ask'].shift(periods=120, axis=0)
    features['V_ask'] = features['P_ask'].shift(periods=10, axis=0)
    features['P_trd'] = features['P_trd'].shift(periods=10, axis=0)
    features['V_trd'] = features['V_trd'].shift(periods=10, axis=0)
    features['Sd_trd'] = features['Sd_trd'].shift(periods=10, axis=0)
    features['P_mid'] = features['P_mid'].shift(periods=120, axis=0)
    features['P_spread'] = features['P_spread'].shift(periods=40, axis=0)
    features['P_sp_ovr_mid'] = features['P_sp_ovr_mid'].shift(periods=40, axis=0)
    features['V_mid'] = features['V_mid'].shift(periods=60, axis=0)

    # target
    features['target'] = features['target'].shift(periods=(-1)*pred_horizon, axis=0)

    # output
    out = features.dropna()
    X = out.drop('target', axis=1).values
    y = out['target'].values.reshape(-1, 1)
    
    # correlation heatmap
    if draw_corr:
        corr = out.corr()
        fig = plt.figure(num=None, figsize=(40, 40), dpi=70, facecolor='w', edgecolor='w')
        colormap = sns.diverging_palette(220, 20, as_cmap=True)
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        ylabel = list(corr.columns)
        ylabel[0] = ''
        ax = sns.heatmap(corr, vmin=-0.6, cmap=colormap, mask=mask, annot=True, fmt=".2f", xticklabels=corr.columns[:-1], yticklabels=ylabel)
        plt.title(label="Correlation Heatmap of All Features", loc='center', fontdict={'fontname':'DejaVu Sans', 'size':'24', 'color':'black', 'weight':'bold',
                      'verticalalignment':'bottom'})
        ax.tick_params(axis='x', rotation=60, labelsize=13)
        ax.tick_params(axis='y', rotation=0, labelsize=13)
        plt.show()
        fig.savefig("./heatmap_mask.png", dpi=120)
    
    return X, y


def main(): 
    ### data inspection
    data_raw = read_and_inspect("./data2.csv", False)

    ### pre-processing
    if os.path.isfile("./data_processedA.csv"):
        data_processedA = pd.read_csv("./data_processedA.csv", index_col=0)
    else:
        data_processedA = pre_processing(data_raw, 'STOCKA')
    if os.path.isfile("./data_processedB.csv"):
        data_processedB = pd.read_csv("./data_processedB.csv", index_col=0)
    else:
        data_processedB = pre_processing(data_raw, 'STOCKB')

    # A data
    total_size = data_processedA.shape[0]
    ntrain = int(total_size*0.75)
    training_data_A = data_processedA[:ntrain].copy()
    test_data_A = data_processedA[ntrain:].copy()
    # B data
    total_size = data_processedB.shape[0]
    ntrain = int(total_size*0.75)
    training_data_B = data_processedB[:ntrain].copy()
    test_data_B = data_processedB[ntrain:].copy()

    ### feature selection
    if True:
        feature_selection(training_data_A)
        feature_selection(training_data_B)
    
    for pred_horizon in [10, 20, 30, 60, 120, 180]:
        print("prediction horizon:", pred_horizon)

        # generate optimal features
        X_train_A, y_train_A = gen_features(training_data_A, pred_horizon)
        X_train_B, y_train_B = gen_features(training_data_B, pred_horizon)
        X_test_A, y_test_A = gen_features(test_data_A, pred_horizon)
        X_test_B, y_test_B = gen_features(test_data_B, pred_horizon)
    
        # correlation between two stocks
        print(np.corrcoef(y_train_A.ravel(), y_train_B.ravel()))
    
        # print results
        modelA, r2_train_A, outtxt = run_linear_regression(X_train_A, y_train_A)
        modelB, r2_train_B, outtxt = run_linear_regression(X_train_B, y_train_B)
        modelA_crossB, r2_crosstrain_A, outtxt = run_linear_regression(X_train_A, y_train_B)
        modelB_crossA, r2_crosstrain_B, outtxt = run_linear_regression(X_train_B, y_train_A)
        
        # model performance
        r2_test_A = r2_score(y_test_A, modelA.predict(X_test_A))
        r2_test_B = r2_score(y_test_B, modelB.predict(X_test_B))
        r2_crosstest_A = r2_score(y_test_B, modelA_crossB.predict(X_test_A))
        r2_crosstest_B = r2_score(y_test_A, modelB_crossA.predict(X_test_B))
        
        print("\ttraining set:", r2_train_A, r2_train_B, r2_crosstrain_A, r2_crosstrain_B)
        print("\ttesting set:", r2_test_A, r2_test_B, r2_crosstest_A, r2_crosstest_B)


if __name__ == "__main__":
    main()


