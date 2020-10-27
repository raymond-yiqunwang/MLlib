import numpy as np
import pandas as pd
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # comment this line if not using Mac OSX


def compute_ret_mkt(data):
    # compute market returns
    wret = data['weight'] * data['ret']
    wret_sum = wret.sum(level=0)
    sum_weight = data['weight'].sum(level=0)
    ret_mkt = wret_sum / sum_weight
    return ret_mkt


def update_var(ix, iy, sum_w, wx, wxx, wy, wxy, weight_decay):
    # update variables
    sum_w = weight_decay * (sum_w + 1)
    wx = weight_decay * (wx + ix)
    wxx = weight_decay * (wxx + ix*ix)
    wy = weight_decay * (wy + iy)
    wxy = weight_decay * (wxy + ix*iy)
    return sum_w, wx, wxx, wy, wxy


def compute_beta(x, y, halflife):
    # skip NaN from start
    istart = 0
    while (np.isnan(x.iloc[istart]) or np.isnan(y.iloc[istart])):
        istart += 1
    out = [np.nan] * istart
    # init var
    weight_decay = np.power(2., -1./halflife)
    sum_w = weight_decay
    sx, sy = x.iloc[istart], y.iloc[istart]
    wx = wxx = wy = wxy = 0
    cutoff = 200 
    for i in range(istart, y.shape[0]):
        if (np.isnan(x.iloc[i]) or np.isnan(y.iloc[i])):
            # skip NaN at the end
            out.append(np.nan)
            continue
        elif (i < istart + cutoff):
            # discard the first 200 data for a robust linear regression model
            out.append(np.nan)
        else:
            # compute beta
            beta = (wxy - wx*wy/sum_w) / (wxx - wx*wx/sum_w)
            out.append(beta)
        # update variables
        ix, iy = x.iloc[i], y.iloc[i]
        sum_w, wx, wxx, wy, wxy = update_var(ix, iy, sum_w, wx, wxx, wy, wxy, weight_decay)
    return out


def stock_selection(data, ret_mkt, stock_names, halflife=120):
    # use benchmark beta120 for stock selection
    results = []
    for stock in stock_names:
        ret_stock = data['ret'].iloc[data.index.get_level_values('uspn') == stock]
        beta = compute_beta(ret_mkt, ret_stock, halflife)
        results.append((stock, np.nanmean(beta)))
    # sort stocks w.r.t. their absolute value of beta
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:12], results[-12:]


def compute_error_metrics(y, yhat):
    ybar = np.mean(y)
    SSres = np.sum((y - yhat)**2)
    SStot = np.sum((y - ybar)**2)
    # compute r-squared
    r2 = 1 - SSres / SStot
    # compute RMSE
    rmse = np.sqrt(SSres / len(y))
    # compute MAE
    mae = np.mean(abs(y - yhat))
    return r2, rmse, mae


def model_selection(data, ret_mkt, stocks):
    # hyperparameters
    halflife_list = [30, 60, 90, 120, 180]
    window_sz_list = [1, 5, 15, 30]
    fig = plt.figure(figsize = (16,8))
    r2_scores = []
    for stock in stocks:
        level1 = []
        for window_size in window_sz_list:
            level2 = []
            for halflife in halflife_list:
                # market returns with rolling mean
                ret_mkt_rolling = ret_mkt.rolling(window=window_size).mean()
                ret_stock = data['ret'].iloc[data.index.get_level_values('uspn') == stock]
                ret_stock = ret_stock.rolling(window=window_size).mean()
                beta = compute_beta(ret_mkt_rolling, ret_stock, halflife)
                # TODO ugly way to drop NaN, to be optimized
                rets = pd.DataFrame(ret_mkt_rolling * beta, columns=['yhat'])
                rets['y'] = ret_stock.tolist()
                rets = rets.dropna()
                r2, rmse, mae = compute_error_metrics(rets['y'], rets['yhat'])
                level2.append(r2)
            level1.append(level2)
        r2_scores.append(level1)
    # customized plot
    for idx, window_size in enumerate(r2_scores):
        plt.subplot(3, 4, idx+1)
        imax = 0
        imin = 1
        for it, halflife in enumerate(window_size):
            handler = plt.scatter(halflife_list, halflife, s=30,
                                  label='window_sz={:d}d'.format(window_sz_list[it]))
            plt.title(stocks[idx])
            if max(halflife) > imax: imax = max(halflife)
            if min(halflife) < imin: imin = min(halflife)
        plt.ylim(imin-0.03, imax+0.03)
        if idx > 7:
            plt.xlabel('halflife (d)')
            plt.xticks()
        else:
            plt.xticks([])
        if idx%4 == 0:
            plt.ylabel('r-squared')
        plt.yticks()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # sys args
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--select_stocks', default=False, type=str,
                        help='select True to print 12 stocks with highest beta120 values')
    args = parser.parse_args()

    # read data
    data = pd.read_hdf('input_data.h5', model='r')
    
    # compute market returns
    ret_mkt = compute_ret_mkt(data)

    # select 12 stocks with largest absolute beta_120 value
    if args.select_stocks:
        stock_names = data.index.levels[1]
        # use benchmark model beta120
        halflife = 120
        stocks_highBeta, stocks_lowBeta = stock_selection(data, ret_mkt, stock_names, halflife)
        print('12 stocks with highest beta120 values are: \n', stocks_highBeta)
        print('\n12 stocks with lowest beta120 values are: \n', stocks_lowBeta)
        
    # hyperparameter tuning
    stocks = [ 'SWKS', 'NVDA', 'AKAM', 'AMD', 'MU', 'MS',
               'NTAP', 'LRCX', 'CBRE', 'AMZN', 'SCHW', 'IVZ' ]
    model_selection(data, ret_mkt, stocks)


