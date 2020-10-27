from fredapi import Fred
fred = Fred(api_key=MY_API_KEY)
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import VAR
import pandas as pd


# fetch data
def data_retriever1():
    """
    We only take data from 1983-01-01 to 2019-04-01
    """
    df = pd.DataFrame(columns = ['GDP', 'DGS3MO', 'DGS10', 'BBB', 'Inflation'])

    # US Nominal GDP, quarterly
    GDP_series = fred.get_series('GDP')[148:]
    # US 3M Treasury Bill, daily
    DGS3MO_series = fred.get_series('DGS3MO')[260:-133]
    # US 10Y Treasury Note, daily
    DGS10_series = fred.get_series('DGS10')[5479:-133]
    # BBB Corporate Spread, monthly
    BBB_series = fred.get_series('BAA10YM')[357:-5]
    # Inflation (%YoY of CPI), monthly
    Inflation_series = fred.get_series('CPIAUCSL')[432:-4]
    
    # helper function
    def get_value(series, itime):
        idx = int(round(series.size * itime))
        if (idx == series.size):
            idx -= 1
        return series.values[idx]
    
    data = []
    time = np.linspace(0, 1, GDP_series.size)
    for cnt, it in enumerate(time):
        GDP = GDP_series.iloc[cnt]
        DGS3MO = get_value(DGS3MO_series, it)
        DGS10 = get_value(DGS10_series, it)
        BBB = get_value(BBB_series, it)
        Inflation = get_value(Inflation_series, it)
        data.append([np.log(GDP), np.log(DGS3MO), DGS10, BBB, np.log(Inflation)])
            
    data = pd.DataFrame(data).diff().dropna()

    return data


def data_retriever2():
    """
    We only take data from 1983-01-01 to 2019-04-01
    """
    # US Nominal GDP, quarterly
    GDP_series = np.log(fred.get_series('GDP')).iloc[148:].diff().dropna()
    # US 3M Treasury Bill, daily
    DGS3MO_series = fred.get_series('DGS3MO').iloc[260:-133].diff().dropna()
    # US 10Y Treasury Note, daily
    DGS10_series = fred.get_series('DGS10').iloc[5479:-133].diff().dropna()
    # BBB Corporate Spread, monthly
    BBB_series = fred.get_series('BAA10YM').iloc[357:-5].diff().dropna()
    # Inflation (%YoY of CPI), monthly
    Inflation_series = np.log(fred.get_series('CPIAUCSL')).iloc[432:-4].diff().dropna()
    
    # dictionary for historical data, grouped by year
    GDP_dict = defaultdict(list)
    for time_stamp in GDP_series.index:
        year = str(time_stamp).split('-')[0]
        GDP_dict[year].append(GDP_series[time_stamp])
    # DGS3MO
    DGS3MO_dict = defaultdict(list)
    for time_stamp in DGS3MO_series.index:
        year = str(time_stamp).split('-')[0]
        DGS3MO_dict[year].append(DGS3MO_series[time_stamp])
    # DGS10
    DGS10_dict = defaultdict(list)
    for time_stamp in DGS10_series.index:
        year = str(time_stamp).split('-')[0]
        DGS10_dict[year].append(DGS10_series[time_stamp])
    # BBB
    BBB_dict = defaultdict(list)
    for time_stamp in BBB_series.index:
        year = str(time_stamp).split('-')[0]
        BBB_dict[year].append(BBB_series[time_stamp])
    # Inflation
    Inflation_dict = defaultdict(list)
    for time_stamp in Inflation_series.index:
        year = str(time_stamp).split('-')[0]
        Inflation_dict[year].append(Inflation_series[time_stamp])

    return GDP_dict, DGS3MO_dict, DGS10_dict, BBB_dict, Inflation_dict



if __name__ == "__main__":
    
    ### get data for vector autoregression analysis
    data = data_retriever1()

    model = VAR(data)
    # model selection
    model_selection = True
    if model_selection:
        for lag in range(11):
            results = model.fit(lag)
            print("AIC: {}, BIC: {}, HQC: {}".format(results.aic, results.bic, results.hqic))
    # prediction
    lag = 3
    results = model.fit(lag)
    pred = results.forecast(data.values[-lag:], 2)
    print(pred)
    print(results.summary())
    
    
    ### get more data
    GDP_dict, DGS3MO_dict, DGS10_dict, BBB_dict, Inflation_dict = data_retriever2()

    min_sum = np.inf
    min_year = -1
    for year in range(1983, 2019):
        year_sum = 0
        # GDP
        GDP_sum = 0
        for idx, val in enumerate(GDP_dict['2019']):
            GDP_sum += (GDP_dict[str(year)][idx] - val) ** 2
        year_sum += np.sqrt(GDP_sum / len(GDP_dict['2019']))
        # DGS3MO
        DGS3MO_sum = 0
        for idx, val in enumerate(DGS3MO_dict['2019']):
            DGS3MO_sum += (DGS3MO_dict[str(year)][idx] - val) ** 2
        year_sum += np.sqrt(DGS3MO_sum / len(DGS3MO_dict['2019']))
        # DGS10
        DGS10_sum = 0
        for idx, val in enumerate(DGS10_dict['2019']):
            DGS10_sum += (DGS10_dict[str(year)][idx] - val) ** 2
        year_sum += np.sqrt(DGS10_sum / len(DGS10_dict['2019']))
        # BBB
        BBB_sum = 0
        for idx, val in enumerate(BBB_dict['2019']):
            BBB_sum += (BBB_dict[str(year)][idx] - val) ** 2
        year_sum += np.sqrt(BBB_sum / len(BBB_dict['2019']))
        # Inflation
        Inflation_sum = 0
        for idx, val in enumerate(Inflation_dict['2019']):
            Inflation_sum += (Inflation_dict[str(year)][idx] - val) ** 2
        year_sum += np.sqrt(Inflation_sum / len(Inflation_dict['2019']))
        print(year_sum)
        if year_sum < min_sum:
            min_sum = year_sum
            min_year = year

    print(min_year)

