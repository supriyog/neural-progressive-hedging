import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


def get_data(start_date, end_date, stocks):
    pathname_for_stock = 'sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv'  ##must fill in symbol

    #ref_date = dt.date(2013, 2, 8)
    ref_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()

    abbr = []
    feats = []
    days = []
    for symb in stocks:
        x = pd.read_csv(pathname_for_stock.format(symb))
        mask = (x['date'] >= start_date) & (x['date'] <= end_date)
        x = x.loc[mask]
        z = {}
        z['day'] = np.array([(dt.datetime.strptime(s, '%Y-%m-%d').date() - ref_date).days for s in x['date']])
        z['open'] = x['open'].values
        z['high'] = x['high'].values
        z['low'] = x['low'].values
        z['close'] = x['close'].values
        z['volume'] = x['volume'].values
        f = np.array([z['day'], z['open'], z['high'], z['low'], z['close'], z['volume']]).T
        if np.isnan(f).any():
            print('skip file with NaN:', symb)
            continue
        feats.append(f)
        abbr.append(symb)
        if len(days) > 0:
            days = np.intersect1d(days, z['day'])
        else:
            days = z['day']

    prices = []
    for f in feats:
        idx = np.nonzero(np.isin(f[:, 0], days))[0]
        prices.append(f[idx, 1:])

    return np.array(prices), abbr


## select top num_stocks based on gain in the specified period
## return list of selected stock names
def get_data_filter(num_stocks, start_date, end_date, stocks=None):
    if not stocks:
        with open('sandp500_all_stock/abbr_all.pickle', 'rb') as fp:
            stocks = pickle.load(fp)
    hist,abbr=get_data(start_date,end_date,stocks)
    ## select top num_stocks based on gain
    idx = np.argsort(hist[:,-1,3]/hist[:,0,3])[-num_stocks:]
    return [abbr[i] for i in idx]

start_date = '2013-02-08'
end_date = '2016-12-30'
print(get_data_filter(100, start_date, end_date))